#!/usr/bin/env python3
"""Numerical IK for SO101 (robust LM, 2-stage: position then configurable orientation refine)."""

from __future__ import annotations
import numpy as np
from so101_forward_kinematics import fk_from_q_vector_deg

JOINT_ORDER = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.83, 96.83),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.21, 162.79),
    "gripper": (0.0, 100.0),
}

LIMITS_RAD = np.deg2rad(np.array([JOINT_LIMITS_DEG[k] for k in JOINT_ORDER], dtype=float))


def wrap_rad(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_R_rad(R: np.ndarray) -> float:
    # world yaw from x-axis projection
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v.copy()
    return v / n


def clamp_q_rad(q_rad: np.ndarray) -> np.ndarray:
    q = np.array(q_rad, dtype=float).reshape(5).copy()
    q = np.clip(q, LIMITS_RAD[:, 0], LIMITS_RAD[:, 1])
    return q


def config_to_q_rad(config: dict) -> np.ndarray:
    return np.deg2rad(np.array([float(config[k]) for k in JOINT_ORDER], dtype=float))


def q_rad_to_config(q_rad: np.ndarray, gripper: float = 50.0) -> dict[str, float]:
    q_deg = np.rad2deg(np.array(q_rad, dtype=float).reshape(5))
    return {
        "shoulder_pan": float(q_deg[0]),
        "shoulder_lift": float(q_deg[1]),
        "elbow_flex": float(q_deg[2]),
        "wrist_flex": float(q_deg[3]),
        "wrist_roll": float(q_deg[4]),
        "gripper": float(np.clip(gripper, *JOINT_LIMITS_DEG["gripper"])),
    }


def fk_qrad(q_rad: np.ndarray):
    q_deg = np.rad2deg(np.array(q_rad, dtype=float).reshape(5))
    return fk_from_q_vector_deg(q_deg)  # returns (p, R, T)


def task_error(
    q_rad: np.ndarray,
    target_p: np.ndarray,
    yaw_des_rad: float | None,
    orient_mode: str = "none",
    target_axis: np.ndarray | None = None,
    axis_index: int = 2,
):
    p, R, _ = fk_qrad(q_rad)
    e_pos = target_p - p
    if orient_mode == "yaw":
        yaw_cur = yaw_from_R_rad(R)
        e_yaw = wrap_rad(yaw_des_rad - yaw_cur)
        e = np.array([e_pos[0], e_pos[1], e_pos[2], e_yaw], dtype=float)
    elif orient_mode == "axis":
        if target_axis is None:
            raise ValueError("target_axis is required when orient_mode='axis'")
        axis_cur = _normalize(R[:, axis_index])
        axis_des = _normalize(np.array(target_axis, dtype=float).reshape(3))
        e_axis = axis_des - axis_cur
        e = np.array([e_pos[0], e_pos[1], e_pos[2], e_axis[0], e_axis[1], e_axis[2]], dtype=float)
    else:
        e = np.array([e_pos[0], e_pos[1], e_pos[2]], dtype=float)
    return e, p, R


def numerical_jacobian(
    q_rad: np.ndarray,
    target_p: np.ndarray,
    yaw_des_rad: float | None,
    orient_mode: str = "none",
    target_axis: np.ndarray | None = None,
    axis_index: int = 2,
    eps_rad: float = 1e-3,
):
    # central difference Jacobian wrt radians
    if orient_mode == "yaw":
        m = 4
    elif orient_mode == "axis":
        m = 6
    else:
        m = 3
    J = np.zeros((m, 5), dtype=float)

    for i in range(5):
        qp = q_rad.copy()
        qm = q_rad.copy()
        qp[i] += eps_rad
        qm[i] -= eps_rad

        ep, _, _ = task_error(
            qp,
            target_p,
            yaw_des_rad,
            orient_mode=orient_mode,
            target_axis=target_axis,
            axis_index=axis_index,
        )
        em, _, _ = task_error(
            qm,
            target_p,
            yaw_des_rad,
            orient_mode=orient_mode,
            target_axis=target_axis,
            axis_index=axis_index,
        )
        de = em - ep
        if orient_mode == "yaw":
            de[3] = wrap_rad(em[3] - ep[3])
        J[:, i] = de / (2.0 * eps_rad)

    return J


def _cost(e: np.ndarray, orient_mode: str = "none") -> float:
    pos_cost = np.linalg.norm(e[:3])
    if orient_mode == "yaw":
        return pos_cost + 0.03 * abs(e[3])  # small yaw penalty (rad)
    if orient_mode == "axis":
        return pos_cost + 0.08 * np.linalg.norm(e[3:])  # soft upright penalty
    return pos_cost


def _solve_stage(
    q0_rad: np.ndarray,
    target_p: np.ndarray,
    yaw_des_rad: float | None,
    orient_mode: str,
    target_axis: np.ndarray | None,
    axis_index: int,
    max_iters: int,
    pos_tol: float,
    yaw_tol_rad: float,
    axis_tol: float,
    pos_weight: float,
    yaw_weight: float,
    axis_weight: float,
    lambda0: float,
    max_step_deg: float,
):
    q = clamp_q_rad(q0_rad)
    lam = float(lambda0)
    max_step = np.deg2rad(max_step_deg)

    if orient_mode == "yaw":
        W = np.diag([pos_weight, pos_weight, pos_weight, yaw_weight])
    elif orient_mode == "axis":
        W = np.diag([pos_weight, pos_weight, pos_weight, axis_weight, axis_weight, axis_weight])
    else:
        W = np.diag([pos_weight, pos_weight, pos_weight])

    best_q = q.copy()
    e, p, _ = task_error(
        q,
        target_p,
        yaw_des_rad,
        orient_mode=orient_mode,
        target_axis=target_axis,
        axis_index=axis_index,
    )
    best_cost = _cost(e, orient_mode=orient_mode)

    for _ in range(max_iters):
        e, p, _ = task_error(
            q,
            target_p,
            yaw_des_rad,
            orient_mode=orient_mode,
            target_axis=target_axis,
            axis_index=axis_index,
        )
        pos_err = np.linalg.norm(e[:3])
        yaw_err = abs(e[3]) if orient_mode == "yaw" else 0.0
        axis_err = np.linalg.norm(e[3:]) if orient_mode == "axis" else 0.0

        c = _cost(e, orient_mode=orient_mode)
        if c < best_cost:
            best_cost = c
            best_q = q.copy()

        if pos_err < pos_tol and orient_mode == "yaw" and yaw_err < yaw_tol_rad:
            return q
        if pos_err < pos_tol and orient_mode == "axis" and axis_err < axis_tol:
            return q
        if pos_err < pos_tol and orient_mode == "none":
            return q

        J = numerical_jacobian(
            q,
            target_p=target_p,
            yaw_des_rad=yaw_des_rad,
            orient_mode=orient_mode,
            target_axis=target_axis,
            axis_index=axis_index,
            eps_rad=1e-3,
        )

        Jw = W @ J
        ew = W @ e

        # LM normal equations in joint space
        H = Jw.T @ Jw + lam * np.eye(5)
        g = Jw.T @ ew

        try:
            dq = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            lam *= 10.0
            continue

        # step clamp
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= (max_step / n)

        # backtracking line search
        improved = False
        q_best_local = q
        c_best_local = c
        for s in (1.0, 0.5, 0.25, 0.1, 0.05):
            qt = clamp_q_rad(q + s * dq)
            et, _, _ = task_error(
                qt,
                target_p,
                yaw_des_rad,
                orient_mode=orient_mode,
                target_axis=target_axis,
                axis_index=axis_index,
            )
            ct = _cost(et, orient_mode=orient_mode)
            if ct < c_best_local:
                q_best_local = qt
                c_best_local = ct
                improved = True
                break

        if improved:
            q = q_best_local
            lam = max(lam * 0.5, 1e-8)
        else:
            lam = min(lam * 5.0, 1e3)

    return best_q


def get_inverse_kinematics(
    target_position,
    target_orientation=None,
    initial_guess=None,
    gripper: float = 50.0,
    max_iters: int = 160,
    damping: float = 1e-3,      # much smaller than your old 0.03
    pos_tol: float = 0.003,     # 3 mm
    yaw_tol_deg: float = 8.0,
    axis_tol_deg: float = 1.0,
    pos_weight: float = 1.0,
    yaw_weight: float = 0.03,   # keep yaw weak so position dominates
    axis_weight: float = 0.35,
    orientation_mode: str = "axis",
    axis_to_align: str = "x",
):
    target_p = np.array(target_position, dtype=float).reshape(3)
    if target_orientation is None:
        # For position-only targets, default to a claw-like downward approach.
        yaw_des = 0.0
        z_des = np.array([0.0, 0.0, -1.0], dtype=float)
    else:
        target_R = np.array(target_orientation, dtype=float).reshape(3, 3)
        yaw_des = yaw_from_R_rad(target_R)
        z_des = target_R[:, 2]
    z_des = _normalize(z_des)
    axis_idx_map = {"x": 0, "y": 1, "z": 2}
    if axis_to_align not in axis_idx_map:
        raise ValueError("axis_to_align must be one of: 'x', 'y', 'z'")
    axis_index = axis_idx_map[axis_to_align]

    if initial_guess is None:
        q0 = np.deg2rad(np.array([0.0, 30.0, -60.0, 40.0, 0.0], dtype=float))
    elif isinstance(initial_guess, dict):
        q0 = config_to_q_rad(initial_guess)
    else:
        q0 = np.deg2rad(np.array(initial_guess, dtype=float).reshape(5))
    q0 = clamp_q_rad(q0)

    # Stage 1: position only (robustly hit xyz)
    q_pos = _solve_stage(
        q0_rad=q0,
        target_p=target_p,
        yaw_des_rad=None,
        orient_mode="none",
        target_axis=None,
        axis_index=axis_index,
        max_iters=max_iters,
        pos_tol=pos_tol,
        yaw_tol_rad=np.deg2rad(yaw_tol_deg),
        axis_tol=2.0 * np.sin(0.5 * np.deg2rad(axis_tol_deg)),
        pos_weight=pos_weight,
        yaw_weight=yaw_weight,
        axis_weight=axis_weight,
        lambda0=damping,
        max_step_deg=6.0,
    )

    if orientation_mode not in ("none", "yaw", "axis"):
        raise ValueError("orientation_mode must be one of: 'none', 'yaw', 'axis'")

    # Stage 2: orientation refinement from position solution
    stage2_mode = orientation_mode
    stage2_yaw_des = yaw_des if stage2_mode == "yaw" else None
    stage2_axis = z_des if stage2_mode == "axis" else None
    q_full = _solve_stage(
        q0_rad=q_pos,
        target_p=target_p,
        yaw_des_rad=stage2_yaw_des,
        orient_mode=stage2_mode,
        target_axis=stage2_axis,
        axis_index=axis_index,
        max_iters=max_iters // 2,
        pos_tol=pos_tol,
        yaw_tol_rad=np.deg2rad(yaw_tol_deg),
        axis_tol=2.0 * np.sin(0.5 * np.deg2rad(axis_tol_deg)),
        pos_weight=pos_weight,
        yaw_weight=yaw_weight,
        axis_weight=axis_weight,
        lambda0=damping,
        max_step_deg=4.0,
    )

    # Evaluate both candidates in stage-2 objective space.
    e_from_pos, _, _ = task_error(
        q_pos,
        target_p,
        stage2_yaw_des,
        orient_mode=stage2_mode,
        target_axis=stage2_axis,
        axis_index=axis_index,
    )
    e_from_full, _, _ = task_error(
        q_full,
        target_p,
        stage2_yaw_des,
        orient_mode=stage2_mode,
        target_axis=stage2_axis,
        axis_index=axis_index,
    )
    if stage2_mode == "none":
        q_final = q_pos
    else:
        pos_from_pos = np.linalg.norm(e_from_pos[:3])
        pos_from_full = np.linalg.norm(e_from_full[:3])
        cost_from_pos = _cost(e_from_pos, orient_mode=stage2_mode)
        cost_from_full = _cost(e_from_full, orient_mode=stage2_mode)
        # Keep orientation-refined solution when it improves stage-2 objective,
        # allowing a small position tradeoff.
        if cost_from_full <= cost_from_pos and pos_from_full <= pos_from_pos + 0.004:
            q_final = q_full
        else:
            q_final = q_pos

    return q_rad_to_config(q_final, gripper=gripper)
