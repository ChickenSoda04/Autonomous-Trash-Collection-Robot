#!/usr/bin/env python3
"""Real-robot IK pick-and-place sequence for SO101."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

_MUJOCO_SO101_DIR = Path(__file__).resolve().parents[1] / "mujoco_so101"
if _MUJOCO_SO101_DIR.exists():
    sys.path.insert(0, str(_MUJOCO_SO101_DIR))

from so101_forward_kinematics import Rx, Rz, get_forward_kinematics
from so101_inverse_kinematics import get_inverse_kinematics

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
GRIP = "gripper"


def parse_args():
    p = argparse.ArgumentParser(description="SO101 real-robot IK pick-and-place")
    p.add_argument("--port-id", type=str, default="/dev/tty.usbmodem5AB01824781")
    p.add_argument("--robot-name", type=str, default="follower")

    p.add_argument("--pick-x", type=float, default=0.02)
    p.add_argument("--pick-y", type=float, default=-0.24)
    p.add_argument("--pick-z", type=float, default=-0.11)
    p.add_argument("--place-x", type=float, default=0.02)
    p.add_argument("--place-y", type=float, default=0.2)
    p.add_argument("--place-z", type=float, default=0.14)

    p.add_argument("--tool-yaw-deg", type=float, default=0.0)
    p.add_argument(
        "--tcp-grasp-offset-x",
        type=float,
        default=0.014,
        # default=0.0,
        help="Tool-local +x offset (m) from TCP frame to grasp center between fingers.",
    )
    p.add_argument("--tcp-grasp-offset-y", type=float, default=0.0)
    p.add_argument("--tcp-grasp-offset-z", type=float, default=0.0)
    p.add_argument(
        "--approach-height",
        type=float,
        default=0.032,
        help="Shared fallback height offset (m) used if a pick/place-specific height is not set.",
    )
    p.add_argument(
        "--pick-approach-height",
        type=float,
        default=0.062,
        help="Height offset (m) above the pick target before descending. Defaults to --approach-height.",
    )
    p.add_argument(
        "--place-retreat-height",
        type=float,
        default=None,
        help="Height offset (m) above the place target after opening. Defaults to --approach-height.",
    )
    p.add_argument("--lift-height", type=float, default=0.0)
    p.add_argument(
        "--approach-steps",
        type=int,
        default=3,
        help="Number of IK sub-steps for segmented pick approach descent (>=1)",
    )
    p.add_argument(
        "--approach-duration-scale",
        type=float,
        default=0.6,
        help="Scale factor for pick approach duration relative to --move-duration",
    )
    p.add_argument(
        "--descend-steps",
        type=int,
        default=6,
        help="Number of IK sub-steps for segmented retreat motions (>=1)",
    )
    p.add_argument(
        "--descend-duration-scale",
        type=float,
        default=1.4,
        help="Scale factor for total descent/retreat duration relative to --move-duration",
    )
    p.add_argument(
        "--segmented-descend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use multi-step IK for pick approach descent and post-drop retreat",
    )

    p.add_argument("--gripper-open-pct", type=float, default=70.0)
    p.add_argument("--gripper-close-pct", type=float, default=3.0)
    p.add_argument(
        "--post-close-hold",
        type=float,
        default=1.2,
        help="Extra hold after close command before lifting",
    )
    p.add_argument(
        "--gripper-settle-timeout",
        type=float,
        default=1.5,
        help="Max wait for gripper position to stabilize after close",
    )
    p.add_argument(
        "--gripper-stable-eps",
        type=float,
        default=0.6,
        help="Stability threshold in gripper percent units",
    )
    p.add_argument(
        "--gripper-stable-cycles",
        type=int,
        default=4,
        help="Consecutive stable samples required to consider gripper settled",
    )
    p.add_argument(
        "--gripper-close-tol",
        type=float,
        default=3.0,
        help="Allowed error (percent units) between measured and close target before lift",
    )
    p.add_argument(
        "--gripper-close-retries",
        type=int,
        default=2,
        help="Extra close attempts before giving up",
    )

    p.add_argument("--move-duration", type=float, default=2.2)
    p.add_argument("--gripper-duration", type=float, default=1.0)
    p.add_argument("--hold-seconds", type=float, default=0.8)
    p.add_argument("--final-hold", type=float, default=1.5)
    p.add_argument("--print-feedback", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--confirm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--return-home", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--x-min", type=float, default=0.00)
    p.add_argument("--x-max", type=float, default=0.27)
    p.add_argument("--y-min", type=float, default=-0.25)
    p.add_argument("--y-max", type=float, default=0.25)
    p.add_argument("--z-min", type=float, default=-0.125)
    p.add_argument("--z-max", type=float, default=0.18)
    p.add_argument(
        "--disable-keepout",
        action="store_true",
        help="Disable keep-out filtering for task and TCP targets.",
    )
    p.add_argument("--keepout-x-min", type=float, default=-0.115)
    p.add_argument("--keepout-x-max", type=float, default=0.115)
    p.add_argument("--keepout-y-min", type=float, default=-0.215)
    p.add_argument("--keepout-y-max", type=float, default=0.215)
    p.add_argument("--keepout-z-min", type=float, default=-0.132)
    p.add_argument("--keepout-z-max", type=float, default=-0.002)
    p.add_argument(
        "--allow-outside-bounds",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow waypoints outside safe xyz bounds",
    )

    p.add_argument("--ik-max-iters", type=int, default=220)
    p.add_argument("--ik-damping", type=float, default=1e-3)
    p.add_argument("--ik-pos-tol", type=float, default=0.003)
    p.add_argument("--ik-yaw-tol-deg", type=float, default=8.0)
    p.add_argument("--ik-orientation", choices=["none", "yaw", "axis"], default="axis")
    p.add_argument("--ik-axis-to-align", choices=["x", "y", "z"], default="x")
    p.add_argument("--ik-axis-weight", type=float, default=0.35)
    p.add_argument("--ik-axis-tol-deg", type=float, default=1.0)
    p.add_argument(
        "--descend-ik-orientation",
        choices=["inherit", "none", "yaw", "axis"],
        default="none",
        help="Orientation mode used only for segmented descend/retreat steps",
    )
    p.add_argument("--max-fk-error-mm", type=float, default=20.0)
    return p.parse_args()


def top_down_orientation(yaw_deg: float) -> np.ndarray:
    return Rz(yaw_deg) @ Rx(180.0)


def is_finite_config(cfg: dict[str, float]) -> bool:
    vals = [cfg[k] for k in ARM + [GRIP]]
    return bool(np.all(np.isfinite(np.array(vals, dtype=float))))


def validate_pose_dict(cfg: dict[str, float]) -> None:
    missing = [k for k in ARM + [GRIP] if k not in cfg]
    if missing:
        raise KeyError(f"Missing joints in pose dict: {missing}")


def in_bounds(xyz: np.ndarray, args) -> bool:
    x, y, z = xyz
    return (
        args.x_min <= x <= args.x_max
        and args.y_min <= y <= args.y_max
        and args.z_min <= z <= args.z_max
    )


def in_keepout(xyz: np.ndarray, args) -> bool:
    if args.disable_keepout:
        return False
    x, y, z = xyz
    return (
        args.keepout_x_min <= x <= args.keepout_x_max
        and args.keepout_y_min <= y <= args.keepout_y_max
        and args.keepout_z_min <= z <= args.keepout_z_max
    )


def solve_ik_pose(
    target_xyz: np.ndarray,
    current_cfg: dict[str, float] | None,
    args,
    orientation_mode: str | None = None,
) -> Tuple[Dict[str, float], float, np.ndarray]:
    mode = args.ik_orientation if orientation_mode is None else orientation_mode
    r_des = top_down_orientation(args.tool_yaw_deg)
    tcp_offset_local = np.array(
        [args.tcp_grasp_offset_x, args.tcp_grasp_offset_y, args.tcp_grasp_offset_z],
        dtype=float,
    )
    tcp_target_xyz = np.array(target_xyz, dtype=float).reshape(3) - (r_des @ tcp_offset_local)
    if in_keepout(tcp_target_xyz, args):
        raise RuntimeError(
            f"TCP target enters keep-out volume for task target {np.round(target_xyz, 4)} "
            f"(tcp={np.round(tcp_target_xyz, 4)})"
        )
    q_sol = get_inverse_kinematics(
        target_position=tcp_target_xyz,
        target_orientation=r_des,
        initial_guess=current_cfg,
        gripper=float(args.gripper_close_pct if current_cfg is None else current_cfg[GRIP]),
        max_iters=args.ik_max_iters,
        damping=args.ik_damping,
        pos_tol=args.ik_pos_tol,
        yaw_tol_deg=args.ik_yaw_tol_deg,
        axis_tol_deg=args.ik_axis_tol_deg,
        axis_weight=args.ik_axis_weight,
        orientation_mode=mode,
        axis_to_align=args.ik_axis_to_align,
    )
    validate_pose_dict(q_sol)
    if not is_finite_config(q_sol):
        raise RuntimeError(f"IK returned non-finite config for target {np.round(target_xyz, 4)}")

    p_fk, _, _ = get_forward_kinematics(q_sol)
    err_mm = 1000.0 * np.linalg.norm(tcp_target_xyz - p_fk)
    if err_mm > args.max_fk_error_mm:
        raise RuntimeError(
            f"IK residual too high ({err_mm:.1f} mm > {args.max_fk_error_mm:.1f} mm) for target {np.round(target_xyz, 4)}"
        )
    return q_sol, float(err_mm), tcp_target_xyz


def make_cartesian_segment(start_xyz: np.ndarray, end_xyz: np.ndarray, steps: int) -> list[np.ndarray]:
    n = max(1, int(steps))
    if n == 1:
        return [np.array(end_xyz, dtype=float)]
    return [
        (1.0 - a) * np.array(start_xyz, dtype=float) + a * np.array(end_xyz, dtype=float)
        for a in np.linspace(1.0 / n, 1.0, n)
    ]


def confirm_or_raise(args, label: str, xyz: np.ndarray) -> None:
    if not args.confirm:
        return
    ans = input(f"Run step '{label}' to target {np.round(xyz, 4)}? [y/N] ").strip().lower()
    if ans not in {"y", "yes"}:
        raise RuntimeError(f"User cancelled at step '{label}'")


def main():
    args = parse_args()

    from so101_utils import hold_position, load_calibration, move_to_pose, setup_motors

    pick_approach_height = (
        float(args.approach_height)
        if args.pick_approach_height is None
        else float(args.pick_approach_height)
    )
    place_retreat_height = (
        float(args.approach_height)
        if args.place_retreat_height is None
        else float(args.place_retreat_height)
    )

    pick_approach = np.array([args.pick_x, args.pick_y, args.pick_z + pick_approach_height], dtype=float)
    pick_grasp = np.array([args.pick_x, args.pick_y, args.pick_z], dtype=float)
    transfer_z = max(args.pick_z, args.place_z) + args.lift_height
    pick_lift = np.array([args.pick_x, args.pick_y, transfer_z], dtype=float)
    place_drop = np.array([args.place_x, args.place_y, args.place_z], dtype=float)
    place_retreat = np.array([args.place_x, args.place_y, args.place_z + place_retreat_height], dtype=float)

    waypoints = [
        ("pick_approach", pick_approach),
        ("pick_descend", pick_grasp),
        ("pick_lift", pick_lift),
        ("place_drop", place_drop),
        ("place_retreat", place_retreat),
    ]
    if not args.allow_outside_bounds:
        oob = [(name, xyz) for name, xyz in waypoints if not in_bounds(xyz, args)]
        if oob:
            details = ", ".join([f"{name}:{np.round(xyz, 4)}" for name, xyz in oob])
            needed_z_max = max(float(xyz[2]) for _, xyz in waypoints)
            raise ValueError(
                "Some waypoints are outside safe bounds. "
                "Adjust --pick/--place/height args or use --allow-outside-bounds. "
                f"Out-of-bounds: {details}. "
                f"For this plan, try --z-max {needed_z_max + 0.005:.3f} or reduce --lift-height."
            )
    if not args.disable_keepout:
        blocked = [(name, xyz) for name, xyz in waypoints if in_keepout(xyz, args)]
        if blocked:
            details = ", ".join([f"{name}:{np.round(xyz, 4)}" for name, xyz in blocked])
            raise ValueError(
                "Some task waypoints are inside the keep-out volume. "
                "Adjust --pick/--place/height args or use --disable-keepout. "
                f"Blocked: {details}."
            )

    calibration = load_calibration(args.robot_name)
    bus = setup_motors(calibration, args.port_id)

    try:
        print(
            "[INFO] workspace bounds: "
            f"x[{args.x_min:.3f},{args.x_max:.3f}] "
            f"y[{args.y_min:.3f},{args.y_max:.3f}] "
            f"z[{args.z_min:.3f},{args.z_max:.3f}]"
        )
        if not args.disable_keepout:
            print(
                "[INFO] keep-out enabled: "
                f"x[{args.keepout_x_min:.3f},{args.keepout_x_max:.3f}] "
                f"y[{args.keepout_y_min:.3f},{args.keepout_y_max:.3f}] "
                f"z[{args.keepout_z_min:.3f},{args.keepout_z_max:.3f}]"
            )

        starting_pose = bus.sync_read("Present_Position")
        validate_pose_dict(starting_pose)
        current = dict(starting_pose)
        current[GRIP] = float(current.get(GRIP, args.gripper_open_pct))

        def wait_for_gripper_settle() -> float:
            prev = float(bus.sync_read("Present_Position")[GRIP])
            stable = 0
            t0 = time.time()
            while time.time() - t0 < args.gripper_settle_timeout:
                time.sleep(0.05)
                cur = float(bus.sync_read("Present_Position")[GRIP])
                if abs(cur - prev) <= args.gripper_stable_eps:
                    stable += 1
                else:
                    stable = 0
                prev = cur
                if stable >= max(1, args.gripper_stable_cycles):
                    break
            return prev

        def set_gripper(
            name: str,
            gripper_pct: float,
            *,
            settle_seconds: float | None = None,
            settle_until_stable: bool = False,
        ) -> float:
            nonlocal current
            goal = dict(current)
            goal[GRIP] = float(np.clip(gripper_pct, 0.0, 100.0))
            print(f"[GRIP] {name:>12} | {goal[GRIP]:5.1f}%")
            move_to_pose(bus, goal, args.gripper_duration, print_feedback=args.print_feedback)
            hold_position(0.5 * args.hold_seconds if settle_seconds is None else max(0.0, settle_seconds))
            if settle_until_stable:
                settled = wait_for_gripper_settle()
                print(f"[GRIP] {'settled':>12} | reached {settled:5.1f}%")
            achieved = bus.sync_read("Present_Position")
            achieved[GRIP] = float(achieved[GRIP])
            current = dict(achieved)
            return float(achieved[GRIP])

        def close_gripper_until_ready() -> float:
            target = float(np.clip(args.gripper_close_pct, 0.0, 100.0))
            measured = set_gripper(
                "close_pick",
                target,
                settle_seconds=args.post_close_hold,
                settle_until_stable=True,
            )
            err = abs(measured - target)
            tries = 0
            while err > args.gripper_close_tol and tries < max(0, args.gripper_close_retries):
                tries += 1
                print(
                    f"[GRIP] {'retry_close':>12} | measured={measured:5.1f}% "
                    f"target={target:5.1f}% err={err:4.1f}% (try {tries})"
                )
                measured = set_gripper(
                    f"close_retry{tries}",
                    target,
                    settle_seconds=0.6 * args.post_close_hold,
                    settle_until_stable=True,
                )
                err = abs(measured - target)

            if err > args.gripper_close_tol:
                raise RuntimeError(
                    f"Gripper did not reach close target before lift "
                    f"(measured={measured:.1f}%, target={target:.1f}%, tol={args.gripper_close_tol:.1f}%)."
            )
            print(f"[GRIP] {'close_ready':>12} | measured={measured:5.1f}%")
            return measured

        def solve_ik_with_retry(
            name: str,
            target_xyz: np.ndarray,
            seed_cfg: dict[str, float],
            ik_orientation_mode: str | None = None,
        ):
            try:
                return solve_ik_pose(
                    target_xyz,
                    seed_cfg,
                    args,
                    orientation_mode=ik_orientation_mode,
                )
            except RuntimeError as exc:
                print(f"[IK] {name:>14} | seeded solve failed, retrying neutral seed: {exc}")
                q_sol, err_mm, tcp_target_xyz = solve_ik_pose(
                    target_xyz,
                    None,
                    args,
                    orientation_mode=ik_orientation_mode,
                )
                print(f"[IK] {name:>14} | neutral-seed retry succeeded")
                return q_sol, err_mm, tcp_target_xyz

        def move_arm(
            name: str,
            target_xyz: np.ndarray,
            *,
            duration: float | None = None,
            hold_scale: float = 1.0,
            ask_confirm: bool = True,
            ik_seed_cfg: dict[str, float] | None = None,
            ik_orientation_mode: str | None = None,
            update_from_achieved: bool = True,
        ):
            nonlocal current
            if ask_confirm:
                confirm_or_raise(args, name, target_xyz)
            seed_cfg = current if ik_seed_cfg is None else ik_seed_cfg
            q_sol, err_mm, tcp_target_xyz = solve_ik_with_retry(
                name,
                target_xyz,
                seed_cfg,
                ik_orientation_mode=ik_orientation_mode,
            )
            q_sol[GRIP] = float(current[GRIP])
            print(
                f"[IK] {name:>14} | task={np.round(target_xyz, 4)} "
                f"tcp={np.round(tcp_target_xyz, 4)} | fk_err={err_mm:6.2f} mm"
            )
            move_to_pose(
                bus,
                q_sol,
                args.move_duration if duration is None else duration,
                print_feedback=args.print_feedback,
            )
            if hold_scale > 0.0:
                hold_position(max(0.0, args.hold_seconds * hold_scale))

            achieved = bus.sync_read("Present_Position")
            achieved[GRIP] = float(current[GRIP])
            p_fk_cmd, _, _ = get_forward_kinematics(q_sol)
            p_fk_real, _, _ = get_forward_kinematics(achieved)
            err_cmd_real_mm = 1000.0 * np.linalg.norm(p_fk_cmd - p_fk_real)
            err_tcp_real_mm = 1000.0 * np.linalg.norm(tcp_target_xyz - p_fk_real)
            print(
                f"[FK] {name:>14} | tcp->fk_real={err_tcp_real_mm:7.2f} mm, "
                f"fk_cmd->fk_real={err_cmd_real_mm:7.2f} mm"
            )
            if update_from_achieved:
                current = dict(achieved)
            else:
                cmd_state = dict(q_sol)
                cmd_state[GRIP] = float(current[GRIP])
                current = cmd_state
            return q_sol

        def move_arm_segmented(
            name: str,
            start_xyz: np.ndarray,
            end_xyz: np.ndarray,
            steps: int,
            duration_scale: float = 1.0,
        ):
            points = make_cartesian_segment(start_xyz, end_xyz, steps)
            confirm_or_raise(args, name, end_xyz)
            total_scale = max(0.2, float(duration_scale))
            d = max(0.2, float(args.move_duration) * total_scale / max(1, len(points)))
            descend_mode = (
                args.ik_orientation
                if args.descend_ik_orientation == "inherit"
                else args.descend_ik_orientation
            )
            ik_seed = dict(current)
            for i, p in enumerate(points):
                q_sol = move_arm(
                    f"{name}_{i+1}/{len(points)}",
                    p,
                    duration=d,
                    hold_scale=0.0,
                    ask_confirm=False,
                    ik_seed_cfg=ik_seed,
                    ik_orientation_mode=descend_mode,
                    update_from_achieved=False,
                )
                ik_seed = dict(q_sol)
            # Re-sync state once at the end of segmented motion.
            achieved = bus.sync_read("Present_Position")
            achieved[GRIP] = float(current[GRIP])
            current.update(achieved)

        print("[INFO] Starting pick-and-place sequence")
        print(
            "[INFO] TCP->grasp offset (tool frame, m): "
            f"[{args.tcp_grasp_offset_x:.4f}, {args.tcp_grasp_offset_y:.4f}, {args.tcp_grasp_offset_z:.4f}]"
        )
        print(
            "[INFO] approach heights: "
            f"pick={pick_approach_height:.3f} m, place_retreat={place_retreat_height:.3f} m"
        )
        set_gripper("open_start", args.gripper_open_pct)
        move_arm("pick_approach", pick_approach)
        if args.segmented_descend:
            move_arm_segmented(
                "pick_descend",
                pick_approach,
                pick_grasp,
                args.approach_steps,
                duration_scale=args.approach_duration_scale,
            )
        else:
            move_arm(
                "pick_descend",
                pick_grasp,
                duration=max(0.2, args.move_duration * args.approach_duration_scale),
                hold_scale=0.0,
            )
        close_gripper_until_ready()
        hold_position(args.hold_seconds)
        move_arm("pick_lift", pick_lift)
        move_arm("place_drop", place_drop)
        set_gripper("open_drop", args.gripper_open_pct)
        hold_position(args.hold_seconds)
        if args.segmented_descend:
            move_arm_segmented(
                "place_retreat",
                place_drop,
                place_retreat,
                max(2, args.descend_steps // 2),
                duration_scale=0.7 * args.descend_duration_scale,
            )
        else:
            move_arm(
                "place_retreat",
                place_retreat,
                duration=max(0.2, args.move_duration * 0.8),
                hold_scale=0.0,
            )
        hold_position(args.final_hold)
        print("[INFO] Pick-and-place sequence complete")

        if args.return_home:
            print("[INFO] Returning to starting pose")
            move_to_pose(bus, starting_pose, args.move_duration, print_feedback=args.print_feedback)
            bus.disable_torque()
    finally:
        bus.disable_torque()
        bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()


"""
python pick_and_place.py \
    --port-id /dev/tty.usbmodem5AB01824781 \
    --robot-name follower-1 \
    --no-segmented-descend \
    --descend-duration-scale 1.0 \
    --gripper-close-pct 3 \
    --gripper-duration 1.0 \
    --post-close-hold 2.0 \
    --gripper-close-tol 20.0 \
    --gripper-close-retries 3 \
    --no-confirm
"""
