#!/usr/bin/env python3
"""Analytic FK for SO101 using constants from so101_new_calib.xml.

Joint inputs are in degrees for readability.
"""

from __future__ import annotations

import numpy as np


JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]


def Rx(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Ry(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def Rz(theta_deg: float) -> np.ndarray:
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def quat_wxyz_to_R(q) -> np.ndarray:
    q = np.array(q, dtype=float).reshape(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def make_T(R: np.ndarray, p) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.array(p, dtype=float).reshape(3)
    return T


# Static transforms from so101_new_calib.xml body/site poses.
P_W1 = np.array([0.0388353, 0.0, 0.0624])
Q_W1 = (3.56167e-16, 1.22818e-15, -1.0, -4.14635e-16)

P_12 = np.array([-0.0303992, -0.0182778, -0.0542])
Q_12 = (0.5, -0.5, -0.5, -0.5)

P_23 = np.array([-0.11257, -0.028, 1.73763e-16])
Q_23 = (0.707107, -5.98613e-17, -2.58051e-17, 0.707107)

P_34 = np.array([-0.1349, 0.0052, 3.62355e-17])
Q_34 = (0.707107, 9.58722e-16, -7.51313e-16, -0.707107)

P_45 = np.array([5.55112e-17, -0.0611, 0.0181])
Q_45 = (0.0172091, -0.0172091, 0.706897, 0.706897)

P_5T = np.array([0.01, -0.000218121, -0.0781274])
Q_5T = (0.707107, -0.0, 0.707107, -2.37788e-17)

R_W1 = quat_wxyz_to_R(Q_W1)
R_12 = quat_wxyz_to_R(Q_12)
R_23 = quat_wxyz_to_R(Q_23)
R_34 = quat_wxyz_to_R(Q_34)
R_45 = quat_wxyz_to_R(Q_45)
R_5T = quat_wxyz_to_R(Q_5T)


def get_gw1(theta1_deg: float) -> np.ndarray:
    return make_T(R_W1 @ Rz(theta1_deg), P_W1)


def get_g12(theta2_deg: float) -> np.ndarray:
    return make_T(R_12 @ Rz(theta2_deg), P_12)


def get_g23(theta3_deg: float) -> np.ndarray:
    return make_T(R_23 @ Rz(theta3_deg), P_23)


def get_g34(theta4_deg: float) -> np.ndarray:
    return make_T(R_34 @ Rz(theta4_deg), P_34)


def get_g45(theta5_deg: float) -> np.ndarray:
    return make_T(R_45 @ Rz(theta5_deg), P_45)


def get_g5t() -> np.ndarray:
    return make_T(R_5T, P_5T)


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def get_forward_kinematics(position_dict: dict):
    """Compute object/tool frame FK from first 5 joints.

    Args:
        position_dict: dictionary in DEGREES with keys in JOINT_KEYS.
    Returns:
        (position_xyz, rotation_3x3, transform_4x4)
    """
    t1 = float(position_dict["shoulder_pan"])
    t2 = float(position_dict["shoulder_lift"])
    t3 = float(position_dict["elbow_flex"])
    t4 = float(position_dict["wrist_flex"])
    t5 = float(position_dict["wrist_roll"])

    gwt = get_gw1(t1) @ get_g12(t2) @ get_g23(t3) @ get_g34(t4) @ get_g45(t5) @ get_g5t()
    p = gwt[:3, 3].copy()
    R = _orthonormalize(gwt[:3, :3].copy())
    gwt[:3, :3] = R
    return p, R, gwt


def fk_from_q_vector_deg(q_deg_5):
    q = np.array(q_deg_5, dtype=float).reshape(5)
    d = {
        "shoulder_pan": q[0],
        "shoulder_lift": q[1],
        "elbow_flex": q[2],
        "wrist_flex": q[3],
        "wrist_roll": q[4],
    }
    return get_forward_kinematics(d)