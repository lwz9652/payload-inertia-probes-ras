"""Coordinates and weighted directional residuals for revolute-chain calibration."""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

FOLLOWUP = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLLOWUP.parent / "review"))
from study_common import matrices, shared_reference_whitener
from robot_geometry import serial_body


def axis_planes(axes):
    """Form orthonormal tangent coordinates at each unit axis."""
    least_aligned = np.eye(3)[np.argmin(abs(axes), axis=1)]
    first = np.cross(axes, least_aligned)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    return np.stack((first, np.cross(axes, first)), axis=-1)


def move_geometry(geometry, planes, step):
    """Rotate a signed line and translate its perpendicular point."""
    axes, points = geometry
    step = step.reshape(-1, 4)
    axis_tangent = (planes @ step[:, :2, None])[..., 0]
    rotation = Rotation.from_rotvec(np.cross(axes, axis_tangent))
    displacement = (planes @ step[:, 2:, None])[..., 0]
    return rotation.apply(axes), rotation.apply(points + displacement)


def directional_residual(q, inertias, a, observed):
    whiten = shared_reference_whitener(len(inertias))

    def residual(geometry):
        j = serial_body(q, *geometry)
        change = matrices(j, inertias) @ a - observed
        return np.einsum("ml,qlik->qmik", whiten, change).ravel()

    return residual


def finite_difference(residual, geometry, center, h=1e-6):
    planes = axis_planes(geometry[0])
    n = 4 * len(planes)
    directions = h * np.eye(n)
    samples = [residual(move_geometry(geometry, planes, offset)) for offset in directions]
    derivative = (np.asarray(samples) - center).T / h
    return derivative, planes
