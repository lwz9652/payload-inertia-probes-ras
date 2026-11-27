"""URDF forward geometry and product-of-exponentials reconstruction."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    return np.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def import_chain(urdf, tool):
    root = ET.parse(urdf).getroot()
    lookup = {j.find("child").get("link"): j for j in root.findall("joint")}
    links = []
    while tool in lookup:
        joint = lookup[tool]
        kind = joint.get("type")
        if kind not in ("fixed", "revolute"):
            raise ValueError(f"Unexpected {kind} joint in serial tool chain")
        origin = joint.find("origin")
        at = {} if origin is None else origin.attrib
        xyz = np.fromstring(at.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(at.get("rpy", "0 0 0"), sep=" ")
        rotation = Rotation.from_euler("xyz", rpy).as_matrix()
        axis = None if kind == "fixed" else np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        links.append((rotation, xyz, axis))
        tool = joint.find("parent").get("link")
    return links[::-1]


def forward_urdf(chain, q, mounting_rotation=None, mounting_offset=None):
    orientation = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    translation = np.zeros((len(q), 3))
    joint_positions, joint_axes = [], []
    for rotation, xyz, axis in chain:
        translation = translation + (orientation @ xyz[:, None])[..., 0]
        orientation = orientation @ rotation
        if axis is not None:
            axis = axis / np.linalg.norm(axis)
            joint_positions.append(translation)
            joint_axes.append((orientation @ axis[:, None])[..., 0])
            theta = q[:, len(joint_axes) - 1]
            orientation = orientation @ Rotation.from_rotvec(theta[:, None] * axis).as_matrix()
    if mounting_offset is not None:
        translation = translation + (orientation @ mounting_offset[:, None])[..., 0]
    if mounting_rotation is not None:
        orientation = orientation @ mounting_rotation
    assert len(joint_axes) == q.shape[1]
    axes = np.asarray(joint_axes).transpose(1, 2, 0)
    points = np.asarray(joint_positions).transpose(1, 2, 0)
    velocities = np.cross(axes, translation[:, :, None] - points, axis=1)
    body = orientation.swapaxes(-2, -1)
    return np.concatenate((body @ axes, body @ velocities), axis=1)


def perturb_chain(chain, rng):
    """Alter revolute joint origins, orientations and encoder zero angles."""
    changed = []
    for orientation, translation, axis in chain:
        if axis is not None:
            axis_error = Rotation.from_rotvec(rng.uniform(-.014, .014, 3))
            encoder_zero = Rotation.from_rotvec(rng.uniform(-.03, .03) * axis)
            orientation = orientation @ (axis_error * encoder_zero).as_matrix()
            translation = translation + rng.uniform(-.004, .004, 3)
        changed.append((orientation, translation, axis))
    return changed


def move_attachment(jacobian, rotation, offset):
    angular, linear = jacobian[:, :3], jacobian[:, 3:]
    return np.concatenate(
        (rotation.T @ angular, rotation.T @ (linear - skew(offset) @ angular)), axis=1,
    )


def append_joint(rotation, translation, axis, point, angle):
    generator = skew(axis)
    theta = -angle[:, None, None]
    step = np.eye(3) + np.sin(theta) * generator + (1 - np.cos(theta)) * (generator @ generator)
    offset = point - (step @ point[:, None])[..., 0]
    return rotation @ step, translation + (rotation @ offset[..., None])[..., 0]


def serial_body(q, axes, points):
    rotation = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    translation = np.zeros((len(q), 3))
    factor = np.empty((len(q), 6, q.shape[1]))
    for joint in reversed(range(q.shape[1])):
        axis, point = axes[joint], points[joint]
        angular = (rotation @ axis[:, None])[..., 0]
        velocity = (rotation @ np.cross(point, axis)[:, None])[..., 0]
        factor[:, :3, joint] = angular
        factor[:, 3:, joint] = np.cross(translation, angular) + velocity
        rotation, translation = append_joint(rotation, translation, axis, point, q[:, joint])
    return factor


def reverse_axes(configurations, aligned, weight=None):
    n_config, _, n_joint = aligned.shape
    rotation = np.broadcast_to(np.eye(3), (n_config, 3, 3)).copy()
    translation = np.zeros((n_config, 3))
    axes, points = np.empty((n_joint, 3)), np.empty((n_joint, 3))
    sqrt_weight = np.ones(n_config) if weight is None else np.sqrt(weight)
    for joint in reversed(range(n_joint)):
        transform = np.zeros((n_config, 6, 6))
        transform[:, :3, :3] = rotation
        transform[:, 3:, 3:] = rotation
        transform[:, 3:, :3] = skew(translation) @ rotation
        screw = np.linalg.lstsq(
            (transform * sqrt_weight[:, None, None]).reshape(-1, 6),
            (aligned[:, :, joint] * sqrt_weight[:, None]).ravel(), rcond=1e-12,
        )[0]
        size = np.linalg.norm(screw[:3])
        if size < 1e-8:
            raise ValueError("Unreliable fit of a revolute screw")
        axes[joint] = screw[:3] / size
        points[joint] = np.cross(axes[joint], screw[3:] / size)
        rotation, translation = append_joint(
            rotation, translation, axes[joint], points[joint], configurations[:, joint],
        )
    return axes, points


def nominal_models():
    """Load the nominal chains exported from the pinned URDF inputs."""
    path = Path(__file__).resolve().parents[1] / "review" / "kinematic_inputs.json"
    models = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for model in models:
        chain = [
            (np.array(rotation), np.array(displacement), None if axis is None else np.array(axis))
            for rotation, displacement, axis in model["chain"]
        ]
        result.append((model, chain, np.asarray(model["bounds"])))
    return result
