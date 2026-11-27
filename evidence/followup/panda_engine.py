"""Constraint-free MuJoCo Panda dynamics for the matched-state stress study."""

import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
MODEL_SHA256 = "2935f86334a1d27a660a5643f03e41fc899ce1d34c6bf7b4105d146249630cda"


def source_xml():
    content = (HERE / "panda_contact_free.xml").read_bytes()
    if hashlib.sha256(content).hexdigest() != MODEL_SHA256:
        raise ValueError("Archived MuJoCo model does not match its pinned dynamics")
    return content.decode("utf-8")


def space_to_com(inertia):
    """Decode a physical spatial inertia in angular-first coordinates."""
    mass = float(inertia[3, 3])
    cross = inertia[:3, 3:]
    center = np.asarray([cross[2, 1], cross[0, 2], cross[1, 0]]) / mass
    at_com = inertia[:3, :3] - mass * (center @ center * np.eye(3) - np.outer(center, center))
    return mass, center, at_com


def values(items):
    return " ".join(format(x, ".17g") for x in items)


def add_payload(base, spatial_inertia):
    mass, center, tensor = space_to_com(spatial_inertia)
    eig, axes = np.linalg.eigh(tensor)
    if np.linalg.det(axes) < 0:
        axes[:, 0] *= -1
    if eig[0] <= 0 or eig[-1] >= eig[0] + eig[1]:
        raise ValueError("Payload inertia is not physical")
    orientation = np.empty(4)
    mujoco.mju_mat2Quat(orientation, axes.ravel())
    root = ET.fromstring(base)
    tool = root.find(".//body[@name='end_effector']")
    if tool is None:
        raise ValueError("The model has no end-effector body")
    payload = ET.SubElement(tool, "body", name="calibration_payload", pos=values(center))
    ET.SubElement(payload, "inertial", mass=values([mass]), pos="0 0 0",
                  diaginertia=values(eig), quat=values(orientation))
    return ET.tostring(root, encoding="unicode")


class Panda:
    def __init__(self, xml):
        self.model = mujoco.MjModel.from_xml_string(xml)
        model = self.model
        if model.nq != 7 or model.nv != 7:
            raise ValueError("Seven joint coordinates required")
        if [model.joint(i).name for i in range(model.njnt)] != [f"joint{i}" for i in range(1,8)]:
            raise ValueError("Unexpected joint order")
        model.opt.disableflags |= (
            int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
        )
        self.data = mujoco.MjData(model)

    def inverse(self, q, velocity, acceleration):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.data.qvel[:] = velocity
        self.data.qacc[:] = acceleration
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()

    def mass_grid(self, configurations):
        values = np.empty((len(configurations), 7, 7))
        for i, q in enumerate(configurations):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = q
            mujoco.mj_forward(self.model, self.data)
            mujoco.mj_fullM(self.model, values[i], self.data.qM)
        return values
