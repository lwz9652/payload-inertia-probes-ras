"""Shared protocol and path handling for the matched sampling study."""

from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE.parent
sys.path.insert(0, str(EVIDENCE / "review"))
sys.path.insert(0, str(EVIDENCE / "followup"))

from study_common import read

ROBOTS = ("Panda", "UR10e")
NOISE_SD = .01
INITIALIZATIONS = ("torque", "nominal")
PRIMARY = ("base", "extra_load", "extra_direction")


def configuration_file(directory, robot, trial):
    return directory / f"{robot}_{trial + 1:02d}.npz"


def designs(protocol, robot):
    entry = protocol["robots"][robot]
    for item in entry["designs"]:
        yield item["name"], item["loads"], np.asarray(item["directions"])


def load_npz(path, names):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[name] for name in names)
