"""Recreate the paired simulation cases used in the submitted-directional draft."""

from pathlib import Path
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / "directional"
sys.path.insert(0, str(ORIGINAL))

from directional_recovery import contrast_noise, shared_reference_whitener
from fit_chain import axes_from_home
from probe_benchmark import BOXES, NOISE, box_inertia, holdout_boxes, matrices, relative_rmse
from robot_geometry import forward_urdf, nominal_models, perturb_chain, serial_body


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def recreate(robot_index, trial):
    model, chain, bounds = nominal_models()[robot_index]
    name = model["robot"]
    spec = read(ORIGINAL / f"confirmation/{name}_protocol.json")
    if model["source_sha256"] != spec["urdf_sha256"]:
        raise ValueError(f"{name} joint geometry does not match the archived protocol")
    low, high = bounds.T
    rng = np.random.default_rng(2026111920 + 100 * robot_index + spec["seed_offset"] + trial)
    n = len(bounds)
    calibration = rng.uniform(low, high, (spec["calibration_configurations"], n))
    heldout = rng.uniform(low, high, (spec["heldout_configurations"], n))
    tests = holdout_boxes(rng)
    mount = Rotation.from_rotvec(rng.uniform(-.08, .08, 3)).as_matrix()
    offset = rng.uniform(-.035, .035, 3)
    changed = perturb_chain(chain, rng)
    q_nom = forward_urdf(chain, np.zeros((1, n)))[0]
    catalog = np.asarray([box_inertia(*row) for row in BOXES])
    measures = {sigma: np.asarray([contrast_noise(rng, 4, n, n, sigma) for _ in calibration])
                for sigma in NOISE}
    return dict(
        name=name, chain=chain, changed=changed, q=calibration, held_q=heldout, tests=tests,
        reference=axes_from_home(q_nom), mount=mount, offset=offset, catalog=catalog,
        j_cal=forward_urdf(changed, calibration, mount, offset),
        j_hold=forward_urdf(changed, heldout, mount, offset), noises=measures,
        directions={d["name"]: np.asarray(d["matrix"]) for d in spec["directions"]},
        unseen=np.asarray(spec["unprobed_test_directions"]),
    )


def errors(prediction, truth, directions):
    return dict(R_M=relative_rmse(prediction, truth),
                R_A=relative_rmse(prediction @ directions, truth @ directions))


def score(geometry, case, truth):
    j = serial_body(case["held_q"], *geometry)
    return errors(matrices(j, case["tests"]), truth, case["unseen"])
