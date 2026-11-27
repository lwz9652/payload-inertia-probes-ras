"""Hold the torque observations fixed while perturbing calibration load descriptors."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from controls_common import fit_case
from study_common import BOXES, box_inertia, matrices, recreate, write


LEVELS = {
    "mass_fraction": (.005, .01, .03),
    "mount_translation_m": (.0005, .002, .005),
    "mount_rotation_deg": (.5, 2., 5.),
}


def perturbed_catalog(errors, mode, level):
    output = []
    for j, (mass, center, sides, attitude) in enumerate(BOXES):
        if mode == "mass_fraction":
            mass *= 1 + level * errors[mode][j]
        elif mode == "mount_translation_m":
            center = np.asarray(center) + level * errors[mode][j]
        elif mode == "mount_rotation_deg":
            shift = Rotation.from_rotvec(np.deg2rad(level) * errors[mode][j])
            center = shift.apply(center)
            attitude = (shift * Rotation.from_euler("xyz", attitude, degrees=True)).as_euler(
                "xyz", degrees=True,
            )
        output.append(box_inertia(mass, center, sides, attitude))
    return np.asarray(output)


def case_job(key):
    robot, trial, sigma = key
    case = recreate(robot, trial)
    rng = np.random.default_rng(2026112904 + 100*robot + trial)
    a = case["directions"]["3loads_3dirs"]
    unseen = np.linalg.qr(a, mode="complete")[0][:,a.shape[1]:]
    clean = matrices(case["j_cal"], case["catalog"])
    # Reuse the same standardized draw at both torque-noise scales.
    observed = (clean + (sigma / .001) * case["noises"][.001]) @ a
    truth = matrices(case["j_hold"], case["tests"])
    offsets = {}
    offsets["mass_fraction"] = rng.choice((-1., 1.), size=len(BOXES))
    for key in ("mount_translation_m", "mount_rotation_deg"):
        directions = rng.normal(size=(len(BOXES),3))
        offsets[key] = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    outcome = []
    for name, size in [("exact",0.)] + [(key,level) for key, levels in LEVELS.items() for level in levels]:
        calibrated = perturbed_catalog(offsets, name, size)
        assert np.linalg.eigvalsh(calibrated).min() > 0
        result = fit_case(case["q"], a, calibrated, observed, case["held_q"], case["tests"], truth, unseen)
        outcome.append(dict(robot=case["name"],trial=trial,mode=name,level=size,fit=result))
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--torque-noise", type=float, default=.001)
    args = parser.parse_args()
    if args.torque_noise <= 0:
        parser.error("torque noise must be positive")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        active = [pool.submit(case_job,(robot,i,args.torque_noise))
                  for robot in range(2) for i in range(args.trials)]
        rows = []
        for task in as_completed(active):
            group = task.result()
            rows.extend(group)
            print(group[0]["robot"],group[0]["trial"],len(rows),"descriptor cases complete",flush=True)
    rows.sort(key=lambda r:(r["robot"],r["trial"],r["mode"],r["level"]))
    write(args.output, dict(rows=rows, levels=LEVELS, raw_torque_sigma=args.torque_noise))


if __name__ == "__main__":
    main()
