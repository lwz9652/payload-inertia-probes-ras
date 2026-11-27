"""Compare fitting only the fixture transform with fitting the complete joint chain."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from controls_common import fit_case
from fit_chain import fit_attachment
from robot_geometry import forward_urdf, move_attachment
from study_common import errors, matrices, read, recreate, write


SAMPLING = Path(__file__).resolve().parents[1] / "sampling"
ROBOTS = ("Panda", "UR10e")


def case_job(key):
    robot, trial = key
    case = recreate(robot, trial)
    file = SAMPLING / "inputs" / f"{case['name']}_{trial + 1:02d}.npz"
    with np.load(file, allow_pickle=False) as data:
        q, test_q = data["calibration_q"], data["heldout_q"]
        loads = data["calibration_inertias"]
        tests = data["heldout_inertias"]
        nominal_starts = list(data["nominal_starts"])
        noise = data["canonical_contrasts"] - data["clean_contrasts"]
        clean = data["clean_contrasts"]
    np.testing.assert_allclose(q, case["q"], atol=1e-14)
    np.testing.assert_allclose(test_q, case["held_q"], atol=1e-14)
    np.testing.assert_allclose(clean, matrices(case["j_cal"], loads), atol=5e-13)
    plan = read(SAMPLING / "probe_design.json")["robots"][case["name"]]
    design = next(item for item in plan["designs"] if item["name"] == "extra_load")
    assert design["loads"] == len(loads)
    a = np.asarray(design["directions"])
    unseen = np.linalg.qr(a, mode="complete")[0][:, a.shape[1]:]
    np.testing.assert_allclose(a.T @ unseen, 0., atol=1e-13)
    nominal = forward_urdf(case["chain"], q)
    nominal_test = forward_urdf(case["chain"], test_q)
    nominal_prediction = matrices(nominal_test, tests)
    rows = []
    for label, chain in (("fixture_only", case["chain"]), ("fixture_and_joint_errors", case["changed"])):
        train_j, test_j = [forward_urdf(chain, states, case["mount"], case["offset"])
                           for states in (q, test_q)]
        observed = (matrices(train_j, loads) + noise) @ a
        truth = matrices(test_j, tests)
        static_fit = fit_attachment(nominal, loads, a, observed)
        parameters = np.asarray(static_fit["parameters"])
        shifted = move_attachment(nominal_test, Rotation.from_rotvec(parameters[:3]).as_matrix(),
                                  parameters[3:])
        full_fit = fit_case(q, a, loads, observed, test_q, tests, truth, unseen, nominal_starts)
        rows.append(dict(robot=case["name"], trial=trial, target=label,
                         uncorrected=errors(nominal_prediction, truth, unseen),
                         fixture_fit=static_fit,
                         fixture=errors(matrices(shifted, tests), truth, unseen), chain=full_fit))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job, (robot, trial))
                for robot in range(len(ROBOTS)) for trial in range(args.trials)]
        rows = []
        for job in as_completed(jobs):
            block = job.result()
            rows.extend(block)
            print(block[0]["robot"], block[0]["trial"], len(rows),
                  "attachment comparisons complete", flush=True)
    rows.sort(key=lambda row: (row["robot"], row["trial"], row["target"]))
    write(args.output, dict(torque_read_sd_Nm=.01, rows=rows))


if __name__ == "__main__":
    main()
