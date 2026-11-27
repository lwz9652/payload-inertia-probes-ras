"""Allocate a fixed number of signed reads to repeat measurements or new poses."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from controls_common import fit_case
from directional_recovery import contrast_noise
from robot_geometry import forward_urdf
from study_common import matrices, recreate, write


CONFIGURATIONS = (2, 4, 8, 16, 32)
SIGMA = .01
SEED = 2026112523


def case_job(key):
    robot, trial, seed = key
    case = recreate(robot, trial)
    anchor = case["q"][0].copy()
    anchor[-1] = -1.1
    second = anchor.copy()
    second[-1] += .50
    q = np.vstack([anchor, second, case["q"][1:31]])
    a = case["directions"]["4loads_2dirs"]
    loads = case["catalog"]
    n, k = a.shape
    unseen = np.linalg.qr(a, mode="complete")[0][:, k:]
    clean = matrices(forward_urdf(case["changed"], q, case["mount"], case["offset"]), loads)
    truth = matrices(case["j_hold"], case["tests"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, robot, trial]))
    # The first reading at any repeated pose is shared by all designs containing that pose.
    noisy = np.array([[contrast_noise(rng, len(loads), n, n, SIGMA) for _ in q]
                      for _ in range(CONFIGURATIONS[-1] // CONFIGURATIONS[0])])
    rows = []
    for count in CONFIGURATIONS:
        repeats = CONFIGURATIONS[-1] // count
        observed = (clean[:count] + noisy[:repeats, :count].mean(axis=0)) @ a
        fit = fit_case(q[:count], a, loads, observed, case["held_q"], case["tests"], truth, unseen)
        rows.append(dict(robot=case["name"], trial=trial, configurations=count, repeats=repeats,
                         signed_reads=2*k*(len(loads)+1)*count*repeats, fit=fit))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job, (robot, trial, args.seed))
                for robot in range(2) for trial in range(args.trials)]
        rows = []
        for job in as_completed(jobs):
            block = job.result()
            rows.extend(block)
            print(block[0]["robot"], block[0]["trial"], len(rows),
                  "read allocations complete", flush=True)
    rows.sort(key=lambda row: (row["robot"], row["trial"], row["configurations"]))
    write(args.output, dict(seed=args.seed, torque_read_sd_Nm=SIGMA, rows=rows))


if __name__ == "__main__":
    main()
