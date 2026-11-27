"""Confirm whether random load sets remain viable after pooling all calibration poses."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from study_common import HERE, box_inertia, contrast_noise, matrices, read, recreate, write
from fit_chain import initialize_from_data
from matched_review import choose_fit
from random_payload_review import load_draw, probe_draws


def fit_random(task):
    robot, group, planned = task
    case = recreate(robot, group)
    values = load_draw(group)
    i = np.asarray([box_inertia(*row) for row in zip(*values, strict=True)])
    n = case["q"].shape[1]
    choices, rng = probe_draws(robot, group, n)
    full = matrices(case["j_cal"], i)
    truth = matrices(case["j_hold"], case["tests"])
    noise = np.asarray([contrast_noise(rng, 4, n, n, 1.) for _ in case["q"]])
    results = []
    for design, selected in planned:
        a = choices[selected]
        for sigma in (.001, .01):
            observed = (full + sigma * noise) @ a
            starts, eligible = initialize_from_data(case["q"], i, a, observed)
            fit = choose_fit(case, truth, i, a, observed, starts)
            nominal = choose_fit(case, truth, i, a, observed, [case["reference"]])
            results.append(dict(
                robot=case["name"], group=group, design=design, sigma=sigma,
                pointwise_starts=eligible, torque=fit, nominal=nominal,
            ))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    old = read(HERE / "random_payload_cases.json")
    selection = {(r["robot"], r["load_group"], r["design"]): r["selected_candidate"]
                 for r in old if r["spread_scale"] == 1 and r["sigma"] == .001}
    jobs = [(j, t, [(d, selection[(r, t, d)]) for d in ("random", "nominal_rank")])
            for j, r in enumerate(("Panda", "UR10e")) for t in range(12)]
    scores = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(fit_random, jobs):
            scores.extend(rows)
            if len(scores) % 24 == 0:
                print(len(scores), "random-label comparisons", flush=True)
    write(args.output, scores)


if __name__ == "__main__":
    main()
