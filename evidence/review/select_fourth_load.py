"""Choose a fourth payload before measuring any torque and compare with an unselected load."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from study_common import box_inertia, forward_urdf, matrices, recreate, write
from fit_chain import initialize_from_data
from matched_review import choose_fit
from random_payload_review import margins


def case_job(job):
    robot, trial = job
    case = recreate(robot, trial)
    a = case["directions"]["3loads_3dirs"]
    base = case["catalog"][:3]
    rng = np.random.default_rng(2026112031 + trial)
    catalog = [
        box_inertia(rng.uniform(.65, 2.5),
                    rng.uniform([-.11, -.11, .025], [.11, .11, .20]),
                    rng.uniform(.045, .21, 3), rng.uniform(-120, 120, 3))
        for _ in range(16)
    ]
    reference = forward_urdf(case["chain"], case["q"][:8])
    scores = []
    for candidate in catalog:
        i = np.concatenate((base, candidate[None]))
        design_actions = matrices(reference, i) @ a
        design_margins = np.prod(margins(i, a, design_actions), axis=1)
        scores.append(float(np.quantile(design_margins, .20)))
    truth = matrices(case["j_hold"], case["tests"])
    outputs = []
    for name, index in (("unselected", 0), ("chosen", int(np.argmax(scores)))):
        i = np.concatenate((base, catalog[index][None]))
        clean = matrices(case["j_cal"], i)
        observed = (clean + case["noises"][.01]) @ a
        starts, eligible = initialize_from_data(case["q"], i, a, observed)
        outputs.append(dict(
            robot=case["name"], trial=trial, design=name, index=index,
            score=scores[index], candidate_scores=scores,
            exact_margin_median=float(np.median(np.prod(margins(i, a, clean @ a), axis=1))),
            used_poses=eligible, fit=choose_fit(case, truth, i, a, observed, starts),
        ))
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    jobs = [(r, t) for r in range(2) for t in range(12)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(case_job, jobs):
            rows.extend(result)
            if len(rows) % 12 == 0:
                print(len(rows), "fourth-load comparisons", flush=True)
    write(args.output, rows)


if __name__ == "__main__":
    main()
