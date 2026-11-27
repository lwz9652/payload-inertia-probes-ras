"""Fit every archived candidate with the re-centered chain solver."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from controls_common import FOLLOWUP, fit_case
from candidate_pool import candidate_pool, nominal_scores
from study_common import matrices, read, recreate, write


def case_job(key, fixed_chart=False):
    robot, trial = key
    case = recreate(robot, trial)
    a = case["directions"]["3loads_3dirs"]
    truth = matrices(case["j_hold"], case["tests"])
    previous = [r for r in read(FOLLOWUP / "candidate_cases.json")
                if (r["robot"], r["trial"]) == (case["name"], trial)]
    assert [r["index"] for r in previous] == list(range(16))
    outcomes = []
    for r, new_load in zip(previous, candidate_pool(trial), strict=True):
        loads = np.concatenate((case["catalog"][:3], new_load[None]))
        scores = nominal_scores(case, loads, a)
        for key in scores:
            np.testing.assert_allclose(scores[key], r["scores"][key], rtol=1e-6, atol=1e-8)
        observed = (matrices(case["j_cal"], loads) + case["noises"][.01]) @ a
        fit = fit_case(case["q"], a, loads, observed, case["held_q"], case["tests"],
                       truth, case["unseen"], fixed_chart=fixed_chart)
        assert fit["starts"] == r["fit"]["starts"] == 2 and fit["eligible"] == r["used_poses"]
        outcomes.append(dict(robot=case["name"], trial=trial, index=r["index"], scores=scores, fit=fit))
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument('--fixed-chart', action='store_true', help='Keep charts anchored with the same damping and residual budget')
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job,(arm,i),args.fixed_chart) for arm in range(2) for i in range(args.trials)]
        cases = []
        for task in as_completed(jobs):
            block = task.result()
            cases.extend(block)
            print(block[0]["robot"],block[0]["trial"],len(cases),"candidates complete",flush=True)
    cases.sort(key=lambda r:(r["robot"],r["trial"],r["index"]))
    write(args.output,cases)


if __name__ == "__main__":
    main()
