"""Compare equal-budget direct and torque-derived starts on identical observations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from controls_common import HERE, fit_case
from geometry_review import perturb_start
from prior_design import planned_directions
from study_common import matrices, read, recreate, write


BOUNDS = (90, 135)
START_SEED = 2026112511


def isotropic_start(joints, radius, rng):
    axes = rng.standard_normal((joints, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    point_directions = rng.standard_normal((joints, 3))
    point_directions /= np.linalg.norm(point_directions, axis=1, keepdims=True)
    points = point_directions * (radius * rng.random((joints, 1)) ** (1 / 3))
    points -= np.sum(points * axes, axis=1, keepdims=True) * axes
    return axes, points


def case_job(item):
    robot, trial, seed = item
    case = recreate(robot, trial)
    _, directions, _ = planned_directions(robot, case)
    archive = read(HERE / "prior_design_cases.json")
    reference = case["reference"]
    radius = float(np.linalg.norm(reference[1], axis=1).max() + .09)
    truth = matrices(case["j_hold"], case["tests"])
    full_observations = matrices(case["j_cal"], case["catalog"]) + case["noises"][.01]
    rows = []
    for bound in BOUNDS:
        recorded = [r for r in archive if (r["robot"], r["trial"], r["bound"], r["design"])
                    == (case["name"], trial, bound, "degraded")]
        assert len(recorded) == 2
        candidate = recorded[0]["candidate"]
        assert all(row["candidate"] == candidate for row in recorded)
        a = directions[candidate]
        # The count uses the matrix metric as in the original degraded-prior control.
        # The action metric uses the orthogonal complement of the actual fit directions.
        unseen = np.linalg.qr(a, mode="complete")[0][:, a.shape[1]:]
        np.testing.assert_allclose(a.T @ unseen, 0., atol=1e-13)
        primary = perturb_start(reference, bound, .09,
                                np.random.default_rng(2026111954 + 500*robot + 20*trial))
        seeds = np.random.SeedSequence([seed, robot, trial, bound])
        rng = np.random.default_rng(seeds)
        conditions = {
            "prior_and_wide_restart": [primary, perturb_start(primary, bound, .09, rng)],
            "isotropic": [isotropic_start(len(reference[0]), radius, rng) for _ in range(2)],
        }
        for initializer, starts in conditions.items():
            outcome = fit_case(case["q"], a, case["catalog"], full_observations @ a,
                               case["held_q"], case["tests"], truth, unseen, starts)
            assert outcome["starts"] == 2
            rows.append(dict(robot=case["name"], trial=trial, bound=bound, candidate=candidate,
                             initializer=initializer, restart_point_radius_m=radius, fit=outcome))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--seed", type=int, default=START_SEED)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job, (robot, trial, args.seed))
                for robot in range(2) for trial in range(args.trials)]
        rows = []
        for job in as_completed(jobs):
            block = job.result()
            rows.extend(block)
            print(block[0]["robot"], block[0]["trial"], len(rows),
                  "initialization comparisons complete", flush=True)
    rows.sort(key=lambda row: (row["robot"], row["trial"], row["bound"], row["initializer"]))
    write(args.output, dict(seed=args.seed, budget_per_start=1600, rows=rows))


if __name__ == "__main__":
    main()
