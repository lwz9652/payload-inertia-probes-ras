"""Use the same degraded geometry for probe design and direct chain fitting."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from controls_common import FOLLOWUP, fit_case
from geometry_review import perturb_start
from probe_benchmark import rank_screen
from study_common import matrices, read, recreate, serial_body, write


BOUNDS = (45, 90, 135)


def planned_directions(robot, case):
    snapshot = read(FOLLOWUP.parent / "review" / "kinematic_inputs.json")[robot]
    ranges = np.asarray(snapshot["bounds"])
    n = len(ranges)
    rng = np.random.default_rng(2026111701 + robot)
    poses = rng.uniform(*ranges.T, (16, n))
    # The three-direction pool consumes these draws first in the original design.
    rng.normal(size=(80, n, 3))
    directions = [np.linalg.qr(rng.normal(size=(n, 2)))[0] for _ in range(80)]
    nominal = design(case["reference"], poses, case["catalog"], directions)
    np.testing.assert_allclose(directions[nominal], case["directions"]["4loads_2dirs"], atol=1e-12)
    return poses, directions, nominal


def design(geometry, poses, inertias, directions):
    jacobians = serial_body(poses, *geometry)
    merits = [rank_screen(jacobians, inertias, a) for a in directions]
    return int(np.argmax(merits))


def case_job(item):
    robot, trial = item
    case = recreate(robot, trial)
    q = case["q"]
    loads = case["catalog"]
    poses, pool, nominal = planned_directions(robot, case)
    truth = matrices(case["j_hold"], case["tests"])
    observed = matrices(case["j_cal"], loads) + case["noises"][.01]
    summaries = []

    def add(angle, prior, candidate, initializers, starts=None):
        a = pool[candidate]
        result = fit_case(q, a, loads, observed @ a, case["held_q"], case["tests"],
                          truth, case["unseen"], starts)
        summaries.append(dict(robot=case["name"], trial=trial, bound=angle, design=prior,
                              candidate=candidate, initializer=initializers, fit=result))
        if result["starts"]:
            assert result["starts"] == 2

    add(None, "nominal", nominal, "torque")
    add(None, "random", 0, "torque")
    for angle in BOUNDS:
        primary = perturb_start(case["reference"], angle, .09,
                                np.random.default_rng(2026111954 + 500*robot + 20*trial))
        second = perturb_start(primary, 10., .02,
                               np.random.default_rng(2026112601 + 500*robot + trial))
        approximate = design(primary, poses, loads, pool)
        add(angle, "nominal", nominal, "geometry", [primary, second])
        add(angle, "degraded", approximate, "geometry", [primary, second])
        add(angle, "degraded", approximate, "torque")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job, (robot, i)) for robot in range(2) for i in range(args.trials)]
        rows = []
        for task in as_completed(jobs):
            block = task.result()
            rows.extend(block)
            print(block[0]["robot"], block[0]["trial"], len(rows), "conditions complete", flush=True)
    rows.sort(key=lambda r:(r["robot"],r["trial"],r["bound"] or 0,r["design"],r["initializer"]))
    write(args.output, rows)


if __name__ == "__main__":
    main()
