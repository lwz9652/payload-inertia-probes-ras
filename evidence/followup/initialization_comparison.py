"""Compare starting chains with the same recentered optimizer and paired torque data."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from recentered_fit import best_recentered
from fit_chain import axes_from_home, initialize_from_data
from geometry_review import perturb_start
from study_common import forward_urdf, matrices, recreate, score, write


ROUGH = ((15, .03), (45, .09), (90, .09), (135, .09))


def case_job(job):
    robot, trial = job
    case = recreate(robot, trial)
    q = case["q"]
    load = case["catalog"]
    directions = case["directions"]["4loads_2dirs"]
    clean = matrices(case["j_cal"], load)
    truth = matrices(case["j_hold"], case["tests"])
    home = forward_urdf(case["changed"], np.zeros((1, q.shape[1])), case["mount"], case["offset"])[0]
    true_axes, _ = axes_from_home(home)
    rough = {}
    for bound, offset in ROUGH:
        rough[bound] = [perturb_start(case["reference"], bound, offset,
                                     np.random.default_rng(2026111954 + 500*robot + 20*trial + k))
                        for k in (0, 1)]
    rows = []
    for sigma in (.001, .01):
        obs = (clean + case["noises"][sigma]) @ directions
        torque, _ = initialize_from_data(q, load, directions, obs)
        if len(torque) != 2:
            raise ValueError("Two torque-derived starts required for this paired audit")
        for label, starts in [("torque", torque), *((str(key), val) for key, val in rough.items())]:
            geo, choice, info = best_recentered(q, load, directions, obs, starts)
            max_axis_errors = [float(np.rad2deg(np.arccos(np.clip((item[0]*true_axes).sum(1), -1, 1))).max())
                               for item in starts]
            axes, points = geo
            np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1., atol=1e-12)
            np.testing.assert_allclose((axes*points).sum(1), 0., atol=1e-12)
            rows.append(dict(
                robot=case["name"], trial=trial, sigma=sigma, starts=label,
                starts_max_axis_degrees=max_axis_errors, winner=choice, stages=info,
                initial=score(starts[choice], case, truth), final=score(geo, case, truth),
            ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(case_job, (robot, trial)) for robot in range(2)
                   for trial in range(args.trials)]
        cases = []
        for job in as_completed(pending):
            block = job.result()
            cases.extend(block)
            print(block[0]["robot"], block[0]["trial"], len(cases), "fits complete", flush=True)
    cases.sort(key=lambda r:(r["robot"], r["trial"], r["sigma"], r["starts"]))
    write(args.output, cases)


if __name__ == "__main__":
    main()
