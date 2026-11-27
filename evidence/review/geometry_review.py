"""Test degraded geometric starts and sign separation in terminal-joint probes."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
from scipy.special import ndtr
from scipy.spatial.transform import Rotation

from study_common import contrast_noise, forward_urdf, matrices, recreate, score, write
from directional_recovery import shared_reference_whitener
from fit_chain import calibrate_chain, initialize_from_data
from matched_review import choose_fit
from robot_geometry import reverse_axes, serial_body


def perturb_start(reference, angular_degrees, point_bound, rng):
    axes, points = reference
    direction = np.cross(axes, rng.normal(size=axes.shape))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    rad = np.deg2rad(angular_degrees) * rng.uniform(0.25, 1, size=(len(axes), 1))
    moved_axes = Rotation.from_rotvec(rad * direction).apply(axes)
    moved_points = points + rng.uniform(-point_bound, point_bound, points.shape)
    moved_points -= (moved_points * moved_axes).sum(1, keepdims=True) * moved_axes
    return moved_axes, moved_points


def initialization_job(task):
    robot, trial = task
    case = recreate(robot, trial)
    a = case["directions"]["4loads_2dirs"]
    loads = case["catalog"]
    clean = matrices(case["j_cal"], loads)
    truth = matrices(case["j_hold"], case["tests"])
    candidates = {(angle, point): [
        perturb_start(case["reference"], angle, point,
                      np.random.default_rng(2026111954 + 500*robot + 20*trial + start))
        for start in (0, 1)
    ] for angle, point in ((15., .03), (45., .09))}
    rows = []
    for sigma in (.001, .01):
        observed = (clean + case["noises"][sigma]) @ a
        for (angle, point), starts in candidates.items():
            result = choose_fit(case, truth, loads, a, observed, starts)
            rows.append(dict(robot=case["name"], trial=trial, sigma=sigma,
                             angular_bound_deg=angle, point_bound_m=point,
                             realized_errors=[score(s, case, truth)["R_M"] for s in starts],
                             result=result))
    return rows


def terminal_job(task):
    robot, trial, sigma, repetitions, angles, moving_chart = task
    if moving_chart:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "followup"))
        from recentered_fit import best_recentered
    case = recreate(robot, trial)
    truth = matrices(case["j_hold"], case["tests"])
    a = case["directions"]["4loads_2dirs"]
    loads = case["catalog"]
    n = len(case["reference"][0])
    q0 = case["q"][0].copy()
    q0[-1] = -1.1
    home = forward_urdf(case["changed"], q0[None], case["mount"], case["offset"])
    correct = reverse_axes(q0[None], home)
    opposite = reverse_axes(q0[None], -home)
    weight = shared_reference_whitener(len(loads))
    rng = np.random.default_rng(2026112020 + 100*robot + trial)
    noise = np.asarray([[contrast_noise(rng, 4, n, n, sigma) for _ in range(2)]
                        for _ in range(repetitions)]).mean(axis=0)
    rows = []
    for delta in angles:
        q = np.vstack((q0, q0))
        q[1, -1] += delta
        known = matrices(forward_urdf(case["changed"], q, case["mount"], case["offset"]), loads)
        true_response = known @ a
        np.testing.assert_allclose(matrices(serial_body(q, *correct), loads) @ a,
                                   true_response, atol=1e-12, rtol=1e-12)
        alternative = matrices(serial_body(q, *opposite), loads) @ a
        diff = np.einsum("ml,qlik->qmik", weight, alternative - true_response)
        distance = np.linalg.norm(diff.ravel())
        probability = float(ndtr(-distance * np.sqrt(repetitions) / (np.sqrt(2) * sigma)))
        observed = (known + noise) @ a
        if moving_chart:
            starts, eligible = initialize_from_data(q, loads, a, observed)
            if starts:
                geometry, selected, stages = best_recentered(q, loads, a, observed, starts)
                outcome = score(geometry, case, truth)
            else:
                selected, stages, outcome = None, [], None
        else:
            fitted = calibrate_chain(q, loads, a, observed).get("from_torques")
            outcome = None if fitted is None else score(fitted["geometry"], case, truth)
        rows.append(dict(
            robot=case["name"], trial=trial, step_rad=float(delta), sigma=sigma,
            clean_orientation_gap_Nm=float(np.sqrt(np.mean(diff ** 2))),
            ideal_known_candidates_error_probability=probability,
            two_pose_fit=outcome,
        ))
        if repetitions != 1:
            rows[-1]["repetitions"] = repetitions
        if moving_chart:
            rows[-1].update(solver="moving_chart", eligible=eligible, selected=selected, stages=stages)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", choices=("initialization", "terminal"), required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--trials", type=int, default=12)
    p.add_argument("--terminal-noise", type=float, default=.001)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--terminal-angles", type=float, nargs="+", default=(.05, .5, np.pi - .05, np.pi))
    p.add_argument("--moving-chart", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    function = initialization_job if args.study == "initialization" else terminal_job
    if args.terminal_noise <= 0 or args.repetitions < 1:
        p.error("terminal noise and repetitions must be positive")
    tasks = [(r, t) for r in range(2) for t in range(args.trials)]
    if args.study == "terminal":
        tasks = [(r, t, args.terminal_noise, args.repetitions, args.terminal_angles, args.moving_chart)
                 for r, t in tasks]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(function, tasks):
            rows.extend(result)
            if len(rows) % 24 == 0:
                print(args.study, len(rows), "comparison rows", flush=True)
    write(args.output, rows)


if __name__ == "__main__":
    main()
