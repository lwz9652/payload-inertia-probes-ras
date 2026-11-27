"""Vary load diversity and probe design with paired poses, inertias and torque noise."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from study_common import box_inertia, contrast_noise, forward_urdf, matrices, recreate, relative_rmse, write
from directional_recovery import recover_initial


def margins(inertias, directions, observations):
    value = []
    for row in observations:
        try:
            _, stats = recover_initial(inertias, directions, row)
        except (ValueError, np.linalg.LinAlgError):
            value.append((0., 0.))
        else:
            value.append((stats["rank6_retained_singular_value"], stats["operator_margin"]))
    return np.asarray(value)


def load_draw(group):
    rng = np.random.default_rng(2026111911 + group)
    masses = rng.uniform(.65, 2.5, 4)
    centers = rng.uniform([-.11, -.11, .025], [.11, .11, .20], (4, 3))
    sides = rng.uniform(.045, .21, (4, 3))
    rotations = rng.uniform(-120, 120, (4, 3))
    return masses, centers, sides, rotations


def probe_draws(robot, group, joint_count):
    rng = np.random.default_rng(2026111937 + 100 * robot + group)
    candidates = np.asarray([np.linalg.qr(rng.normal(size=(joint_count, 2)))[0]
                             for _ in range(32)])
    return candidates, rng


def group_job(task):
    robot, group = task
    case = recreate(robot, group % 12)
    masses, centers, sides, rotations = load_draw(group)
    center_mean = centers.mean(axis=0)
    nominal_q = case["q"][:8]
    reference = forward_urdf(case["chain"], nominal_q)
    jac = case["j_cal"][8:16]
    target = case["tests"][:16]
    truth = matrices(jac, target)
    n = jac.shape[-1]
    candidates, probes_rng = probe_draws(robot, group, n)
    noises = {sigma: np.asarray([contrast_noise(probes_rng, 4, n, n, sigma) for _ in jac])
              for sigma in (.001, .01)}
    rows = []
    for spread in (1., .1, 0.):
        centered = center_mean + spread * (centers - center_mean)
        inertias = np.asarray([box_inertia(m, c, l, a)
                              for m, c, l, a in zip(masses, centered, sides, rotations, strict=True)])
        nominal = matrices(reference, inertias)
        clean = matrices(jac, inertias)
        merits = [np.quantile(np.prod(margins(inertias, d, nominal @ d), axis=1), .2)
                  for d in candidates]
        for design, index in (("random", 0), ("nominal_rank", int(np.argmax(merits)))):
            a = candidates[index]
            exact = margins(inertias, a, clean @ a)
            for sigma, noise in noises.items():
                values = []
                for position, y in enumerate((clean + noise) @ a):
                    try:
                        factor, _ = recover_initial(inertias, a, y)
                    except (ValueError, np.linalg.LinAlgError):
                        values.append(None)
                    else:
                        values.append(relative_rmse(matrices(factor[None], target)[0], truth[position]))
                rows.append(dict(
                    robot=case["name"], load_group=group, design=design, spread_scale=spread, sigma=sigma,
                    center_rms_m=float(np.linalg.norm(centered-center_mean)/2.),
                    exact_rank_six_values=exact[:, 0].tolist(), exact_operator_margins=exact[:, 1].tolist(),
                    selected_candidate=index, nominal_margin_product=float(merits[index]),
                    heldout_payload_errors=values,
                ))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--groups", type=int, default=40)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    tasks = [(i, g) for i in range(2) for g in range(args.groups)]
    full = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for group in pool.map(group_job, tasks):
            full.extend(group)
            if len(full) % 120 == 0:
                print(len(full) // 12, "load geometries scored", flush=True)
    write(args.output, full)


if __name__ == "__main__":
    main()
