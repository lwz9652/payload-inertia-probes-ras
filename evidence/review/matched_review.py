"""Separate measurement error, local initialization and global fit on paired observations."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations, product
from pathlib import Path
from time import perf_counter

import numpy as np

from study_common import ORIGINAL, errors, matrices, read, recreate, relative_rmse, score, write
from directional_recovery import recover_initial, shared_reference_whitener
from fit_chain import initialize_from_data, refine_axes
from robot_geometry import reverse_axes


SIGNS = np.asarray([(1,) + rest for rest in product((-1, 1), repeat=5)])


def spectral_complete(inertias, observation):
    """Separate complete-matrix method using Gram whitening and six load orderings."""
    symmetric = (observation + observation.swapaxes(-1, -2)) * 0.5
    whiten = shared_reference_whitener(len(inertias))
    best_loss, best_factor = np.inf, None
    for anchor, align, _ in permutations(range(len(inertias))):
        eig, vectors = np.linalg.eigh(symmetric[anchor])
        if eig[-6] <= 0:
            continue
        w = np.sqrt(eig[-6:])[:, None] * vectors[:, -6:].T
        wi = w.T / eig[-6:]
        metric, axes = np.linalg.eigh(inertias[anchor])
        invroot = (axes / np.sqrt(metric)) @ axes.T
        _, u = np.linalg.eigh(invroot @ inertias[align] @ invroot)
        _, v = np.linalg.eigh(wi.T @ symmetric[align] @ wi)
        factor = invroot @ (u[None] * SIGNS[:, None]) @ v.T @ w
        diff = factor[:, None].swapaxes(-1, -2) @ inertias @ factor[:, None] - observation
        weighted = np.einsum("ml,slij->smij", whiten, diff)
        loss = np.sum(weighted ** 2, axis=(1, 2, 3))
        index = int(np.argmin(loss))
        if loss[index] < best_loss:
            best_loss, best_factor = loss[index], factor[index]
    return best_factor


def spectral_starts(q, factors):
    indices = [k for k, factor in enumerate(factors) if factor is not None and
               0.1 < np.linalg.norm(factor[:3, -1]) < 10.0]
    if not indices:
        return []
    j = np.asarray([factors[k] for k in indices])
    last = j[:, :3, -1].copy()
    last /= np.linalg.norm(last, axis=1, keepdims=True)
    axis = np.linalg.svd(last, full_matrices=False)[2][0]
    sign = np.where(last @ axis >= 0, 1., -1.)
    starts = []
    for s in (1, -1):
        try:
            starts.append(reverse_axes(q[indices], j * (s * sign)[:, None, None]))
        except ValueError:
            continue
    return starts


def choose_fit(case, clean_test, i, a, observed, starts):
    results = []
    for axes, points in starts:
        tick = perf_counter()
        fit = refine_axes(case["q"], i, a, observed, axes, points, max_evaluations=600)
        fit["seconds"] = perf_counter() - tick
        results.append((fit, (axes, points)))
    if not results:
        return None
    fit, initial = min(results, key=lambda item: item[0]["train_rmse"])
    return dict(score(fit["geometry"], case, clean_test),
                initial=score(initial, case, clean_test),
                train_rmse=fit["train_rmse"],
                evaluations=sum(item[0]["evaluations"] for item in results),
                seconds=sum(item[0]["seconds"] for item in results),
                starts=len(results), converged=fit["converged"])


def local_scores(factors, truth, test_inertias):
    kept = [i for i, factor in enumerate(factors) if factor is not None]
    if not kept:
        return dict(usable=0, cases=[])
    values = [relative_rmse(matrices(factors[j][None], test_inertias)[0], truth[j])
              for j in kept]
    return dict(usable=len(kept), cases=values,
                R_M=float(np.median(values)), worst_R_M=max(values))


def case_job(task):
    robot, trial = task
    case = recreate(robot, trial)
    catalog = case["catalog"]
    truth = matrices(case["j_hold"], case["tests"])
    local_truth = matrices(case["j_cal"], case["tests"])
    clean = matrices(case["j_cal"], catalog)
    d = case["directions"]
    identity = d["3loads_full"]
    designs = [
        ("3x3", 3, d["3loads_3dirs"]), ("3xfull", 3, identity),
        ("4x2", 4, d["4loads_2dirs"]), ("4x3", 4, d["3loads_3dirs"]),
        ("4xfull", 4, identity),
    ]
    rows = []
    prior = read(ORIGINAL / "confirmation/report.json")["robots"][robot]["scores"]
    original_names = {"3x3": "3loads_3dirs", "3xfull": "3loads_full", "4x2": "4loads_2dirs"}
    for name, count, a in designs:
        inertia = catalog[:count]
        clean_actions = clean[:, :count] @ a
        exact_margins = [recover_initial(inertia, a, row)[1] for row in clean_actions]
        for sigma in (.001, .01):
            observed = (clean[:, :count] + case["noises"][sigma][:, :count]) @ a
            factors, seconds = [], []
            for obs in observed:
                tick = perf_counter()
                try:
                    factor, _ = recover_initial(inertia, a, obs)
                except (ValueError, np.linalg.LinAlgError):
                    factor = None
                seconds.append(perf_counter() - tick)
                factors.append(factor)
            row = dict(robot=case["name"], trial=trial, sigma=sigma, design=name,
                       loads=count, directions=a.shape[1], reads=2*(count+1)*a.shape[1],
                       exact_rank_six_values=[x["rank6_retained_singular_value"] for x in exact_margins],
                       exact_operator_margins=[x["operator_margin"] for x in exact_margins],
                       local=local_scores(factors, local_truth, case["tests"]),
                       local_seconds=float(np.median(seconds)))
            if a.shape[1] == len(identity):
                sym = (observed + observed.swapaxes(-1, -2)) / 2
                eigenvalues, basis = np.linalg.eigh(sym)
                clipped = eigenvalues.clip(0)
                clipped[..., :-6] = 0
                restored = (basis * clipped[..., None, :]) @ basis.swapaxes(-1, -2)
                row["independent_calibration_matrices_R_M"] = relative_rmse(restored, clean[:, :count])
            starts, _ = initialize_from_data(case["q"], inertia, a, observed)
            row["kernel"] = choose_fit(case, truth, inertia, a, observed, starts)
            row["nominal"] = choose_fit(case, truth, inertia, a, observed, [case["reference"]])
            if name in original_names:
                for source, key in (("from_torques", "kernel"), ("from_URDF", "nominal")):
                    archived = next(r["relative_error"] for r in prior if
                                    (r["trial"], r["arm"], r["start"], r["noise_SD_Nm"])
                                    == (trial, original_names[name], source, sigma))
                    np.testing.assert_allclose(row[key]["R_M"], archived, rtol=1e-5, atol=1e-8)
            if name == "3xfull":
                beginning = perf_counter()
                full_factors = [spectral_complete(inertia, obs) for obs in observed]
                row["spectral_local_seconds_per_configuration"] = (perf_counter() - beginning) / len(observed)
                row["spectral_local"] = local_scores(full_factors, local_truth, case["tests"])
                guesses = spectral_starts(case["q"], full_factors)
                row["spectral"] = choose_fit(case, truth, inertia, a, observed, guesses)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trial-count", type=int, default=12)
    args = ap.parse_args()
    args.output.mkdir()
    tasks = [(robot, trial) for robot in range(2) for trial in range(args.trial_count)]
    full = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in pool.map(case_job, tasks):
            full.extend(batch)
            print(batch[0]["robot"], batch[0]["trial"], len(full), "rows", flush=True)
    write(args.output / "cases.json", full)


if __name__ == "__main__":
    main()
