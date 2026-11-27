"""Compare load-selection rules over every candidate in the original paired pools."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from coordinates import FOLLOWUP, directional_residual, finite_difference
from study_common import box_inertia, forward_urdf, matrices, read, recreate, write
from fit_chain import initialize_from_data
from matched_review import choose_fit
from random_payload_review import margins


def candidate_pool(trial):
    rng = np.random.default_rng(2026112031 + trial)
    return [
        box_inertia(rng.uniform(.65, 2.5),
                    rng.uniform([-.11, -.11, .025], [.11, .11, .20]),
                    rng.uniform(.045, .21, 3), rng.uniform(-120, 120, 3))
        for _ in range(16)
    ]


def nominal_scores(case, inertias, directions):
    planned = case["q"][:8]
    nominal = forward_urdf(case["chain"], planned)
    design_actions = matrices(nominal, inertias) @ directions
    rank = margins(inertias, directions, design_actions)
    response = directional_residual(planned, inertias, directions, np.zeros_like(design_actions))
    center = response(case["reference"])
    s = np.linalg.svd(finite_difference(response, case["reference"], center)[0], compute_uv=False)
    if s[-1] <= s[0] * 1e-10:
        raise ValueError("Nominal sensitivity has deficient numerical rank")
    return dict(
        product=float(np.quantile(rank.prod(axis=1), .2)),
        sigma6=float(np.quantile(rank[:, 0], .2)),
        operator=float(np.quantile(rank[:, 1], .2)),
        d_opt=float(2*np.log(s).sum()),
        sensitivity_min=float(s[-1]),
    )


def case_job(job):
    robot, trial = job
    case = recreate(robot, trial)
    probe = case["directions"]["3loads_3dirs"]
    base = case["catalog"][:3]
    truth = matrices(case["j_hold"], case["tests"])
    previous = read(FOLLOWUP.parent / "review" / "fourth_load_cases.json")
    controls = {row["index"]:row for row in previous
                if (row["robot"], row["trial"]) == (case["name"], trial)}
    outputs = []
    for index, payload in enumerate(candidate_pool(trial)):
        inertias = np.concatenate((base, payload[None]))
        scores = nominal_scores(case, inertias, probe)
        projected = (matrices(case["j_cal"], inertias) + case["noises"][.01]) @ probe
        starts, used = initialize_from_data(case["q"], inertias, probe, projected)
        fit = choose_fit(case, truth, inertias, probe, projected, starts)
        if index in controls:
            expected = controls[index]
            np.testing.assert_allclose(scores["product"], expected["candidate_scores"][index], rtol=1e-9)
            np.testing.assert_allclose(fit["R_M"], expected["fit"]["R_M"], rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(fit["R_A"], expected["fit"]["R_A"], rtol=1e-5, atol=1e-8)
        outputs.append(dict(robot=case["name"], trial=trial, index=index, used_poses=used,
                            scores=scores, fit=fit))
    top = max(outputs, key=lambda r:r["scores"]["product"])
    chosen, = [r for r in previous if (r["robot"],r["trial"],r["design"]) ==
               (case["name"],trial,"chosen")]
    assert top["index"] == chosen["index"]
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--trials", type=int, default=12)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        queue = [pool.submit(case_job, (robot, trial))
                 for robot in range(2) for trial in range(args.trials)]
        rows = []
        for job in as_completed(queue):
            block = job.result()
            rows.extend(block)
            print(block[0]["robot"], block[0]["trial"], len(rows), "fits complete", flush=True)
    rows.sort(key=lambda r:(r["robot"],r["trial"],r["index"]))
    write(args.output, rows)


if __name__ == "__main__":
    main()
