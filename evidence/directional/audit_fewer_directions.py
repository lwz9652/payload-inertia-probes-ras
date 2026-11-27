"""Replay the nominal-geometry fit with one and two directions and three loads."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from directional_recovery import contrast_noise, shared_reference_whitener
from fit_chain import axes_from_home, refine_axes
from probe_benchmark import BOXES, NOISE, box_inertia, holdout_boxes, matrices, relative_rmse, write
from robot_geometry import forward_urdf, nominal_models, perturb_chain, serial_body


def choose_directions(q, reference, inertias, seed):
    """Choose one and two directions using only nominal response sensitivity."""
    w, p = reference
    pairs = []
    step = 1e-5
    for joint in range(len(w)):
        first = np.cross(w[joint], np.eye(3)[np.argmin(np.abs(w[joint]))])
        first /= np.linalg.norm(first)
        for mode in ("axis", "point"):
            for d in (first, np.cross(w[joint], first)):
                contributions = []
                for sign in (1, -1):
                    axes, points = w.copy(), p.copy()
                    if mode == "axis":
                        axes[joint] += sign * step * d
                        axes[joint] /= np.linalg.norm(axes[joint])
                    else:
                        points[joint] += sign * step * d
                    points[joint] -= (points[joint] @ axes[joint]) * axes[joint]
                    contributions.append(matrices(serial_body(q, axes, points), inertias))
                pairs.append((contributions[0] - contributions[1]) / (2 * step))
    jac = np.stack(pairs, axis=-1)
    jac = np.einsum("ml,qlijd->qmijd", shared_reference_whitener(len(inertias)), jac)
    random = np.random.default_rng(seed)
    results = {}
    for count in (1, 2):
        candidates = [np.linalg.qr(random.normal(size=(len(w), count)))[0] for _ in range(80)]
        merit = []
        for a in candidates:
            probes = np.einsum("qmijd,jk->qmikd", jac, a, optimize=True).reshape(-1, jac.shape[-1])
            merit.append(float(np.linalg.svd(probes, compute_uv=False)[-1]))
        selected = int(np.argmax(merit))
        results[count] = dict(directions=candidates[selected].tolist(), min_singular_value=merit[selected])
    return results


def read_json(path):
    return json.loads(path.read_text())


def evaluate(output):
    root = Path(__file__).resolve().parent
    summary = read_json(root / "confirmation/report.json")
    inertias = np.asarray([box_inertia(*load) for load in BOXES[:3]])
    results = []
    for robot_id, (snapshot, chain, bounds) in enumerate(nominal_models()):
        previous = summary["robots"][robot_id]
        protocol = previous["protocol"]
        low, high = bounds.T
        assert snapshot["source_sha256"] == protocol["urdf_sha256"]
        nj = len(bounds)
        reference = axes_from_home(forward_urdf(chain, np.zeros((1, nj)))[0])
        design = np.random.default_rng(2026111701 + robot_id).uniform(low, high, (16, nj))
        new_designs = choose_directions(design, reference, inertias, 2026111749 + robot_id)
        rows = []
        for trial in range(protocol["repeats"]):
            rng = np.random.default_rng(2026111920 + 100 * robot_id + protocol["seed_offset"] + trial)
            q = rng.uniform(low, high, (protocol["calibration_configurations"], nj))
            q_new = rng.uniform(low, high, (protocol["heldout_configurations"], nj))
            new_loads = holdout_boxes(rng)
            rot = Rotation.from_rotvec(rng.uniform(-.08, .08, 3)).as_matrix()
            offset = rng.uniform(-.035, .035, 3)
            perturbed = perturb_chain(chain, rng)
            clean = matrices(forward_urdf(perturbed, q, rot, offset), inertias)
            truth = matrices(forward_urdf(perturbed, q_new, rot, offset), new_loads)
            for sigma in NOISE:
                # Draw all four original motions before retaining the same three loads.
                errors = np.asarray([contrast_noise(rng, 4, nj, nj, sigma)[:3] for _ in q])
                if sigma != 0.01:
                    continue
                arms = {str(k): np.asarray(item["directions"]) for k, item in new_designs.items()}
                arms["3"] = np.asarray(protocol["directions"][0]["matrix"])
                for name, directions in arms.items():
                    fit = refine_axes(q, inertias, directions, (clean + errors) @ directions,
                                      *reference, max_evaluations=600)
                    observed = matrices(serial_body(q_new, *fit["geometry"]), new_loads)
                    error = relative_rmse(observed, truth)
                    if name == "3":
                        old = next(r["relative_error"] for r in previous["scores"]
                                   if (r["trial"], r["arm"], r["start"], r["noise_SD_Nm"])
                                   == (trial, "3loads_3dirs", "from_URDF", sigma))
                        np.testing.assert_allclose(error, old, atol=1e-9, rtol=1e-6)
                    rows.append(dict(trial=trial, directions=int(name), relative_RM=error))
        per_count = {}
        for n in (1, 2, 3):
            errors = [r["relative_RM"] for r in rows if r["directions"] == n]
            per_count[str(n)] = dict(
                reads_per_configuration=8 * n,
                median_percent=float(100 * np.median(errors)),
                maximum_percent=float(100 * max(errors)),
            )
        print(protocol["robot"], per_count, flush=True)
        results.append(dict(robot=protocol["robot"], directions=new_designs, data=rows, summary=per_count))
    write(output, dict(
        pairing="Same 24 distributed geometries, held-out cases, three loads and 0.01 Nm noise as confirmation.",
        selection="One or two bases selected among 80 candidates by nominal calibration-Jacobian "
                  "minimum singular value on 16 nominal design poses. Three directions from confirmation.",
        results=results,
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.output)
