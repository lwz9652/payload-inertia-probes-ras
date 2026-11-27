"""Freeze probes on nominal poses and evaluate recovery on new robot configurations."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from directional_recovery import contrast_noise, recover_initial
from fit_chain import axes_from_home, calibrate_chain, fit_attachment
from robot_geometry import forward_urdf, move_attachment, nominal_models, perturb_chain, serial_body, skew


# Three diverse manufactured calibration boxes and an optional fourth fixture.
# Spatial coordinates refer to the same attachment frame.
BOXES = [
    (1.45, [-0.10, 0.03, 0.08], [0.18, 0.049, 0.062], [21, -35, 48]),
    (2.25, [0.082, -0.096, 0.095], [0.07, 0.195, 0.047], [-33, 42, 15]),
    (1.8, [0.045, 0.109, 0.155], [0.11, 0.054, 0.184], [42, 30, -54]),
    (2.35, [-0.058, -0.095, 0.182], [0.125, 0.088, 0.162], [-40, -27, 72]),
]
NOISE = (0.0, 0.001, 0.010)
AMPLITUDE = 1.0
CALIBRATION_CONFIGURATIONS = 32
VALIDATION_CONFIGURATIONS = 128
VALIDATION_PAYLOADS = 48


def box_inertia(mass, center, side, degrees):
    center = np.asarray(center)
    side = np.asarray(side)
    rotated = Rotation.from_euler("xyz", degrees, degrees=True).as_matrix()
    ic = rotated @ np.diag(mass / 12 * (np.sum(side ** 2) - side ** 2)) @ rotated.T
    io = ic + mass * ((center @ center) * np.eye(3) - np.outer(center, center))
    h = skew(mass * center)
    return np.block([[io, h], [-h, mass * np.eye(3)]])


def holdout_boxes(rng):
    return np.asarray([
        box_inertia(
            rng.uniform(.45, 2.6), rng.uniform([-.115, -.115, .02], [.115, .115, .205]),
            rng.uniform(.042, .23, 3), rng.uniform(-180, 180, 3),
        ) for _ in range(VALIDATION_PAYLOADS)
    ])


def matrices(jacobians, inertias):
    blocks = jacobians[:, None]
    return blocks.swapaxes(-1, -2) @ inertias @ blocks


def relative_rmse(prediction, target):
    return float(np.linalg.norm((prediction - target).ravel()) / np.linalg.norm(target.ravel()))


def rank_screen(jacobians, inertias, directions):
    responses = matrices(jacobians, inertias) @ directions
    margins = []
    for y in responses:
        try:
            _, s = recover_initial(inertias, directions, y)
            margins.append(s["rank6_retained_singular_value"] * s["operator_margin"])
        except ValueError:
            margins.append(0.)
    return float(np.quantile(margins, .2))


def freeze_directions(jacobians, inertias, size, rng):
    joints = jacobians.shape[-1]
    proposals = [np.linalg.qr(rng.normal(size=(joints, size)))[0] for _ in range(80)]
    merit = [rank_screen(jacobians, inertias, arm) for arm in proposals]
    index = int(np.argmax(merit))
    return proposals[index], merit[index]


def append(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def noise_joint(clean, directions, base_noise):
    # A single isotropic noise realization on orthogonal joint directions
    # couples the alternatives. Projection preserves white noise per direction.
    return (clean + base_noise) @ directions


def point_recovery(jacobians, inertias, direction, test_inertias):
    observations = matrices(jacobians, inertias) @ direction
    truth = matrices(jacobians, test_inertias)
    errors, margins = [], []
    for pose, observed in enumerate(observations):
        j, receipt = recover_initial(inertias, direction, observed)
        fitted = matrices(j[None], test_inertias)[0]
        errors.append(relative_rmse(fitted, truth[pose]))
        margins.append(receipt["operator_margin"])
    return dict(
        count=len(errors), max_relative_error=max(errors), median_relative_error=float(np.median(errors)),
        min_operator_margin=min(margins), error_by_pose=errors,
    )


def simulate_robot(model_id, model, catalog, repeats, seed_offset, distributed, out):
    snapshot, chain, bounds = model
    robot = snapshot["robot"]
    lower, upper = bounds.T
    joints = len(bounds)
    design_rng = np.random.default_rng(2026111701 + model_id)
    reference_q = design_rng.uniform(lower, upper, (16, joints))
    reference_j = forward_urdf(chain, reference_q)
    reference = axes_from_home(forward_urdf(chain, np.zeros((1, joints)))[0])
    tri, tri_merit = freeze_directions(reference_j, catalog[:3], 3, design_rng)
    duo, duo_merit = freeze_directions(reference_j, catalog[:4], 2, design_rng)
    arms = [
        ("3loads_3dirs", 3, tri, tri_merit),
        ("4loads_2dirs", 4, duo, duo_merit),
        ("3loads_full", 3, np.eye(joints), None),
    ]
    joint_span = np.concatenate((tri, duo), axis=1)
    unprobed = np.linalg.svd(joint_span, full_matrices=True)[0][:, 5:]
    protocol = dict(
        robot=robot, urdf_source=snapshot["source_name"], urdf_sha256=snapshot["source_sha256"],
        design_rule="Maximize the 20th percentile of the product of rank-six torque margin and nullspace gap "
                    "over sixteen nominal design poses. Fixed independent seed, 80 orthonormal candidates.",
        directions=[dict(name=name, loads=loads, matrix=direction.tolist(), merit=merit,
                         torque_vectors_per_configuration=2 * direction.shape[1] * (loads + 1))
                    for name, loads, direction, merit in arms],
        calibration_payload_boxes=BOXES, repeats=repeats, raw_torque_noise_SD_Nm=NOISE,
        amplitude_rad_s2=AMPLITUDE, calibration_configurations=CALIBRATION_CONFIGURATIONS,
        heldout_configurations=VALIDATION_CONFIGURATIONS, heldout_payloads=VALIDATION_PAYLOADS,
        independent_truth="URDF forward geometry with random unknown tool mounting SE(3) offsets. "
                          "Recovery uses reverse screw recursion and observes only directional torques.",
        distributed_geometry=distributed, seed_offset=seed_offset,
        unprobed_test_directions=unprobed.tolist(),
    )
    write(out / f"{robot}_protocol.json", protocol)
    rows, pointwise = [], []
    for trial in range(repeats):
        rng = np.random.default_rng(2026111920 + 100 * model_id + seed_offset + trial)
        q_cal = rng.uniform(lower, upper, (CALIBRATION_CONFIGURATIONS, joints))
        q_eval = rng.uniform(lower, upper, (VALIDATION_CONFIGURATIONS, joints))
        test_inertias = holdout_boxes(rng)
        rot = Rotation.from_rotvec(rng.uniform(-.08, .08, 3)).as_matrix()
        offset = rng.uniform(-.035, .035, 3)
        case_chain = perturb_chain(chain, rng) if distributed else chain
        j_cal = forward_urdf(case_chain, q_cal, rot, offset)
        j_nominal_train = forward_urdf(chain, q_cal)
        m_cal = matrices(j_cal, catalog)
        j_eval = forward_urdf(case_chain, q_eval, rot, offset)
        j_nominal_eval = forward_urdf(chain, q_eval)
        for sig in NOISE:
            errors = np.asarray([
                contrast_noise(rng, len(catalog), joints, joints, sig, AMPLITUDE) for _ in q_cal
            ])
            sealed = []
            for name, loads, direction, _ in arms:
                observed = noise_joint(m_cal[:, :loads], direction, errors[:, :loads])
                fitted = calibrate_chain(q_cal, catalog[:loads], direction, observed, reference)
                fitted["mounting_only"] = fit_attachment(j_nominal_train, catalog[:loads], direction, observed)
                sealed.append((name, fitted))
            # Evaluate after all starts and model choices for this trial are fixed.
            m_eval = matrices(j_eval, test_inertias)
            nominal = relative_rmse(matrices(j_nominal_eval, test_inertias), m_eval)
            for name, fitted in sealed:
                for start, fit in fitted.items():
                    if start == "mounting_only":
                        angle = Rotation.from_rotvec(fit["parameters"][:3]).as_matrix()
                        translation = np.asarray(fit["parameters"][3:])
                        j_pred = move_attachment(j_nominal_eval, angle, translation)
                        fit_receipt = fit
                    else:
                        j_pred = serial_body(q_eval, *fit["geometry"])
                        fit_receipt = {**fit, "geometry": [x.tolist() for x in fit["geometry"]]}
                    estimate = matrices(j_pred, test_inertias)
                    row = dict(robot=robot, trial=trial, arm=name, start=start, noise_SD_Nm=sig,
                               relative_error=relative_rmse(estimate, m_eval), nominal_error=nominal,
                               unprobed_direction_error=relative_rmse(estimate @ unprobed, m_eval @ unprobed),
                               fit=fit_receipt)
                    append(out / "cases.jsonl", row)
                    rows.append(row)
            print(robot, trial, sig, {k: np.round([r["relative_error"] for r in rows
                        if r["trial"] == trial and r["noise_SD_Nm"] == sig and r["arm"] == k], 5).tolist()
                        for k, *_ in arms}, flush=True)
        for name, loads, direction, _ in arms:
            point = point_recovery(j_eval[:24], catalog[:loads], direction, test_inertias)
            pointwise.append(dict(trial=trial, arm=name, **point))
    return dict(protocol=protocol, scores=rows, clean_pointwise=pointwise)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()
    args.output.mkdir()
    start = time.monotonic()
    catalog = np.asarray([box_inertia(*box) for box in BOXES])
    assert np.linalg.eigvalsh(catalog).min() > 0
    reports = [
        simulate_robot(index, model, catalog, args.repeats, args.seed_offset, args.distributed, args.output)
        for index, model in enumerate(nominal_models())
    ]
    write(args.output / "report.json", dict(robots=reports, elapsed_seconds=time.monotonic() - start))


if __name__ == "__main__":
    main()
