"""Replay the ideal Panda calibrations from bundled inputs and MuJoCo returns."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "followup"))

from fit_chain import calibrate_chain
from probe_benchmark import matrices, relative_rmse
from robot_geometry import serial_body, skew


PINS = {
    "prediction_inputs.npz": "46667eb13737cc20b033916e84cd39c547c66d5f886878cedf404bd32a945f84",
    "truth.npz": "798d2b0ca0da3ddcbae48aa9cb600dac03aa3dd082c6b5f31a6ec0597db5f31a",
}


def measured_actions(reads):
    half = (reads[..., 0, :] - reads[..., 1, :]) * .5
    return (half[:, 1:] - half[:, :1]).swapaxes(-2, -1)


def fit_all(terminal_only, refresh):
    label = "mujoco_terminal_check" if terminal_only else "mujoco_probe_check"
    directory = HERE / label
    archived = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    fits = []
    unseen = None
    for row in archived["rows"]:
        with np.load(directory / f"fold{row['fold']}_probes.npz", allow_pickle=False) as pack:
            q, velocity = pack["q"], pack["qd"]
            a, inertias, reads = pack["applied_directions"], pack["inertias"], pack["torques"]
        if refresh:
            fresh = signed_reads(q, velocity, a, inertias)
            np.testing.assert_allclose(fresh, reads, rtol=1e-13, atol=6e-13)
            reads = fresh
        fitted = calibrate_chain(q, inertias, a, measured_actions(reads))["from_torques"]
        assert fitted["pose_initializations"] == len(q)
        fits.append(dict(fold=row["fold"], geometry=[x.tolist() for x in fitted["geometry"]]))
        # Same QR basis as the original experiment. Reserve the remaining columns.
        basis = np.linalg.qr(np.random.default_rng(2026111728).normal(size=(7, 3)), mode="complete")[0]
        np.testing.assert_allclose(a, basis[:, :3], atol=8e-15, rtol=0)
        unseen = basis[:, 3:]
    return fits, unseen, archived


def load_pinned(name, keys):
    path = HERE / "mujoco_holdout" / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != PINS[name]:
        raise ValueError(f"Holdout pin failed for {name}")
    with np.load(path, allow_pickle=False) as archive:
        return tuple(archive[key] for key in keys)


def spatial_inertia(descriptor):
    mass = descriptor[0]
    h = skew(descriptor[1:4])
    xx, yy, zz, xy, xz, yz = descriptor[4:]
    rotational = np.asarray([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    return np.block([[rotational, h], [-h, mass * np.eye(3)]])


def signed_reads(q, velocity, a, inertias):
    from panda_engine import Panda, add_payload, source_xml

    original = source_xml()
    models = [Panda(original)] + [Panda(add_payload(original, inertia)) for inertia in inertias]
    return np.array([[[[arm.inverse(qi, vi, sign * direction) for sign in (1, -1)] for direction in a.T]
                      for arm in models] for qi, vi in zip(q, velocity, strict=True)])


def mass_grid(q, inertias):
    from panda_engine import Panda, add_payload, source_xml

    original = source_xml()
    base = Panda(original).mass_grid(q)
    return np.stack([Panda(add_payload(original, inertia)).mass_grid(q) - base
                     for inertia in inertias], axis=1)


def audit(output, refresh):
    protocols = [fit_all(terminal_only, refresh) for terminal_only in (False, True)]
    output.mkdir()
    seal = output / "fitted_screws.json"
    seal.write_text(json.dumps([group[0] for group in protocols], indent=2), encoding="utf-8")

    q, pi = load_pinned("prediction_inputs.npz", ("q", "pi"))
    truth, = load_pinned("truth.npz", ("deltaM",))
    inertias = np.asarray([spatial_inertia(row) for row in pi])
    if refresh:
        regenerated = mass_grid(q, inertias)
        np.testing.assert_allclose(regenerated, truth, atol=1e-12, rtol=1e-12)
        truth = regenerated
    replays = []
    for fits, directions, archived in protocols:
        configs = archived["calibration_configurations"]
        for row, prior in zip(fits, archived["rows"], strict=True):
            geometry = [np.asarray(value) for value in row["geometry"]]
            fitted = matrices(serial_body(q, *geometry), inertias)
            rm = relative_rmse(fitted, truth)
            ra = relative_rmse(fitted @ directions, truth @ directions)
            # Last digits at numerical precision vary with BLAS and payload diagonalization.
            assert max(rm, ra) < (1.5e-13 if configs == 2 else 8e-15)
            np.testing.assert_allclose(
                [rm, ra], [prior["full_relative_error"], prior["unprobed_relative_error"]],
                atol=2e-14 if configs == 2 else 6e-15, rtol=.10,
            )
            replays.append(dict(configurations=configs, split=row["fold"] + 1, R_M=rm, R_A=ra))
            print(f"{configs:2d} configurations, split {row['fold'] + 1}: R_M={rm:.4g} R_A={ra:.4g}")
    (output / "metrics.json").write_text(json.dumps(replays, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate raw inverse-dynamics reads and mass-matrix references with MuJoCo")
    args = parser.parse_args()
    audit(args.output, args.regenerate)
