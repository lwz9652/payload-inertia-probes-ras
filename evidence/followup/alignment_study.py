"""Measure state and acceleration alignment sensitivity with raw inverse-dynamics reads."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from recentered_fit import best_recentered
from fit_chain import initialize_from_data
from panda_engine import Panda, MODEL_SHA256, add_payload, source_xml
from study_common import errors, matrices, recreate, serial_body, write


LEVELS = {
    "cross_load_q": (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
    "sign_pair_q": (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2),
    "sign_pair_v": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    "off_space": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
}
# Each level is the last point below the 1% target in all single-factor trials.
JOINT_LEVELS = {"cross_load_q": .003, "sign_pair_q": .0003, "sign_pair_v": .01, "off_space": .003}
JOINT_FRACTIONS = (.25, .5, 1.)
SIGMA = .001


def normalize(vec, size):
    return size * vec / np.linalg.norm(vec, axis=-1, keepdims=True)


def realized_pulses(q, velocity, directions, direction_noise, mode, size):
    l, k, n = direction_noise.shape
    positions = np.broadcast_to(q[:, None, None, None, :], (len(q), l, k, 2, n)).copy()
    speeds = np.broadcast_to(velocity[:, None, None, None, :], positions.shape).copy()
    a = np.broadcast_to(directions.T, (l,k,n)).copy()
    change = size * normalize(direction_noise, np.sqrt(n))
    if mode == "cross_load_q":
        change[0] = 0.
        positions += change[None, :, :, None]
    elif mode == "sign_pair_q":
        positions += change[None, :, :, None] * np.array([.5,-.5])[None,None,None,:,None]
    elif mode == "sign_pair_v":
        speeds += change[None, :, :, None] * np.array([.5,-.5])[None,None,None,:,None]
    elif mode == "off_space":
        projected = direction_noise - direction_noise @ directions @ directions.T
        a += size * normalize(projected, 1.)
    return positions, speeds, a


def acquire(arms, q, velocity, accel):
    reads = np.empty_like(q)
    for pose, arm, probe, sign in np.ndindex(q.shape[:-1]):
        a = accel[arm, probe] * (1. if sign == 0 else -1.)
        reads[pose, arm, probe, sign] = arms[arm].inverse(q[pose,arm,probe,sign],
                                                       velocity[pose,arm,probe,sign], a)
    return reads


def contrasts(torques, aligned_basis=None):
    half = ((torques[..., 0, :] - torques[..., 1, :]) * .5).swapaxes(-1, -2)
    if aligned_basis is not None:
        half = half @ np.linalg.inv(aligned_basis)
    return half[:,1:] - half[:,:1]


def joint_pulses(q, velocity, a, noise, fraction, base):
    cross_q, _, _ = realized_pulses(q, velocity, a, noise["cross_load_q"],
                                   "cross_load_q", fraction * JOINT_LEVELS["cross_load_q"])
    within_q, _, _ = realized_pulses(q, velocity, a, noise["sign_pair_q"],
                                    "sign_pair_q", fraction * JOINT_LEVELS["sign_pair_q"])
    _, speeds, _ = realized_pulses(q, velocity, a, noise["sign_pair_v"],
                                  "sign_pair_v", fraction * JOINT_LEVELS["sign_pair_v"])
    _, _, probes = realized_pulses(q, velocity, a, noise["off_space"],
                                   "off_space", fraction * JOINT_LEVELS["off_space"])
    return cross_q + (within_q - base), speeds, probes


def case_job(trial, joint_only=False, joint_fractions=JOINT_FRACTIONS, sigma=SIGMA,
             passing_levels_only=False):
    case = recreate(0, trial)
    q, heldout = case["q"], case["held_q"]
    loads = case["catalog"]
    a = case["directions"]["3loads_3dirs"]
    n, k = a.shape
    nullspace = np.linalg.qr(a, mode="complete")[0][:,k:]
    np.testing.assert_allclose(a.T @ nullspace, 0., atol=1e-13)
    velocity = .18 * np.cos(q)
    xml = source_xml()
    arms = [Panda(xml)] + [Panda(add_payload(xml, payload)) for payload in loads]
    mass = np.stack([arm.mass_grid(q) for arm in arms], axis=1)
    clean = (mass[:,1:] - mass[:,:1]) @ a
    reference_mass = arms[0].mass_grid(heldout)
    truth = np.stack([Panda(add_payload(xml, payload)).mass_grid(heldout) - reference_mass
                      for payload in case["tests"]], axis=1)
    rng = np.random.default_rng(2026112300 + trial)
    mode_vectors = {mode: rng.normal(size=(len(arms), k, n)) for mode in LEVELS}
    shape = (len(q),len(arms),k,2,n)
    read_noise = rng.normal(scale=sigma, size=shape)
    unshifted = realized_pulses(q, velocity, a, mode_vectors["off_space"], "aligned", 0.)
    readings = acquire(arms, *unshifted)
    np.testing.assert_allclose(contrasts(readings), clean, atol=5e-13, rtol=1e-11)
    transversal = rng.normal(size=(len(arms), k, k))
    transversal *= .12 / np.linalg.norm(transversal, axis=(1,2), keepdims=True)
    mixing = np.eye(k) + transversal
    mixed_accels = (a @ mixing).swapaxes(-2, -1)
    raw_mix = acquire(arms, unshifted[0], unshifted[1], mixed_accels)
    mix_error = float(np.max(abs(contrasts(raw_mix, mixing) - clean)))
    np.testing.assert_allclose(contrasts(raw_mix, mixing), clean, atol=5e-13, rtol=1e-11)

    outputs = []
    levels = [("aligned", 0.)] + [(mode, size) for mode, sizes in LEVELS.items() for size in sizes]
    if passing_levels_only:
        levels = [("aligned", 0.)] + list(JOINT_LEVELS.items())
    if joint_only:
        levels = [("joint", fraction) for fraction in joint_fractions]
    for mode, size in levels:
        if mode == "aligned":
            torques = readings
        elif mode == "joint":
            realized = joint_pulses(q, velocity, a, mode_vectors, size, unshifted[0])
            torques = acquire(arms, *realized)
        else:
            realized = realized_pulses(q, velocity, a, mode_vectors[mode], mode, size)
            torques = acquire(arms, *realized)
        if mode == "off_space":
            difference = realized[2].swapaxes(-2,-1) - a
            np.testing.assert_allclose(a.T @ difference, 0., atol=1e-13)
            leakage = mass @ difference
            np.testing.assert_allclose(contrasts(torques) - clean, leakage[:,1:] - leakage[:,:1],
                                       atol=5e-13, rtol=1e-10)
        observed = contrasts(torques + read_noise)
        starts, usable = initialize_from_data(q, loads, a, observed)
        scores, rows, winner = None, [], None
        if starts:
            geometry, winner, rows = best_recentered(q, loads, a, observed, starts)
            scores = errors(matrices(serial_body(heldout, *geometry), case["tests"]), truth, nullspace)
        outputs.append(dict(
            trial=trial, mode=mode, level=size, sigma=sigma, eligible_poses=usable,
            contrast_bias_RMS_Nm=float(np.sqrt(np.mean((contrasts(torques)-clean)**2))),
            corrected_within_subspace_max_abs_Nm=mix_error, winner=winner, fits=rows,
            heldout=scores,
        ))
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--joint", action="store_true", help="Combine the four individually screened levels")
    parser.add_argument("--start-trial", type=int, default=0)
    parser.add_argument("--joint-fraction", type=float, nargs="+", default=JOINT_FRACTIONS)
    parser.add_argument("--torque-noise", type=float, default=SIGMA)
    parser.add_argument("--passing-levels-only", action="store_true",
                        help="Use the aligned condition and the four archived passing levels")
    args = parser.parse_args()
    if args.torque_noise <= 0:
        parser.error("torque noise must be positive")
    if args.joint and args.passing_levels_only:
        parser.error("choose joint levels or passing single-factor levels")
    trials = range(args.start_trial, args.start_trial + args.trials)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job, trial, args.joint, args.joint_fraction,
                            args.torque_noise, args.passing_levels_only) for trial in trials]
        rows = []
        for task in as_completed(jobs):
            block = task.result()
            rows.extend(block)
            print("MuJoCo Panda", block[0]["trial"], len(rows), "conditions complete", flush=True)
    rows.sort(key=lambda row:(row["trial"],row["mode"],row["level"]))
    write(args.output, dict(mujoco=mujoco.__version__, xml_sha256=MODEL_SHA256,
                            numpy=np.__version__, rows=rows, levels=LEVELS,
                            combined_levels=JOINT_LEVELS if args.joint else None))


if __name__ == "__main__":
    main()
