"""Stress torque contrasts with friction, velocity mismatch, and causal filtering."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from controls_common import fit_case
from alignment_study import contrasts
from panda_engine import Panda, add_payload, source_xml
from study_common import recreate, write


VISCOSITY_TRANSITION = .005  # rad/s, smooth Coulomb approximation
FRICTION_NM = .20
VELOCITY_MISMATCH = .002  # rad/s RMS per read
SIGMA_NM = .01  # after torque filtering
TIME_CONSTANT = .008
SAMPLE_PERIOD = .001
HISTORY_STEPS = 48


def readings(arms, q, velocities, a, time=0., *, friction=True):
    output = np.empty((len(q), len(arms), a.shape[1], 2, q.shape[1]))
    signed_a = a.T[:, None, :] * np.array([1., -1.])[None, :, None]
    instant_v = velocities + signed_a * time
    displacement = velocities * time + signed_a * (time**2 / 2)
    for pose, load, probe, sign in np.ndindex(output.shape[:-1]):
        output[pose, load, probe, sign] = arms[load].inverse(
            q[pose] + displacement[load, probe, sign], instant_v[load, probe, sign],
            signed_a[probe, sign])
    if friction:
        output += FRICTION_NM * np.tanh(instant_v / VISCOSITY_TRANSITION)
    return output


def filtered_readings(arms, q, velocities, a):
    decay = np.exp(-SAMPLE_PERIOD / TIME_CONSTANT)
    output = readings(arms, q, velocities, a, -HISTORY_STEPS * SAMPLE_PERIOD)
    for sample in range(1 - HISTORY_STEPS, 1):
        output = decay * output + (1 - decay) * readings(
            arms, q, velocities, a, sample * SAMPLE_PERIOD)
    return output


def case_job(trial):
    case = recreate(0, trial)
    q, q_test = case["q"], case["held_q"]
    loads = case["catalog"]
    a = case["directions"]["3loads_3dirs"]
    n, k = a.shape
    unseen = np.linalg.qr(a, mode="complete")[0][:, k:]
    xml = source_xml()
    arms = [Panda(xml)] + [Panda(add_payload(xml, load)) for load in loads]
    mass = np.stack([arm.mass_grid(q) for arm in arms], axis=1)
    ideal = (mass[:, 1:] - mass[:, :1]) @ a
    empty = arms[0].mass_grid(q_test)
    truth = np.stack([Panda(add_payload(xml, load)).mass_grid(q_test) - empty
                      for load in case["tests"]], axis=1)
    rng = np.random.default_rng(2026112528 + trial)
    shift = rng.normal(size=(len(arms), k, 2, n))
    shift *= VELOCITY_MISMATCH / np.sqrt(np.mean(shift**2, axis=-1, keepdims=True))
    shift[0] = 0.
    noise = rng.normal(scale=SIGMA_NM, size=(len(q), len(arms), k, 2, n))
    rows = []
    for speed, paired_speed in (("same_direction", [.18, .18]), ("near_reversal", [.005, -.005])):
        exact = np.broadcast_to(np.asarray(paired_speed)[None, None, :, None], shift.shape)
        matched = readings(arms, q, exact, a)
        np.testing.assert_allclose(contrasts(matched), ideal, atol=5e-13, rtol=1e-11)
        misaligned = exact + shift
        biased = readings(arms, q, misaligned, a)
        friction_only = contrasts(biased - readings(arms, q, misaligned, a, friction=False))
        filtered_exact = filtered_readings(arms, q, exact, a)
        smoothed = filtered_readings(arms, q, misaligned, a)
        for acquisition, raw in (("matched", matched), ("velocity_error", biased),
                                 ("filter", filtered_exact),
                                 ("velocity_error_with_filter", smoothed)):
            bias = contrasts(raw) - ideal
            fit = fit_case(q, a, loads, contrasts(raw + noise), q_test, case["tests"], truth, unseen)
            rows.append(dict(trial=trial, speed=speed, acquisition=acquisition,
                             clean_contrast_bias_RMS_Nm=float(np.sqrt(np.mean(bias**2))),
                             instantaneous_friction_bias_RMS_Nm=float(np.sqrt(np.mean(friction_only**2)))
                             if acquisition == "velocity_error" else None, fit=fit))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job, trial) for trial in range(args.trials)]
        rows = []
        for job in as_completed(jobs):
            block = job.result()
            rows.extend(block)
            print("MuJoCo Panda", block[0]["trial"], len(rows),
                  "friction and filtering conditions complete", flush=True)
    rows.sort(key=lambda row: (row["trial"], row["speed"], row["acquisition"]))
    write(args.output, dict(mujoco=mujoco.__version__, friction_Nm=FRICTION_NM,
                            transition_rad_s=VISCOSITY_TRANSITION, noise_Nm=SIGMA_NM,
                            mismatch_rad_s=VELOCITY_MISMATCH, filter_s=TIME_CONSTANT,
                            interval_s=SAMPLE_PERIOD, history_steps=HISTORY_STEPS, rows=rows))


if __name__ == "__main__":
    main()
