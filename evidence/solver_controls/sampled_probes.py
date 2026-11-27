"""Reconstruct directional contrasts from noisy, time-stamped motion windows."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import mujoco
import numpy as np

from controls_common import fit_case
from panda_engine import Panda, add_payload, source_xml
from study_common import recreate, write


HALF_WINDOW = .04
ENCODER_HZ = 500
TIMES = np.arange(-20, 21) / ENCODER_HZ
TORQUE_NOISE = .001
REPEATS = 4

# Condition, encoder SD in rad, maximum clock error in s, repeats,
# relative command along the third direction, apply the basis correction.
CONDITIONS = (
    ("exact",       0.,      0.,    1, 1.,   True),
    ("uncorrected", 2e-6,    0.,    1, 1.,   False),
    ("quiet",       2e-6,    0.,    1, 1.,   True),
    ("noisy",       1e-5,    0.,    1, 1.,   True),
    ("noisy_four",  1e-5,    0.,    4, 1.,   True),
    ("clock",       2e-6,    .001,  1, 1.,   True),
    ("clock_four",  2e-6,    .001,  4, 1.,   True),
    ("weak",        2e-6,    0.,    1, .10,  True),
    ("weak_four",   2e-6,    0.,    4, .10,  True),
    ("very_weak",   2e-6,    0.,    1, .03,  True),
)


def tracking_samples(q, v, acc, jerk, shifts):
    t = (TIMES[None, None, None] + shifts[..., None])[None, ..., None]
    return (q[:, None, None, None, None] +
            v[:, None, None, None, None] * t +
            acc[None, ..., None, :] * (t**2 / 2) +
            jerk[None, ..., None, :] * (t**3 / 6))


def center_torques(arms, q, v, accelerations):
    result = np.empty((len(q), *accelerations.shape))
    for i, load, column, sign in np.ndindex(result.shape[:-1]):
        result[i, load, column, sign] = arms[load].inverse(
            q[i], v[i], accelerations[load, column, sign])
    return result


def contrast(torques):
    return .5 * (torques[..., 0,:] - torques[...,1,:]).swapaxes(-1,-2)


def rms(value):
    return float(np.sqrt(np.mean(value**2)))


def case_job(trial, torque_sigma=TORQUE_NOISE):
    case = recreate(0, trial)
    q, q_test = case["q"], case["held_q"]
    v = .18 * np.cos(q)
    loads = case["catalog"]
    a = case["directions"]["3loads_3dirs"]
    n, k = a.shape
    unseen = np.linalg.qr(a, mode="complete")[0][:,k:]
    xml = source_xml()
    arms = [Panda(xml)] + [Panda(add_payload(xml, I)) for I in loads]
    calibration_mass = np.stack([arm.mass_grid(q) for arm in arms], axis=1)
    ideal = (calibration_mass[:,1:] - calibration_mass[:,:1]) @ a
    empty_test = arms[0].mass_grid(q_test)
    truth = np.stack([Panda(add_payload(xml,I)).mass_grid(q_test) - empty_test
                      for I in case["tests"]], axis=1)
    count = len(arms)
    rng = np.random.default_rng(2026113250 + trial)
    raw_shape = (REPEATS,len(q),count,k,2,n)
    torque_noise = torque_sigma * rng.normal(size=raw_shape)
    encoder_noise = rng.normal(size=(*raw_shape[:-1],len(TIMES),n))
    mix = rng.normal(size=(count,k,k))
    mix *= .12 / np.linalg.norm(mix, axis=(1,2), keepdims=True)
    jerk = rng.normal(size=(count,k,2,n))
    jerk *= 6. / np.linalg.norm(jerk,axis=-1,keepdims=True)
    clock = rng.uniform(-1.,1.,size=(count,k,2))
    x = TIMES / HALF_WINDOW
    weights = np.linalg.pinv(np.column_stack((np.ones(len(x)),x,x*x,x*x*x)))[:3]
    weights *= np.array([1.,1./HALF_WINDOW,2./HALF_WINDOW**2])[:, None]
    signs = np.array([1.,-1.])[None,None,:,None]
    rows = []
    for name, sigma_q, clock_seconds, repeats, amplitude, correction in CONDITIONS:
        T = (np.eye(k) + mix) @ np.diag([1.,1.,amplitude])
        acc = (a @ T).swapaxes(-1,-2)[:,:,None,:] * signs
        clean_torque = center_torques(arms, q, v, acc)
        true_b = contrast(clean_torque)
        true_aligned = true_b @ np.linalg.inv(T)
        np.testing.assert_allclose(true_aligned[:,1:] - true_aligned[:,:1], ideal,
                                   atol=5e-13, rtol=1e-11)

        samples = tracking_samples(q, v, acc, jerk, clock_seconds*clock)
        samples += sigma_q * encoder_noise[:repeats].mean(axis=0)
        estimated = np.einsum("ft,qlkstj->qlksfj", weights, samples, optimize=True)
        achieved = .5 * (estimated[:,:,:,0,2,:] - estimated[:,:,:,1,2,:]).swapaxes(-1,-2)
        estimated_T = a.T @ achieved
        error = estimated_T - T
        smallest = np.linalg.svd(estimated_T, compute_uv=False)[..., -1]
        if sigma_q == 0. and clock_seconds == 0.:
            np.testing.assert_allclose(estimated_T,np.broadcast_to(T,estimated_T.shape),atol=2e-11)
        actual_b = contrast(clean_torque + torque_noise[:repeats].mean(axis=0))
        values = actual_b @ np.linalg.inv(estimated_T) if correction else actual_b
        measured = values[:,1:] - values[:,:1]
        assigned_q = estimated[...,0,:].mean(axis=(1,2,3))
        fit = fit_case(assigned_q,a,loads,measured,q_test,case["tests"],truth,unseen)
        delta_q = estimated[...,0,:] - q[:,None,None,None]
        delta_v = estimated[...,1,:] - v[:,None,None,None]
        rows.append(dict(
            trial=trial, name=name, encoder_sigma_rad=sigma_q, max_clock_s=clock_seconds,
            repeats=repeats, weak_amplitude=amplitude, basis_correction=correction,
            q_RMS_rad=rms(delta_q), v_RMS_rad_s=rms(delta_v), a_RMS_rad_s2=rms(estimated[...,2,:]-acc),
            minimum_sigma_T=float(smallest.min()), basis_error_RMS=rms(error),
            probe_orthogonal_RMS_rad_s2=rms(np.linalg.norm(achieved-a@estimated_T,axis=-2)),
            torque_contrast_RMS_Nm=rms(measured-ideal), fit=fit,
        ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",type=int,default=12)
    parser.add_argument("--workers",type=int,default=5)
    parser.add_argument("--torque-noise",type=float,default=TORQUE_NOISE)
    args = parser.parse_args()
    if args.torque_noise <= 0:
        parser.error("torque noise must be positive")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(case_job,i,args.torque_noise) for i in range(args.trials)]
        rows = []
        for job in as_completed(jobs):
            block = job.result()
            rows.extend(block)
            print("MuJoCo Panda",block[0]["trial"],len(rows),"sampling cases complete",flush=True)
    rows.sort(key=lambda r:(r["trial"],r["name"]))
    write(args.output,dict(rows=rows, mj_version=mujoco.__version__, encoder_Hz=ENCODER_HZ,
                           samples=len(TIMES), raw_torque_sigma=args.torque_noise))


if __name__ == "__main__":
    main()
