"""Recover a shared payload Jacobian from partial acceleration probes."""

import numpy as np


def shared_reference_whitener(load_count):
    # At unit probe amplitude, a difference of two centered contrasts has
    # covariance (sigma_torque**2 / 2) * (I + 11.T) across loads.
    return np.eye(load_count) - (
        (1 - 1 / np.sqrt(load_count + 1)) / load_count
    ) * np.ones((load_count, load_count))


def nullspace_operator(inertias, coefficients):
    """Matrix of U -> sum_l I_l U Z_l, with U and the result in row-major order."""
    loads, probes, nullity = coefficients.shape
    degree = inertias.shape[1]
    assert inertias.shape == (loads, degree, degree)
    return np.einsum("lij,lkh->ihjk", inertias, coefficients).reshape(
        degree * nullity, degree * probes,
    )


def metric_whiten(inertias):
    lower = np.linalg.cholesky(np.mean(inertias, axis=0))
    change = np.linalg.solve(lower, np.eye(lower.shape[0]))
    return change @ inertias @ change.T, lower


def recover_initial(inertias, directions, observed):
    """Return J and the two rank margins using only the observed torques."""
    loads, joints, probes = observed.shape
    if directions.shape != (joints, probes):
        raise ValueError("Observation and direction shapes disagree")
    degree = inertias.shape[1]
    if joints < degree or loads * probes <= degree:
        raise ValueError("Too few joints or observations for the linear recovery")
    h, lower = metric_whiten(inertias)
    weight = np.kron(shared_reference_whitener(loads), np.eye(probes))
    stacked = observed.swapaxes(0, 1).reshape(joints, loads * probes) @ weight
    _, sigma, right = np.linalg.svd(stacked, full_matrices=True)
    if sigma[degree - 1] <= 1e-11 * sigma[0]:
        raise ValueError("Stacked torque response has rank below six")
    coefficients = (weight @ right[degree:].T).reshape(loads, probes, -1)
    operator = nullspace_operator(h, coefficients)
    if operator.shape[0] < operator.shape[1] - 1:
        raise ValueError("The nullspace constraints have too few rows")
    _, gap_spectrum, vectors = np.linalg.svd(operator, full_matrices=True)
    # Reject numerical degeneracy. This threshold does not certify accuracy
    # once observation noise perturbs the nullspace.
    tolerance = np.finfo(gap_spectrum.dtype).eps * max(operator.shape) * gap_spectrum[0]
    if gap_spectrum[degree * probes - 2] <= tolerance:
        raise ValueError("The probe state is ambiguous beyond a common scale")
    direction_state = vectors[-1].reshape(degree, probes)
    anchor_blocks = direction_state.T @ h @ direction_state
    observed_blocks = directions.T @ observed
    scale_squared = np.vdot(anchor_blocks, observed_blocks) / np.vdot(anchor_blocks, anchor_blocks)
    if not np.isfinite(scale_squared) or scale_squared <= 0:
        raise ValueError("Probe energy does not identify a positive scale")
    excitation = (h @ (np.sqrt(scale_squared) * direction_state)).swapaxes(0, 1).reshape(
        degree, loads * probes,
    ) @ weight
    spectrum = np.linalg.svd(excitation, compute_uv=False)
    if spectrum[-1] <= 1e-10 * spectrum[0]:
        raise ValueError("Recovered probes do not span the six-dimensional wrench space")
    b = (stacked @ np.linalg.pinv(excitation, rcond=1e-10)).T
    jacobian = np.linalg.solve(lower.T, b)
    diagnostics = dict(
        torque_rank_margin=float(sigma[degree - 1] / sigma[0]),
        operator_margin=float(gap_spectrum[degree * probes - 2] / gap_spectrum[0]),
        operator_last=float(gap_spectrum[-1] / gap_spectrum[0]),
        rank6_retained_singular_value=float(sigma[degree - 1]),
    )
    return jacobian, diagnostics


def contrast_noise(rng, loads, joints, directions, sigma, amplitude=1.0, repetitions=1):
    """Draw raw plus/minus read errors, including one reused unloaded reference."""
    reads = rng.normal(scale=sigma,
                       size=(repetitions, loads + 1, 2, joints, directions))
    measured = (reads[:, :, 0] - reads[:, :, 1]).mean(0) / (2 * amplitude)
    return measured[1:] - measured[:1]
