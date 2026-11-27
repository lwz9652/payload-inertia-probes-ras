"""Fit revolute body axes from sparse load responses and transfer across q."""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from directional_recovery import recover_initial, shared_reference_whitener
from robot_geometry import move_attachment, reverse_axes, serial_body


def axes_from_home(jacobian):
    axes = jacobian[:3].T
    return axes, np.cross(axes, jacobian[3:].T)


def initialize_from_data(q, inertias, directions, responses):
    factors, margins, indices = [], [], []
    for index, observation in enumerate(responses):
        try:
            estimate, margin = recover_initial(inertias, directions, observation)
        except ValueError:
            continue
        size = np.linalg.norm(estimate[:3, -1])
        if 0.1 < size < 10.0:
            factors.append(estimate)
            margins.append(margin["rank6_retained_singular_value"] * margin["operator_margin"])
            indices.append(index)
    if not factors:
        return [], 0
    estimate = np.asarray(factors)
    weights = np.asarray(margins)
    weights = (weights / weights.max()).clip(1e-4)
    last = estimate[:, :3, -1].copy()
    last /= np.linalg.norm(last, axis=1, keepdims=True)
    direction = np.linalg.svd(last * weights[:, None] ** 0.5, full_matrices=False)[2][0]
    signs = np.where(last @ direction >= 0, 1., -1.)
    estimate = estimate * signs[:, None, None]
    aligned = []
    for orientation in (1, -1):
        try:
            aligned.append(reverse_axes(q[indices], orientation * estimate, weights))
        except ValueError:
            continue
    return aligned, len(indices)


def refine_axes(q, inertias, directions, observed, starting_axes, starting_points,
                max_evaluations=100):
    # Per joint, perturb two tangent directions of the unit axis and two of
    # the line point. Reproject the point perpendicular to its updated axis.
    normal = np.eye(3)[np.argmin(np.abs(starting_axes), axis=1)]
    first = np.cross(starting_axes, normal)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    plane = np.stack((first, np.cross(starting_axes, first)), axis=-1)
    whiten = shared_reference_whitener(len(inertias))

    def geometry(delta):
        local = delta.reshape(-1, 4)
        axis = starting_axes + np.einsum("nij,nj->ni", plane, local[:, :2])
        axis /= np.linalg.norm(axis, axis=1, keepdims=True)
        point = starting_points + np.einsum("nij,nj->ni", plane, local[:, 2:])
        point -= (point * axis).sum(1, keepdims=True) * axis
        return axis, point

    def residual(delta):
        j = serial_body(q, *geometry(delta))
        predicted = j[:, None].swapaxes(-2, -1) @ inertias @ j[:, None] @ directions
        difference = np.einsum("ml,qlik->qmik", whiten, predicted - observed)
        return difference.ravel()

    initial = np.zeros(4 * q.shape[1])
    solution = least_squares(
        residual, initial, method="lm", jac="2-point", max_nfev=max_evaluations,
        x_scale="jac", ftol=1e-11, xtol=1e-11, gtol=1e-11,
    )
    return dict(
        geometry=geometry(solution.x), train_rmse=float(np.sqrt(2 * solution.cost / observed.size)),
        start_rmse=float(np.mean(residual(initial) ** 2) ** 0.5), evaluations=int(solution.nfev),
        converged=bool(solution.success),
    )


def calibrate_chain(q, inertias, directions, observed, reference=None):
    starts, usable = initialize_from_data(q, inertias, directions, observed)
    methods = {}
    candidates = [refine_axes(q, inertias, directions, observed, *start, max_evaluations=600)
                  for start in starts]
    if candidates:
        methods["from_torques"] = min(candidates, key=lambda fit: fit["train_rmse"])
        methods["from_torques"]["pose_initializations"] = usable
        methods["from_torques"]["orientation_objectives"] = [x["train_rmse"] for x in candidates]
    if reference is not None:
        methods["from_URDF"] = refine_axes(q, inertias, directions, observed, *reference,
                                            max_evaluations=600)
    return methods


def fit_attachment(nominal_jacobians, inertias, directions, observed):
    """Strong six-parameter baseline using the nominal arm and a free attachment."""
    whiten = shared_reference_whitener(len(inertias))

    def response(parameters):
        rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        j = move_attachment(nominal_jacobians, rotation, parameters[3:])
        return j[:, None].swapaxes(-2, -1) @ inertias @ j[:, None] @ directions

    def residual(parameters):
        error = response(parameters) - observed
        return np.einsum("ml,qlik->qmik", whiten, error).ravel()

    solution = least_squares(residual, np.zeros(6), method="lm",
                             max_nfev=600, x_scale="jac")
    return dict(
        parameters=solution.x.tolist(), evaluations=int(solution.nfev),
        train_rmse=float(np.sqrt(2 * solution.cost / observed.size)),
        converged=bool(solution.success),
    )
