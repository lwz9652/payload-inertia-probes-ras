"""Damped least squares on the revolute-axis manifold with moving charts."""

import numpy as np

from coordinates import axis_planes, directional_residual, finite_difference, move_geometry


def bounded_step(step):
    lengths = np.linalg.norm(step.reshape(-1, 4)[:, :2], axis=1)
    return step / max(1., np.max(lengths) / .75)


def fixed_chart_geometry(start, plane, coordinates):
    axes, points = start
    local = coordinates.reshape(-1, 4)
    omega = axes + (plane @ local[:, :2, None])[..., 0]
    omega /= np.linalg.norm(omega, axis=1, keepdims=True)
    point = points + (plane @ local[:, 2:, None])[..., 0]
    point -= (point * omega).sum(1, keepdims=True) * omega
    return omega, point


def refine_recentered(q, inertias, a, observed, start, max_evaluations=1600, *, fixed_chart=False):
    residual = directional_residual(q, inertias, a, observed)
    geometry = start
    current = residual(geometry)
    evaluations = 1
    loss = current @ current / 2
    init_loss = loss
    damping = 1e-3
    expand = 2.
    steps = 0
    crossed = False
    reason = "evaluation_budget"
    count = 4 * len(start[0])
    identity = np.eye(count)
    chart_coordinates = np.zeros(count)
    if fixed_chart:
        origin_plane = axis_planes(start[0])

    while evaluations + count + 1 <= max_evaluations:
        if fixed_chart:
            samples = [residual(fixed_chart_geometry(start, origin_plane, chart_coordinates+1e-6*d))
                       for d in identity]
            jac = (np.asarray(samples) - current).T / 1e-6
        else:
            jac, planes = finite_difference(residual, geometry, current)
        evaluations += count
        norms = np.linalg.norm(jac, axis=0).clip(1e-10)
        scaled = jac / norms
        gradient = scaled.T @ current
        if np.max(np.abs(gradient)) < 1e-8:
            reason = "gradient"
            break

        normal = scaled.T @ scaled
        accepted = False
        while evaluations < max_evaluations and damping < 1e10:
            newton = np.linalg.solve(normal + damping * identity, -gradient) / norms
            step = bounded_step(newton)
            if fixed_chart:
                trial_geometry = fixed_chart_geometry(start, origin_plane, chart_coordinates + step)
            else:
                trial_geometry = move_geometry(geometry, planes, step)
            trial = residual(trial_geometry)
            evaluations += 1
            trial_loss = trial @ trial / 2
            predicted_change = jac @ step
            predicted = -(current @ predicted_change + predicted_change @ predicted_change / 2)
            decrease = loss - trial_loss
            if predicted > 0 and decrease > 0:
                ratio = decrease / predicted
                damping *= max(1/3, 1 - (2 * ratio - 1)**3)
                damping = max(damping, 1e-12)
                expand = 2.
                geometry, current, loss = trial_geometry, trial, trial_loss
                if fixed_chart:
                    chart_coordinates += step
                steps += 1
                crossed |= bool(np.any(np.sum(geometry[0] * start[0], axis=1) < 0))
                accepted = True
                break
            damping *= expand
            expand *= 2.

        if not accepted:
            if evaluations < max_evaluations:
                reason = "damping"
            break
        if decrease <= 1e-10 * max(loss, 1e-8):
            reason = "relative_objective"
            break

    return dict(
        geometry=geometry, train_rmse=float(np.sqrt(2 * loss / len(current))),
        start_rmse=float(np.sqrt(2 * init_loss / len(current))), nfev=evaluations,
        accepted_steps=steps, crossed_initial_hemisphere=crossed, termination=reason,
    )


def best_recentered(q, inertias, a, observed, starts, *, fixed_chart=False):
    fits = [refine_recentered(q, inertias, a, observed, start, fixed_chart=fixed_chart)
            for start in starts]
    index = int(np.argmin([fit["train_rmse"] for fit in fits]))
    best = fits[index]
    info = [dict((k, v) for k, v in fit.items() if k != "geometry") for fit in fits]
    return best["geometry"], index, info
