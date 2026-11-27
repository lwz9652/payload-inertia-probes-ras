"""Exact regular-point example with ambiguous responses for three loads, two directions."""

import argparse
import json
from pathlib import Path

import sympy as sp

from exact_certificate import home_jacobian, physical_inertias, rational_matrix


def audit():
    jacobian = home_jacobian()
    inertias = physical_inertias()
    state = jacobian[:, :2]
    wrench = sp.Matrix.hstack(*(inertia * state for inertia in inertias[:3]))
    assert rational_matrix(wrench).rank() == 6

    def gram_derivative(state_delta):
        blocks = [state_delta.T * inertia * state + state.T * inertia * state_delta
                  for inertia in inertias[:3]]
        return sp.Matrix([block[i, j] for block in blocks for i, j in ((0, 0), (0, 1), (1, 1))])

    columns = []
    for idx in range(12):
        perturbation = sp.zeros(6, 2)
        perturbation[idx] = 1
        columns.append(gram_derivative(perturbation))
    derivative = rational_matrix(sp.Matrix.hstack(*columns))
    assert derivative.rank() == 9
    null_basis = derivative.nullspace().to_Matrix()
    result = []
    for row in null_basis.tolist():
        delta = sp.Matrix(6, 2, row)
        delta_wrench = sp.Matrix.hstack(*(inertia * delta for inertia in inertias[:3]))
        delta_jac = (-jacobian.T * delta_wrench * wrench.inv()).T
        assert delta_jac[:, :2] == delta
        heldout = delta_jac.T * inertias[3] * jacobian + jacobian.T * inertias[3] * delta_jac
        result.append(dict(heldout_change_is_nonzero=heldout != sp.zeros(7)))
    assert any(r["heldout_change_is_nonzero"] for r in result)
    return dict(
        calibration_loads=3, directions=2, wrench_matrix_rank=6,
        gram_map_derivative_rank=9, local_ambiguity_dimension=3,
        fourth_load_response_changes=True, tangent_directions=result,
        scope="Fixed-configuration, general full-row-rank tool Jacobian. "
              "Variations are not restricted to a serial revolute chain.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    evidence = audit()
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(evidence, indent=2) + "\n")
    print(evidence)
