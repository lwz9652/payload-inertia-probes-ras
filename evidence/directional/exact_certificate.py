"""Exact rank certificates for physically realizable directional designs."""

import argparse
import json
from pathlib import Path
from random import Random

import sympy as sp
from sympy.polys.matrices import DomainMatrix


def rational_matrix(matrix):
    return DomainMatrix.from_Matrix(matrix).convert_to(sp.QQ)


def cross_matrix(x):
    a, b, c = x
    return sp.Matrix([[0, -c, b], [c, 0, -a], [-b, a, 0]])


def physical_inertias():
    rng = Random(1147)
    result = []
    for _ in range(4):
        points = [sp.Matrix([sp.Rational(rng.randrange(-10, 11), 100) for _ in range(3)])
                  for _ in range(5)]
        masses = [sp.Rational(rng.randrange(2, 9), 10) for _ in points]
        h = sum((m * p for p, m in zip(points, masses)), sp.zeros(3, 1))
        rotation = sum((m * (p.dot(p) * sp.eye(3) - p * p.T)
                        for p, m in zip(points, masses)), sp.zeros(3))
        c = cross_matrix(h)
        inertia = rotation.row_join(c).col_join((-c).row_join(sum(masses) * sp.eye(3)))
        assert all(inertia[:k, :k].det() > 0 for k in range(1, 7))
        result.append(inertia)
    return result


def home_jacobian():
    # Each joint is a rational unit revolute screw with zero pitch.
    axes = [sp.eye(3)[:, j % 3] for j in range(7)]
    cm = [(1, 3, 7), (2, -1, 3), (-5, 4, 8), (3, -3, 10),
          (5, 5, -1), (6, 1, 5), (1, 1, 3)]
    points = [sp.Matrix(p) / 20 for p in cm]
    jacobian = sp.Matrix.hstack(*(w.col_join(c.cross(w)) for w, c in zip(axes, points)))
    assert rational_matrix(jacobian).rank() == 6
    return jacobian


def certificate(inertias, jacobian, probes):
    directions = sp.eye(jacobian.shape[1])[:, :probes]
    state = jacobian * directions
    wrenches = sp.Matrix.hstack(*(inertia * state for inertia in inertias))
    torque = jacobian.T * wrenches
    exact_torque = rational_matrix(torque)
    blocks = exact_torque.nullspace().to_Matrix().T
    nullity = blocks.cols
    assert wrenches * blocks == sp.zeros(6, nullity)
    operator = sp.Matrix(6 * nullity, 6 * probes, lambda i, j: sum(
        inertia[i // nullity, j // probes] * blocks[l * probes + j % probes, i % nullity]
        for l, inertia in enumerate(inertias)
    ))
    state_vector = sp.Matrix(list(state))
    assert operator * state_vector == sp.zeros(operator.rows, 1)
    rank = rational_matrix(operator).rank()
    assert rank == 6 * probes - 1 and exact_torque.rank() == 6
    return dict(loads=len(inertias), probes=probes, stacked_rank=6,
                nullspace_constraint_rank=rank, unique_scale=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inertias = physical_inertias()
    jacobian = home_jacobian()
    report = dict(cases=[certificate(inertias[:3], jacobian, 3), certificate(inertias, jacobian, 2)])
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
