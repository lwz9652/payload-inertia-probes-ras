"""Recompute the major-revision summaries from the case-level archive."""

import json
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]
RESULTS = SOURCE / "evidence" / "review"
ROBOTS = ("Panda", "UR10e")
NOISE = (.001, .01)
DESIGNS = ("3x3", "3xfull", "4x2", "4x3", "4xfull")


def load(name, count):
    rows = json.loads((RESULTS / name).read_text(encoding="utf-8"))
    assert len(rows) == count, name
    return rows


def by(rows, **conditions):
    return [row for row in rows if all(row[key] == value for key, value in conditions.items())]


def summary(scores, multiplier=100):
    return multiplier * np.quantile(scores, [.5, .9, 1.])


def check_matched():
    rows = load("matched_cases.json", 240)
    print("Matched design medians, R_M in percent")
    for design in DESIGNS:
        entries = []
        for robot in ROBOTS:
            for sigma in NOISE:
                cases = by(rows, robot=robot, design=design, sigma=sigma)
                assert sorted(row["trial"] for row in cases) == list(range(12))
                for case in cases:
                    assert case["reads"] == 2 * (case["loads"] + 1) * case["directions"]
                    assert len(case["exact_operator_margins"]) == 32
                entries.append(summary([row["kernel"]["R_M"] for row in cases])[0])
        if design == "4x3":
            np.testing.assert_allclose(entries, [.0433, .3881, .0319, .3000], atol=.00006)
        print(design, *(f"{value:.3f}" for value in entries))
    for robot, target in zip(ROBOTS, (.260, .141), strict=True):
        cases = by(rows, robot=robot, design="3xfull", sigma=.001)
        error = summary([row["independent_calibration_matrices_R_M"] for row in cases])[0]
        np.testing.assert_allclose(error, target, atol=.0005)
        stages = [summary([row[key]["R_M"] for row in cases])[0]
                  for key in ("local", "spectral_local", "kernel", "spectral")]
        print(robot, "sweep stages", *(f"{value:.3f}" for value in (error, *stages)))
    for robot, max_action in zip(ROBOTS, (.526, .488), strict=True):
        cases = by(rows, robot=robot, design="4x3", sigma=.01)
        np.testing.assert_allclose(summary([row["kernel"]["R_A"] for row in cases])[-1],
                                   max_action, atol=.0005)


def check_payloads():
    rows = load("random_payload_cases.json", 960)
    pooled = load("random_chain_cases.json", 96)
    fourth = load("fourth_load_cases.json", 48)
    for robot, failures, old_count, old_tail, new_tail in (
        ("Panda", 296, 7, 24.40, .518), ("UR10e", 278, 6, 25.57, .386),
    ):
        broad = by(rows, robot=robot, design="nominal_rank", sigma=.001, spread_scale=1.)
        centered = by(rows, robot=robot, design="nominal_rank", sigma=.001, spread_scale=0.)
        assert len(broad) == len(centered) == 40
        assert all(margin > 0 for row in broad for margin in row["exact_operator_margins"])
        assert all(margin == 0 for row in centered for margin in row["exact_operator_margins"])
        errors = [error for row in broad for error in row["heldout_payload_errors"]]
        assert len(errors) == 320
        assert sum(error is None or error > .1 for error in errors) == failures
        print(robot, "local large errors or rejections", f"{failures}/{len(errors)}")
        low_noise = by(pooled, robot=robot, design="nominal_rank", sigma=.001)
        assert len(low_noise) == 12
        print(robot, "random sets after chain fit", summary([row["torque"]["R_M"] for row in low_noise]))
        for method, count, tail in (("unselected", old_count, old_tail), ("chosen", 12, new_tail)):
            cases = by(fourth, robot=robot, design=method)
            assert len(cases) == 12
            for row in cases:
                expected = int(np.argmax(row["candidate_scores"])) if method == "chosen" else 0
                assert row["index"] == expected
            errors = [row["fit"]["R_M"] for row in cases]
            assert sum(e < .01 for e in errors) == count
            np.testing.assert_allclose(summary(errors)[1], tail, atol=.005)
            print(robot, method, "below 1 percent", f"{count}/12", "median/p90/max", summary(errors))


def check_geometry():
    starts = load("initialization_cases.json", 96)
    terminal = load("terminal_cases.json", 96)
    charts = load("chart_coverage_cases.json", 96)
    for robot, early, late, gap, chart_counts in (
        ("Panda", 87.09, .402, 65.8, (4, 10, 4)),
        ("UR10e", 80.52, .276, 84.7, (5, 6, 5)),
    ):
        group = by(starts, robot=robot, angular_bound_deg=45., sigma=.01)
        assert len(group) == 12
        np.testing.assert_allclose(summary([row["result"]["initial"]["R_M"] for row in group])[0],
                                   early, atol=.005)
        np.testing.assert_allclose(summary([row["result"]["R_M"] for row in group])[0],
                                   late, atol=.0005)
        steps = by(terminal, robot=robot, step_rad=.5)
        assert len(steps) == 12
        np.testing.assert_allclose(summary([row["clean_orientation_gap_Nm"] for row in steps], 1000)[0],
                                   gap, atol=.05)
        half_turn = by(terminal, robot=robot, step_rad=float(np.pi))
        assert len(half_turn) == 12
        assert max(row["clean_orientation_gap_Nm"] for row in half_turn) < 1e-12
        full = by(charts, robot=robot, design="3loads_full", sigma=.001)
        sparse = by(charts, robot=robot, design="4loads_2dirs", sigma=.01)
        assert (sum(row["kernel"] for row in full), sum(row["spectral"] for row in full),
                sum(row["kernel"] for row in sparse)) == chart_counts
        print(robot, "45 degree start", f"{early:.2f} -> {late:.3f} percent",
              "chart coverage counts", chart_counts)


if __name__ == "__main__":
    check_matched()
    check_payloads()
    check_geometry()
