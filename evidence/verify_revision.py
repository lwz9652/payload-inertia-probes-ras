"""Check manuscript values against the archived directional experiments."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "directional"
sys.path.insert(0, str(DATA))
from directional_recovery import recover_initial
from verify_results import aggregate


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_direct_probes():
    two = DATA / "mujoco_terminal_check"
    score = read(two / "report.json")
    assert score["fit_folds"] == 4 and score["calibration_configurations"] == 2
    assert score["new_configurations"] == 128 and score["new_payloads"] == 64
    assert score["terminal_only_change_rad"] == 0.5
    assert 2 * score["actual_mj_inverse_calls_per_configuration"] == 48
    worst = 0.
    for case in score["rows"]:
        worst = max(worst, case["full_relative_error"], case["unprobed_relative_error"])
        with np.load(two / f"fold{case['fold']}_probes.npz", allow_pickle=False) as pack:
            q, p = pack["q"], pack["applied_directions"]
            np.testing.assert_array_equal(q[0, :-1], q[1, :-1])
            np.testing.assert_allclose(abs(q[1, -1] - q[0, -1]), 0.5, atol=1e-12)
            np.testing.assert_allclose(p.T @ p, np.eye(3), atol=1e-12)
            torque = pack["torques"]
            assert torque.shape == (2, 4, 3, 2, 7)
            contrast = (torque[..., 0, :] - torque[..., 1, :]) * 0.5
            observation = (contrast[:, 1:] - contrast[:, :1]).swapaxes(-2, -1)
            for row in observation:
                j, _ = recover_initial(pack["inertias"], p, row)
                np.testing.assert_allclose(j.T @ pack["inertias"] @ j @ p, row, atol=1e-9, rtol=1e-9)
        start = case["orientation_objectives"]
        assert max(start) > 0.01 and min(start) < 1e-12
    assert worst < 1.1e-13
    return worst


def check_geometry_archive():
    summary = aggregate(DATA)
    assert summary == read(DATA / "verified_summary.json")
    nominal = read(DATA / "fewer_directions_audit.json")
    for robot in nominal["results"]:
        for n in (1, 2, 3):
            records = [r["relative_RM"] for r in robot["data"] if r["directions"] == n]
            row = robot["summary"][str(n)]
            assert len(records) == 12 and row["reads_per_configuration"] == 8 * n
            assert abs(row["median_percent"] - 100 * np.median(records)) < 1e-12
    for robot, count in (("Panda", 56), ("UR10e", 48)):
        row, = [x for x in summary["grouped"] if
                (x["robot"], x["arm"], x["start"], x["noise_SD_Nm"]) ==
                (robot, "3loads_full", "from_URDF", 0.01)]
        assert row["reads_per_configuration"] == count
    return summary["benchmark_case_count"]


def check_printed_splits(manuscript_root):
    printed = (manuscript_root / "appendix_simulation.tex").read_text(encoding="utf-8")
    reports = [
        read(DATA / name / "report.json")["rows"]
        for name in ("mujoco_probe_check", "mujoco_terminal_check")
    ]
    for split, rows in enumerate(zip(*reports, strict=True), start=1):
        cells = [str(split)]
        for row in rows:
            for field in ("full_relative_error", "unprobed_relative_error"):
                mantissa, exponent = f"{row[field]:.2e}".split("e")
                cells.append(rf"\({mantissa}\times10^{{{int(exponent)}}}\)")
        assert " & ".join(cells) + r"\\" in printed, cells


def check_subspace_model():
    rng = np.random.default_rng(12)
    a = np.linalg.qr(rng.normal(size=(7, 3)))[0]
    h = rng.normal(size=(2, 7, 7))
    m0, increment = h @ h.swapaxes(1, 2)
    t0, t1 = np.eye(3) + 0.2 * rng.normal(size=(2, 3, 3))
    e = (np.eye(7) - a @ a.T) @ rng.normal(size=(2, 7, 3))
    loaded = m0 + increment
    b0 = m0 @ (a @ t0 + e[0])
    b1 = loaded @ (a @ t1 + e[1])
    observed = b1 @ np.linalg.inv(t1) - b0 @ np.linalg.inv(t0)
    expected = increment @ a + loaded @ e[1] @ np.linalg.inv(t1) - m0 @ e[0] @ np.linalg.inv(t0)
    np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    manuscript = ROOT.parent if (ROOT.parent / "main.tex").is_file() else None
    parser.add_argument("--manuscript-root", type=Path, default=manuscript,
                        help="Check the ideal split errors printed in this manuscript.")
    args = parser.parse_args()
    worst = check_direct_probes()
    cases = check_geometry_archive()
    if args.manuscript_root is not None:
        check_printed_splits(args.manuscript_root)
    check_subspace_model()
    aligned = read(ROOT / "aligned_val_report.json")
    assert len(aligned["records"]) == 20
    assert max(x["rows"] for x in aligned["filters"]) == 0
    assert read(DATA / "two_direction_ambiguity.json")["local_ambiguity_dimension"] == 3
    print(f"Verified {cases} perturbed-geometry scores and matched nominal-start controls.")
    print(f"Four two-configuration MuJoCo splits, 48 reads per split, maximum relative error {worst:.4g}.")
    print("Verified the acceleration-subspace identity and the held-out alignment counts.")


if __name__ == "__main__":
    main()
