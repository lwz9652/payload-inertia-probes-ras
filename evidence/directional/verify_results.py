"""Verify the probe budgets and aggregate over independent simulated geometries."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(root):
    records = read(root / "confirmation/report.json")
    detail = [json.loads(line) for line in (root / "confirmation/cases.jsonl").read_text().splitlines()]
    assert detail == [row for robot in records["robots"] for row in robot["scores"]]
    result = []
    for robot in records["robots"]:
        protocol = robot["protocol"]
        assert protocol["distributed_geometry"] and protocol["seed_offset"] == 137
        assert protocol["repeats"] == 12
        assert protocol["heldout_configurations"] == 128 and protocol["heldout_payloads"] == 48
        arms = {arm["name"]: arm for arm in protocol["directions"]}
        for arm in arms.values():
            direction = np.array(arm["matrix"])
            np.testing.assert_allclose(direction.T @ direction, np.eye(direction.shape[1]), atol=1e-12)
            assert arm["torque_vectors_per_configuration"] == 2 * direction.shape[1] * (1 + arm["loads"])
        tri = np.array(arms["3loads_3dirs"]["matrix"])
        duo = np.array(arms["4loads_2dirs"]["matrix"])
        unprobed = np.array(protocol["unprobed_test_directions"])
        np.testing.assert_allclose(tri.T @ unprobed, 0, atol=1e-12)
        np.testing.assert_allclose(duo.T @ unprobed, 0, atol=1e-12)
        assert len(robot["clean_pointwise"]) == 36
        assert max(r["max_relative_error"] for r in robot["clean_pointwise"]) < 1e-8
        assert len({(r["trial"], r["arm"], r["start"], r["noise_SD_Nm"])
                    for r in robot["scores"]}) == len(robot["scores"]) == 324
        groups = defaultdict(list)
        for row in robot["scores"]:
            groups[row["noise_SD_Nm"], row["arm"], row["start"]].append(row)
        for (noise, arm, start), observations in groups.items():
            assert {r["trial"] for r in observations} == set(range(12))
            matrix = np.array([r["relative_error"] for r in observations])
            direction = np.array([r["unprobed_direction_error"] for r in observations])
            assert np.all(np.isfinite(matrix)) and np.all(np.isfinite(direction))
            result.append(dict(
                robot=protocol["robot"], noise_SD_Nm=noise, arm=arm, start=start,
                reads_per_configuration=arms[arm]["torque_vectors_per_configuration"],
                median_RM_percent=float(100 * np.median(matrix)),
                p90_RM_percent=float(100 * np.quantile(matrix, .90)),
                worst_RM_percent=float(100 * np.max(matrix)),
                median_unprobed_percent=float(100 * np.median(direction)),
                worst_unprobed_percent=float(100 * np.max(direction)),
            ))
    assert len(result) == 54

    direct = read(root / "mujoco_probe_check/report.json")
    assert direct["fit_folds"] == 4 and direct["actual_mj_inverse_calls_per_configuration"] == 24
    assert direct["full_response_measurement_baseline"] == 56
    assert direct["new_configurations"] == 128 and direct["new_payloads"] == 64
    for row in direct["rows"]:
        with np.load(root / f"mujoco_probe_check/fold{row['fold']}_probes.npz", allow_pickle=False) as data:
            assert data["torques"].shape == (32, 4, 3, 2, 7)
            assert np.all(np.isfinite(data["torques"]))
        assert max(row["full_relative_error"], row["unprobed_relative_error"]) < 1e-8
    exact = read(root / "exact_rank_verified.json")
    assert [r["nullspace_constraint_rank"] for r in exact["cases"]] == [17, 11]
    return dict(
        benchmark_case_count=len(detail), configurations_per_case=128, payloads_per_case=48,
        grouped=result, direct_MuJoCo=direct, rational_certificates=exact,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    full = aggregate(Path(__file__).resolve().parent)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(full, indent=2, allow_nan=False) + "\n")
    print(f"Verified {full['benchmark_case_count']} results, all probe budgets, and 4 MuJoCo folds.")
    for row in full["grouped"]:
        if row["noise_SD_Nm"] > 0 and row["start"] in ("from_URDF", "from_torques"):
            print(row["robot"], row["noise_SD_Nm"], row["arm"], row["start"],
                  f"median {row['median_RM_percent']:.3f}%, max {row['worst_RM_percent']:.3f}%, "
                  f"unprobed {row['median_unprobed_percent']:.3f}%")
