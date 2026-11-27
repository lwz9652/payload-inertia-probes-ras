"""Check paired inputs, fits, train diagnostics, and held-out metrics."""

import argparse
from pathlib import Path

import numpy as np

from common import HERE, INITIALIZATIONS, PRIMARY, ROBOTS, designs, read
from evaluate_sampling import compute


DRAW_FILES = (
    ("fits.json", "sampling_metrics.json", (18, 24, 14, 22, 24)),
    ("replicate_fits.json", "replicate_metrics.json", (16, 24, 19, 21, 24)),
)


def same_record(expected, actual):
    if isinstance(expected, dict):
        assert set(expected) == set(actual)
        for key, value in expected.items():
            same_record(value, actual[key])
    elif isinstance(expected, list):
        for a, b in zip(expected, actual, strict=True):
            same_record(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=5e-9)
    else:
        assert expected == actual


def check_design(plan):
    for robot, n in zip(ROBOTS, (7, 6), strict=True):
        group = plan["robots"][robot]
        directions = np.asarray(group["evaluation_directions"])
        np.testing.assert_allclose(directions.T @ directions, np.eye(n - 5), atol=2e-13)
        base, extra_load, extra_direction, _, sweep = list(designs(plan, robot))
        np.testing.assert_array_equal(base[2], extra_load[2])
        np.testing.assert_array_equal(base[2], extra_direction[2][:, :3])
        assert extra_direction[1] == base[1]
        np.testing.assert_array_equal(sweep[2], np.eye(n))
        assert group["selected_direction_index"] == np.argmax(group["nominal_scores"])
        for (name, loads, a), reads in zip(designs(plan, robot), (24, 30, 32, 20, 10*n), strict=True):
            assert 2*a.shape[1]*(loads + 1) == reads
            np.testing.assert_allclose(a.T @ a, np.eye(a.shape[1]), atol=2e-13)
            if name != "sweep":
                np.testing.assert_allclose(a.T @ directions, 0., atol=2e-13)


def check_fit_and_counts(result, archive, expected):
    scores = {start: {row["design"]: 0 for row in result["cases"][0]["scores"]}
              for start in INITIALIZATIONS}
    fits = {(row["robot"], row["trial"], row["design"], row["initialization"]): row
            for row in archive["fits"]}
    assert len(fits) == 240 and len(result["cases"]) == 24
    failures = 0
    for case in result["cases"]:
        for row in case["scores"]:
            assert row["metrics"] is not None
            kind = row["initialization"]
            design = row["design"]
            key = case["robot"], case["trial"], design, kind
            fitted = fits[key]
            rms = [stage["train_rmse"] for stage in fitted["stages"]]
            assert len(rms) == 2 and fitted["selected"] == int(np.argmin(rms))
            assert all(0 < stage["nfev"] <= 1600 for stage in fitted["stages"])
            success = row["metrics"]["R_M"] < .01
            scores[kind][design] += success
            if not success and kind == "torque" and design in PRIMARY:
                nominal = fits[(*key[:3], "nominal")]
                assert nominal["stages"][nominal["selected"]]["train_rmse"] < min(rms)
                failures += 1
    order = (*PRIMARY, "two_direction_ref", "sweep")
    assert tuple(scores["torque"][name] for name in order) == expected
    assert all(value == 24 for value in scores["nominal"].values())
    return failures


def verify(replay=None):
    plan = read(HERE / "probe_design.json")
    check_design(plan)
    if replay is not None:
        same_record(plan, read(replay / "probe_design.json"))
    for fits_name, metrics_name, expected in DRAW_FILES:
        original = read(HERE / fits_name)
        archived = read(HERE / metrics_name)
        path = HERE if replay is None else replay
        archive = original if replay is None else read(replay / fits_name)
        result = compute(path / "inputs", plan, archive)
        same_record(archived, result)
        failures = check_fit_and_counts(result, archive, expected)
        run = "stored fits" if replay is None else "fresh fits"
        print(f"{fits_name}: {run}, 240 fits and 24 geometries; below 1% {expected}; "
              f"{failures} primary torque fits with a lower-loss nominal fit on the same data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-directory", type=Path)
    args = parser.parse_args()
    verify(args.replay_directory)
