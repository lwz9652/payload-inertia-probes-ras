"""Check the paired controls and summarize the case-level evidence."""

import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROBOTS = ("Panda", "UR10e")
CHOICES = ("sigma6", "operator", "d_opt", "product")
BOUNDARIES = {"cross_load_q": .003, "sign_pair_q": .0003, "sign_pair_v": .01, "off_space": .003}


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def percentile(values):
    return 100 * np.quantile(values, (.5, .9, 1.))


def pools(rows, robot):
    for trial in range(12):
        candidates = sorted((r for r in rows if (r["robot"], r["trial"]) == (robot, trial)),
                            key=lambda r: r["index"])
        assert [r["index"] for r in candidates] == list(range(16))
        yield candidates


def select(pool, method):
    return max(pool, key=lambda row: row["scores"][method])


def check_candidates():
    rows = read("candidate_cases.json")
    assert len(rows) == 384
    for robot, expected in zip(ROBOTS, ((12,12,11,12), (11,12,7,12)), strict=True):
        blocks = list(pools(rows, robot))
        methods = {"uniform": [r for b in blocks for r in b], "first": [b[0] for b in blocks]}
        methods.update({m: [select(b,m) for b in blocks] for m in CHOICES})
        limits = dict(zip(CHOICES, expected, strict=True))
        limits.update(uniform=136, first=7 if robot=="Panda" else 6)
        for method, entries in methods.items():
            result = [r["fit"]["R_M"] for r in entries]
            count = sum(r < .01 for r in result)
            assert count == limits[method], (robot, method, count)
            print(robot, method, f"{count}/{len(result)}", np.round(percentile(result), 5))


def check_initialization():
    rows = read("recentered_cases.json")
    assert len(rows) == 240 and all(len(r["stages"]) == 2 for r in rows)
    assert all(1 <= f["nfev"] <= 1600 for r in rows for f in r["stages"])
    new_recoveries = 0
    for robot, counts in zip(ROBOTS, ((11,12,12,11,5), (11,12,12,12,4)), strict=True):
        for start, count in zip(("torque","15","45","90","135"), counts, strict=True):
            blocks = [r for r in rows if (r["robot"],r["starts"],r["sigma"]) == (robot,start,.01)]
            result = [r["final"]["R_M"] for r in blocks]
            assert len(result) == 12 and sum(r < .01 for r in result) == count
            if start == "torque":
                new_recoveries += sum(r["final"]["R_M"] < .01 and min(r["starts_max_axis_degrees"])>=90
                                      for r in blocks)
            print(robot, start, f"{count}/12", np.round(percentile(result),5))
    assert new_recoveries == 13
    print("Below 1 percent from torque-derived starts outside the two fixed charts:", new_recoveries)


def assess_alignment(rows, mode, level, start_trial=0):
    subset = [r for r in rows if (r["mode"], r["level"]) == (mode, level)]
    assert sorted(r["trial"] for r in subset) == list(range(start_trial, start_trial + 12))
    assert all(r["heldout"] is not None for r in subset)
    error = [r["heldout"]["R_A"] for r in subset]
    return sum(e < .01 for e in error), percentile(error)


def check_alignment():
    data = read("alignment_cases.json")
    rows = data["rows"]
    assert len(rows) == 300 and data["mujoco"] == "3.2.6"
    assert max(r["corrected_within_subspace_max_abs_Nm"] for r in rows) < 3.7e-14
    print("Aligned", assess_alignment(rows, "aligned", 0.))
    for mode, passing in BOUNDARIES.items():
        levels = data["levels"][mode]
        summaries = [assess_alignment(rows, mode, amount) for amount in levels]
        excluded = next(i for i,(count,_) in enumerate(summaries) if count < 12)
        assert levels[excluded-1] == passing and all(count < 12 for count,_ in summaries[excluded:])
        print(mode, "passing level", passing, summaries[excluded-1][1],
              "next level", levels[excluded], summaries[excluded])
    joint = read("alignment_joint_cases.json")
    assert len(joint["rows"]) == 36 and joint["combined_levels"] == BOUNDARIES
    for fraction, count in ((.25,12),(.5,12),(1.,9)):
        found, values = assess_alignment(joint["rows"], "joint", fraction)
        assert found == count
        print("Joint", fraction, f"{found}/12", np.round(values, 5))
    confirmation = read("alignment_confirmation_cases.json")
    assert len(confirmation["rows"]) == 12 and confirmation["combined_levels"] == BOUNDARIES
    count, values = assess_alignment(confirmation["rows"], "joint", .5, start_trial=12)
    assert count == 12
    np.testing.assert_allclose(values[-1], .515, atol=.0005)
    print("Joint half-level confirmation", f"{count}/12", np.round(values, 5))


if __name__ == "__main__":
    check_candidates()
    check_initialization()
    check_alignment()
