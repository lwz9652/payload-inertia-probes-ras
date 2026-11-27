"""Aggregate the solver, geometry-prior, acquisition, and descriptor controls."""

from pathlib import Path

import numpy as np

from controls_common import HERE, FOLLOWUP
from study_common import read
from verify_controls import pools, select


ROBOTS = ("Panda", "UR10e")
SCREENS = ("sigma6", "operator", "d_opt", "product")


def scores(rows, key="R_M"):
    return np.array([100*r["fit"]["final"][key] for r in rows])


def report(label, values):
    med, tail, worst = np.quantile(values, [.5,.9,1.])
    count = int(np.sum(values < 1))
    print(label, f"{count}/{len(values)}", f"median={med:.5g}", f"p90={tail:.5g}", f"max={worst:.5g}")
    return count


def check_candidates():
    fixed = read(HERE / "matched_fixed_candidate_cases.json")
    moving = read(HERE / "moving_candidate_cases.json")
    old = read(FOLLOWUP / "candidate_cases.json")
    assert len(fixed) == len(moving) == len(old) == 384
    assert sum(r["fit"]["R_M"] < .01 for r in old) == 272
    for left, right in zip(fixed, moving, strict=True):
        assert [left[k] for k in ("robot","trial","index")] == [right[k] for k in ("robot","trial","index")]
        assert left["scores"] == right["scores"]
        assert left["fit"]["eligible"] == right["fit"]["eligible"]
        assert all(len(r["fit"]["stages"])==2 and max(x["nfev"] for x in r["fit"]["stages"])<=1600
                   for r in (left,right))
        assert not any(t["crossed_initial_hemisphere"] for t in left["fit"]["stages"])
    frozen_success = scores(fixed) < 1
    moving_success = scores(moving) < 1
    assert (sum(frozen_success), sum(moving_success)) == (265,354)
    assert sum(~frozen_success & moving_success) == 89 and sum(frozen_success & ~moving_success) == 0
    positive = 0
    for robot, counts in zip(ROBOTS, ((12,12,12,12),(12,12,11,12)), strict=True):
        for label,data in (("fixed",fixed),("moving",moving)):
            group = list(pools(data, robot))
            report(f"{robot} {label} all", scores([r for g in group for r in g]))
        for choice, count in zip(SCREENS, counts, strict=True):
            assert report(f"{robot} moving {choice}", scores([select(g, choice) for g in group])) == count
        positive += sum(np.any(scores(g) >= 1) for g in group)
    assert positive == 18


def check_prior():
    data = read(HERE / "prior_design_cases.json")
    assert len(data) == 264
    for bound, outcomes in ((45,(24,18)),(90,(17,20)),(135,(5,21))):
        for kind, count in zip(("geometry", "torque"), outcomes, strict=True):
            r = [v for v in data if (v["bound"],v["design"],v["initializer"]) == (bound,"degraded",kind)]
            assert len(r)==24 and report(f"degraded {bound} degrees {kind}", scores(r)) == count
    random = [r for r in data if r["design"]=="random"]
    assert len(random)==24 and report("random directions, torque starts", scores(random))==19


def check_sampled():
    data = read(HERE / "sampled_cases.json")
    assert (data["encoder_Hz"],data["samples"],data["raw_torque_sigma"]) == (500,41,.001)
    cases = data["rows"]
    assert len(cases)==120
    for key, target, successes in (
        ("uncorrected",8.964578,0), ("quiet",.137962,12), ("noisy",.659925,12),
        ("noisy_four",.334496,12), ("clock",.283941,12), ("very_weak",2.414957,0),
    ):
        rows = [r for r in cases if r["name"]==key]
        assert len(rows)==len({r["trial"] for r in rows})==12
        x = scores(rows,"R_A")
        assert report(f"sampled windows {key}",x)==successes
        np.testing.assert_allclose(np.median(x),target,atol=1e-6)


def check_descriptors():
    data = read(HERE / "descriptor_cases.json")["rows"]
    assert len(data)==240
    for mode, level in (("mass_fraction",.005),("mount_translation_m",.002),("mount_rotation_deg",.5)):
        rows = [r for r in data if (r["mode"],r["level"])==(mode,level)]
        assert len(rows)==24 and report(f"descriptor error {mode} {level}",scores(rows,"R_A"))==24


if __name__ == "__main__":
    check_candidates()
    check_prior()
    check_sampled()
    check_descriptors()
