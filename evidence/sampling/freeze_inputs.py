"""Freeze nominal probe choices and reconstruct the matched R3 contrasts."""

import argparse
from pathlib import Path

import numpy as np

from common import EVIDENCE, NOISE_SD, ROBOTS, configuration_file
from coordinates import directional_residual
from geometry_review import perturb_start
from probe_benchmark import rank_screen
from study_common import BOXES, box_inertia, forward_urdf, matrices, read, recreate, write


def lock_designs():
    snapshots = read(EVIDENCE / "review" / "kinematic_inputs.json")
    catalog = np.asarray([box_inertia(*box) for box in BOXES])
    report = {"torque_read_sd_Nm": NOISE_SD, "robots": {}}
    for index, robot in enumerate(ROBOTS):
        model = snapshots[index]
        prior = read(EVIDENCE / "directional" / "confirmation" / f"{robot}_protocol.json")
        assert model["robot"] == robot and model["source_sha256"] == prior["urdf_sha256"]
        recorded = {d["name"]: np.asarray(d["matrix"]) for d in prior["directions"]}
        a3, a2 = recorded["3loads_3dirs"], recorded["4loads_2dirs"]
        heldout = np.asarray(prior["unprobed_test_directions"])
        projected = a2 - a3 @ (a3.T @ a2)
        basis, values, _ = np.linalg.svd(projected, full_matrices=False)
        assert len(values) == 2 and values[-1] > 1e-10
        bounds = np.asarray(model["bounds"])
        rng = np.random.default_rng(2026111701 + index)
        nominal_configurations = rng.uniform(bounds[:,0], bounds[:,1], (16,len(bounds)))
        chain = [(np.asarray(r), np.asarray(t), None if a is None else np.asarray(a))
                 for r,t,a in model["chain"]]
        j = forward_urdf(chain, nominal_configurations)
        angles = np.pi * np.arange(80) / 80
        candidates = basis @ np.stack((np.cos(angles), np.sin(angles)))
        scores = [rank_screen(j,catalog[:3],np.column_stack((a3,v))) for v in candidates.T]
        winner = int(np.argmax(scores))
        assert scores[winner] > 0.
        a4 = np.column_stack((a3,candidates[:,winner]))
        protocols = [
            ("base",3,a3), ("extra_load",4,a3), ("extra_direction",3,a4),
            ("two_direction_ref",4,a2), ("sweep",4,np.eye(len(bounds))),
        ]
        for name, _, a in protocols:
            np.testing.assert_allclose(a.T @ a,np.eye(a.shape[1]),atol=2e-13)
            if name != "sweep":
                np.testing.assert_allclose(a.T @ heldout,0.,atol=2e-13)
        np.testing.assert_allclose(heldout.T @ heldout,np.eye(len(bounds)-5),atol=2e-13)
        report["robots"][robot] = dict(
            nominal_source_sha256=model["source_sha256"],
            design_configurations=nominal_configurations.tolist(),
            direction_pool=candidates.T.tolist(), nominal_scores=scores,
            selected_direction_index=winner, evaluation_directions=heldout.tolist(),
            nominal_start_jitter_deg=5, nominal_start_jitter_m=.01,
            designs=[dict(name=name, loads=loads, directions=a.tolist(), reads=2*a.shape[1]*(loads+1))
                     for name,loads,a in protocols],
        )
    return report


def check_archived_residuals(case, canonical, checks):
    for row in checks:
        arm = row["arm"]
        a = case["directions"][arm]
        load_count = 4 if arm == "4loads_2dirs" else 3
        observed = canonical[:,:load_count] @ a
        residual = directional_residual(case["q"],case["catalog"][:load_count],a,observed)
        geometry = tuple(np.asarray(x) for x in row["fit"]["geometry"])
        actual = np.sqrt(np.mean(residual(geometry)**2))
        np.testing.assert_allclose(actual,row["fit"]["train_rmse"],atol=5e-13,rtol=1e-9)


def freeze_inputs(directory, trials):
    archived = read(EVIDENCE / "directional" / "confirmation" / "report.json")
    directory.mkdir()
    checks_total = 0
    for index, robot in enumerate(ROBOTS):
        scores = archived["robots"][index]["scores"]
        for trial in range(trials):
            case = recreate(index,trial)
            assert case["name"] == robot
            exact = matrices(case["j_cal"],case["catalog"])
            full = exact + case["noises"][NOISE_SD]
            checks = [row for row in scores if row["trial"] == trial and
                      row["noise_SD_Nm"] == NOISE_SD and row["start"] in ("from_torques","from_URDF")]
            assert len(checks) == 6
            check_archived_residuals(case,full,checks)
            checks_total += len(checks)
            jitter = perturb_start(case["reference"],5.,.01,
                                   np.random.default_rng(2026111803 + 100*index + trial))
            np.savez_compressed(configuration_file(directory,robot,trial),
                                calibration_q=case["q"], calibration_inertias=case["catalog"],
                                canonical_contrasts=full, clean_contrasts=exact,
                                heldout_q=case["held_q"], heldout_inertias=case["tests"],
                                heldout_body_jacobians=case["j_hold"],
                                nominal_starts=np.asarray([case["reference"],jitter]))
    print("Reconstructed cases:",2*trials,"archived calibration residuals reproduced:",checks_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--trials",type=int,default=12)
    args = parser.parse_args()
    if not 1 <= args.trials <= 12:
        parser.error("Reconstruction is defined for the 12 archived cases on each robot.")
    design = lock_designs()
    write(args.design,design)
    for robot, group in design["robots"].items():
        print("Locked nominal design:",robot,"candidate",group["selected_direction_index"],
              "meanings and costs",[(d["name"],d["reads"]) for d in group["designs"]],flush=True)
    freeze_inputs(args.inputs,args.trials)
