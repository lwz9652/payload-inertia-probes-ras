"""Recompute matrix and physical action errors from frozen fits and held-out inputs."""

import argparse
from itertools import groupby
from pathlib import Path

import numpy as np

from common import INITIALIZATIONS, PRIMARY, ROBOTS, configuration_file, designs, load_npz
from directional_recovery import recover_initial
from study_common import matrices, read, relative_rmse, serial_body, write


def distribution(values):
    median, p95, worst = np.quantile(values, (.50,.95,1.))
    return {"median": float(median), "p95": float(p95), "max": float(worst)}


def action_example(q, loads, basis, true_action, action_error, at):
    pose, load, direction = map(int, at)
    tau = true_action[pose,load,:,direction]
    error = action_error[pose,load,:,direction]
    physical = loads[load]
    mass = float(physical[3,3])
    com = np.array([physical[2,4],physical[0,5],physical[1,3]])/mass
    return dict(configuration=pose,load=load,direction=direction,q_rad=q[pose].tolist(),
                mass_kg=mass,com_m=com.tolist(),acceleration_rad_s2=basis[:,direction].tolist(),
                reference_Nm=tau.tolist(),prediction_Nm=(tau+error).tolist(),
                reference_RMS_Nm=float(np.linalg.norm(tau)/np.sqrt(q.shape[1])),
                error_RMS_Nm=float(np.linalg.norm(error)/np.sqrt(q.shape[1])))


def evaluate_case(key, rows, inputs, plan):
    robot, trial = key
    q, test_loads, j, train_loads, clean = load_npz(
        configuration_file(inputs,robot,trial),
        ("heldout_q","heldout_inertias","heldout_body_jacobians","calibration_inertias","clean_contrasts"),
    )
    truth = matrices(j,test_loads)
    directions = np.asarray(plan["robots"][robot]["evaluation_directions"])
    true_actions = truth @ directions
    reference = np.linalg.norm(true_actions,axis=-2)/np.sqrt(q.shape[1])
    assert reference.shape == (128,48,q.shape[1]-5)
    ident = dict(robot=robot,trial=trial,actions=reference.size,reference_Nm=distribution(reference))
    conditions = []
    for name, count, a in designs(plan,robot):
        diagnostics = []
        for d in clean[:,:count] @ a:
            _, record = recover_initial(train_loads[:count],a,d)
            diagnostics.append([record["rank6_retained_singular_value"],record["operator_margin"]])
        conditions.append(dict(design=name, sigma6_eta=np.asarray(diagnostics).tolist()))
    scores, examples = [], []
    for r in rows:
        if r["geometry"] is None:
            scores.append(dict(design=r["design"],initialization=r["initialization"],metrics=None))
            continue
        fitted = serial_body(q,*[np.asarray(z) for z in r["geometry"]])
        m = matrices(fitted,test_loads)
        action_error = (m-truth) @ directions
        absolute = np.linalg.norm(action_error,axis=-2)/np.sqrt(q.shape[1])
        rm = relative_rmse(m,truth)
        ra = float(np.linalg.norm(action_error.ravel())/np.linalg.norm(true_actions.ravel()))
        scores.append(dict(design=r["design"],initialization=r["initialization"],
                           metrics=dict(R_M=rm,R_A=ra,torque_error_Nm=distribution(absolute))))
        if r["initialization"] == "torque" and r["design"] in PRIMARY:
            at = np.unravel_index(int(np.argmax(absolute)),absolute.shape)
            row = action_example(q,test_loads,directions,true_actions,action_error,at)
            examples.append(dict(row,robot=robot,trial=trial,design=r["design"],R_M=rm))
    for kind in INITIALIZATIONS:
        results = [r for r in scores if r["initialization"] == kind]
        baseline = next(r["metrics"] for r in results if r["design"] == "base")
        for row in results:
            m = row["metrics"]
            if m is not None and baseline is not None:
                row["delta_matrix_percentage_points"] = 100*(m["R_M"] - baseline["R_M"])
    return dict(ident,conditions=conditions,scores=scores), examples


def compute(inputs, plan, fit_archive):
    data, actions = [], []
    key = lambda r:(r["robot"],r["trial"])
    for case_key, group in groupby(sorted(fit_archive["fits"],key=key),key):
        rows = list(group)
        pairs = {(r["design"],r["initialization"]) for r in rows}
        assert len(rows) == len(pairs) == 5*len(INITIALIZATIONS)
        case, examples = evaluate_case(case_key,rows,inputs,plan)
        data.append(case)
        actions.extend(examples)
    return dict(replicate_seed=fit_archive["replicate_seed"],cases=data,
                largest_action_error_among_subpercent_matrix_fits=max(
                    (r for r in actions if r["R_M"] < .01),key=lambda x:x["error_RMS_Nm"],default=None),
                largest_primary_action_error=max(actions,key=lambda x:x["error_RMS_Nm"],default=None))


def summarize(report, plan):
    def median(data):
        return float(np.median(data))

    for robot in ROBOTS:
        cases = [c for c in report["cases"] if c["robot"] == robot]
        reference = median([c["reference_Nm"]["median"] for c in cases])
        print(f"{robot}: {len(cases)} geometries; {cases[0]['actions']} actions each; reference median {reference:.6g} Nm")
        print("scheme, start, count<1%, median RM pct, max RM pct, paired median pp, median RA pct, "
              "median E_tau, median within-case P95 E_tau, worst E_tau (Nm)")
        for scheme, _, _ in designs(plan,robot):
            for kind in INITIALIZATIONS:
                rows = [r for c in cases for r in c["scores"]
                        if (r["design"],r["initialization"]) == (scheme,kind)]
                valid = [r["metrics"] for r in rows if r["metrics"] is not None]
                rm = [100*r["R_M"] for r in valid]
                if not rm:
                    print(scheme,kind,"no fitted cases")
                    continue
                delta = [r["delta_matrix_percentage_points"] for r in rows
                         if "delta_matrix_percentage_points" in r]
                error = [r["torque_error_Nm"] for r in valid]
                numbers = [median(rm),max(rm),median(delta) if delta else float("nan"),
                           100*median([r["R_A"] for r in valid]),median([r["median"] for r in error]),
                           median([r["p95"] for r in error]),max(r["max"] for r in error)]
                print(scheme,kind,f"{sum(x<1 for x in rm)}/{len(rows)}",
                      " ".join(f"{x:.6g}" for x in numbers))
    for label in ("largest_action_error_among_subpercent_matrix_fits","largest_primary_action_error"):
        action = report[label]
        if action is not None:
            print(label,{k:v for k,v in action.items() if k not in ("reference_Nm","prediction_Nm")})


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--fits",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    plan=read(args.design)
    evidence=compute(args.inputs,plan,read(args.fits))
    write(args.output,evidence)
    summarize(evidence,plan)
