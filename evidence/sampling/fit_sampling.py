"""Fit every probe design on the same frozen contrasts and two-start budget."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from common import INITIALIZATIONS, ROBOTS, configuration_file, designs, load_npz
from fit_chain import initialize_from_data
from recentered_fit import best_recentered
from study_common import read, write


def fit_one_case(robot, trial, inputs, protocol, replicate_seed=None):
    file = configuration_file(inputs,robot,trial)
    q, inertias, full, clean, nominal = load_npz(
        file, ("calibration_q","calibration_inertias","canonical_contrasts","clean_contrasts","nominal_starts"),
    )
    if replicate_seed is not None:
        from directional_recovery import contrast_noise
        n = q.shape[1]
        rng = np.random.default_rng(replicate_seed + 100*ROBOTS.index(robot) + trial)
        errors = [contrast_noise(rng,len(inertias),n,n,protocol["torque_read_sd_Nm"]) for _ in q]
        full = clean + np.asarray(errors)
    results = []
    for design, loads, a in designs(protocol,robot):
        measured = full[:,:loads] @ a
        parameters = inertias[:loads]
        torque_starts, usable = initialize_from_data(q,parameters,a,measured)
        for initialization, starts in zip(INITIALIZATIONS,(torque_starts,nominal),strict=True):
            record = dict(robot=robot,trial=trial,design=design,initialization=initialization,starts=len(starts))
            if initialization == "torque":
                record["usable_pointwise_estimates"] = usable
            if len(starts) != 2:
                results.append(dict(record,selected=None,geometry=None,stages=[]))
                continue
            geometry, selected, stages = best_recentered(q,parameters,a,measured,starts)
            assert all(stage["nfev"] <= 1600 for stage in stages)
            np.testing.assert_allclose(np.linalg.norm(geometry[0],axis=1),1.,atol=2e-12)
            np.testing.assert_allclose(np.sum(geometry[0]*geometry[1],axis=1),0.,atol=2e-12)
            results.append(dict(record,selected=selected,geometry=[z.tolist() for z in geometry],stages=stages))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs",type=Path,required=True)
    parser.add_argument("--design",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--trials",type=int,default=12)
    parser.add_argument("--workers",type=int,default=7)
    parser.add_argument("--replicate-seed",type=int)
    args=parser.parse_args()
    plan=read(args.design)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(fit_one_case,robot,trial,args.inputs,plan,args.replicate_seed)
              for robot in ROBOTS for trial in range(args.trials)]
        rows=[]
        for job in as_completed(jobs):
            group=job.result()
            rows.extend(group)
            print(group[0]["robot"],group[0]["trial"]+1,len(rows),"fits complete",flush=True)
    rows.sort(key=lambda x:(x["robot"],x["trial"],x["design"],x["initialization"]))
    write(args.output,dict(replicate_seed=args.replicate_seed, fits=rows))
