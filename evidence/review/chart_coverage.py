"""Check when the local axis charts contain the simulated true chain."""

import argparse
from pathlib import Path

import numpy as np

from study_common import forward_urdf, matrices, recreate, write
from fit_chain import axes_from_home, initialize_from_data
from matched_review import spectral_complete, spectral_starts


def contains_axes(starts, true_axes):
    return any(np.all(np.sum(axes * true_axes, axis=1) > 0) for axes, _ in starts)


def audit(trial_limit):
    rows = []
    for robot in range(2):
        for trial in range(trial_limit):
            case = recreate(robot, trial)
            joint_count = case["q"].shape[1]
            true_home = forward_urdf(case["changed"], np.zeros((1, joint_count)),
                                     case["mount"], case["offset"])[0]
            true_axes, _ = axes_from_home(true_home)
            clean = matrices(case["j_cal"], case["catalog"])
            for design in ("3loads_full", "4loads_2dirs"):
                directions = case["directions"][design]
                load_count = int(design[0])
                inertias = case["catalog"][:load_count]
                for sigma in (0.001, 0.01):
                    measured = (clean[:, :load_count] + case["noises"][sigma][:, :load_count]) @ directions
                    kernel, _ = initialize_from_data(case["q"], inertias, directions, measured)
                    spectral = None
                    if design == "3loads_full":
                        factors = [spectral_complete(inertias, torques) for torques in measured]
                        spectral = contains_axes(spectral_starts(case["q"], factors), true_axes)
                    rows.append(dict(robot=case["name"], trial=trial, sigma=sigma, design=design,
                                     kernel=contains_axes(kernel, true_axes), spectral=spectral,
                                     nominal=contains_axes([case["reference"]], true_axes)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trial-count", default=12, type=int)
    args = parser.parse_args()
    rows = audit(args.trial_count)
    write(args.output, rows)


if __name__ == "__main__":
    main()
