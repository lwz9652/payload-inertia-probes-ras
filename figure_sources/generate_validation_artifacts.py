"""Generate the fixture and acquisition tables from the archived results."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from draw_directional_figures import BLUE, GRAY, GRID, ROOT, RUST, plt, save


EVIDENCE = ROOT / "evidence"
VALIDATION = EVIDENCE / "validation"
ROBOTS = ("Panda", "UR10e")


def load(directory, name):
    return json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))


def fit_errors(records, metric, count, selector=lambda row: row["fit"]["final"]):
    assert len(records) == count
    values = [selector(row) for row in records]
    errors = np.array([value[metric] if value is not None else np.inf for value in values])
    assert np.all(errors >= 0)
    return 100 * errors


def number(value):
    if not np.isfinite(value):
        return r"\(\infty\)"
    return f"{value:.3f}" if value < 10 else f"{value:.2f}"


def statistics(errors):
    return [number(np.median(errors)), number(np.max(errors)),
            f"{np.count_nonzero(errors < 1)}/{len(errors)}"]


def table(label, caption, columns, headers, rows):
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        *headers,
        r"\midrule",
    ]
    for row in rows:
        lines.append(row if isinstance(row, str) else " & ".join(map(str, row)) + r"\\")
    return "\n".join([*lines, r"\bottomrule", r"\end{tabular}", r"\end{table}"])


def read_allocation_records():
    original = load(VALIDATION, "pose_budget")
    repeat = load(VALIDATION, "pose_budget_repeat")
    assert original["seed"] != repeat["seed"]
    conditions = (original["rows"], repeat["rows"])
    keys = lambda data: {(r["robot"], r["trial"], r["configurations"]) for r in data}
    assert keys(conditions[0]) == keys(conditions[1])
    assert all(c["torque_read_sd_Nm"] == .01 for c in (original, repeat))
    allocations = []
    for configurations in (2, 4, 8, 16, 32):
        repetitions = 32 // configurations
        errors = []
        for records in conditions:
            selected = [r for r in records if r["configurations"] == configurations]
            assert all(r["signed_reads"] == 640 and r["repeats"] == repetitions for r in selected)
            errors.append(fit_errors(selected, "R_M", 24))
        allocations.append((configurations, repetitions, errors))
    return allocations


def fixture_records():
    data = load(VALIDATION, "attachment_baseline")
    assert data["torque_read_sd_Nm"] == .01
    rows = sorted(data["rows"], key=lambda row: row["trial"])
    assert len(rows) == 48
    return rows


def fixture_table(records):
    rows = []
    for label, target in (("Fixture only", "fixture_only"),
                          ("Fixture and joints", "fixture_and_joint_errors")):
        for index, (name, model) in enumerate(
            (("Nominal", "uncorrected"), ("Fixture transform", "fixture"), ("Full chain", "chain"))
        ):
            row = [label if index == 0 else "", name]
            for robot in ROBOTS:
                selected = [r for r in records if (r["robot"], r["target"]) == (robot, target)]
                row.extend(statistics(fit_errors(
                    selected, "R_M", 12, lambda r: r[model]["final"] if model == "chain" else r[model],
                )))
            rows.append(row)
        if target == "fixture_only":
            rows.append(r"\addlinespace")
    return table(
        "fixture-baseline",
        r"Transfer error \(R_M\) in percent with an uncertain fixture pose and, in the second condition, joint geometry errors. Fits share calibration data at \(0.01\,\torqueunit\) raw noise.",
        "llrrrrrr",
        [r"& & \multicolumn{3}{c}{Panda} & \multicolumn{3}{c}{UR10e}\\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Target changes & Model fit & Median & Max. & Below \(1\%\) & Median & Max. & Below \(1\%\)\\"],
        rows,
    )


def acquisition_table():
    aligned = load(VALIDATION, "alignment_high")["rows"]
    combined = load(VALIDATION, "alignment_joint_high")["rows"]
    encoder = load(VALIDATION, "sampled_high")
    friction = load(VALIDATION, "friction_filter")
    assert all(r["sigma"] == .01 for r in [*aligned, *combined])
    assert encoder["raw_torque_sigma"] == friction["noise_Nm"] == .01

    rows = []
    for label, mode, level, records in (
        ("Alignment, no offset", "aligned", 0., aligned),
        ("Alignment, half levels combined", "joint", .5, combined),
        ("Alignment, full levels combined", "joint", 1., combined),
    ):
        selected = [r for r in records if (r["mode"], r["level"]) == (mode, level)]
        rows.append([label, *statistics(fit_errors(selected, "R_A", 12, lambda r: r["heldout"]))])

    rows.append(r"\addlinespace")
    for label, name in (
        (r"\(2\,\mathrm{\mu rad}\) encoder noise, no correction", "uncorrected"),
        (r"\(2\,\mathrm{\mu rad}\) encoder noise, corrected", "quiet"),
    ):
        selected = [r for r in encoder["rows"] if r["name"] == name]
        rows.append([label, *statistics(fit_errors(selected, "R_A", 12))])

    rows.append(r"\addlinespace")
    for label, speed in (("Same-direction crossing", "same_direction"), ("Near reversal", "near_reversal")):
        selected = [
            r for r in friction["rows"]
            if r["speed"] == speed and r["acquisition"] == "velocity_error_with_filter"
        ]
        rows.append([f"{label}, velocity offsets and filter", *statistics(fit_errors(selected, "R_A", 12))])

    return table(
        "acquisition",
        r"Acquisition tests across 12 simulated Panda cases at \(0.01\,\torqueunit\) raw noise. "
        r"The full levels include 3~mrad of cross-load position error, 0.3~mrad position and "
        r"10~mrad/s velocity within a sign pair, and \(0.3\%\) off-span acceleration. "
        r"The half levels halve all four offsets. \(R_A\) measures transfer outside the probe span.",
        "lrrr", [r"Condition & Median \(R_A\) (\%) & Max. \(R_A\) (\%) & Below \(1\%\)\\"], rows,
    )


def plot_fixture(records, preview):
    fig, panels = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    fig.subplots_adjust(left=.105, bottom=.24, right=.98, top=.88, wspace=.17)
    treatments = (("fixture_only", BLUE, "Fixture only"), ("fixture_and_joint_errors", RUST, "Fixture and joints"))
    for ax, robot in zip(panels, ROBOTS, strict=True):
        for group, (target, color, label) in enumerate(treatments):
            data = [r for r in records if (r["robot"], r["target"]) == (robot, target)]
            assert len(data) == 12
            fixture = [100 * r["fixture"]["R_M"] for r in data]
            chain = [100 * r["chain"]["final"]["R_M"] for r in data]
            position = np.array([0., 1.]) + (.035 if group else -.035)
            for pair in zip(fixture, chain, strict=True):
                ax.plot(position, pair, color=color, alpha=.25, lw=.72)
            ax.plot(position, [np.median(fixture), np.median(chain)], color=color,
                    marker="D" if group else "o", markersize=4.4, linewidth=1.25, label=label)
        ax.axhline(1., color=GRAY, linestyle="--", linewidth=.68, zorder=0)
        ax.grid(axis="y", color=GRID, linewidth=.45)
        ax.set_axisbelow(True)
        ax.set_title(robot, loc="left", fontweight="bold", fontsize=9.5)
        ax.set(yscale="log", xlim=(-.22, 1.22), ylim=(.029, 6.),
               xticks=[0., 1.], xticklabels=["Fixture transform", "Full chain"],
               yticks=[.03, .1, .3, 1., 3.], yticklabels=["0.03", "0.1", "0.3", "1", "3"])
        ax.minorticks_off()
        ax.tick_params(axis="both", length=2.6)
        ax.spines[["top", "right"]].set_visible(False)
    panels[0].set_ylabel(r"Held-out $R_M$ (%)")
    handles, labels = panels[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .005),
               ncol=2, frameon=False)
    save(fig, "fixture_vs_chain", preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the frozen table sources")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    records = fixture_records()
    outputs = {
        "fixture_baseline.tex": fixture_table(records),
        "acquisition_summary.tex": acquisition_table(),
    }
    for name, body in outputs.items():
        content = ("% Generated from the case-level records by generate_validation_artifacts.py.\n\n"
                   + body + "\n").encode("ascii")
        path = ROOT / name
        if args.check:
            assert content == path.read_bytes(), f"{name} differs from the case-level results"
        else:
            path.write_bytes(content)
        print(f"{name}, SHA-256 {hashlib.sha256(content).hexdigest()}")
    if not args.check:
        plot_fixture(records, args.preview_dir)


if __name__ == "__main__":
    main()
