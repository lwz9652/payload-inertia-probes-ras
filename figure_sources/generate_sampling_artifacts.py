"""Generate editable plots and LaTeX tables from the paired sampling records."""

from pathlib import Path
import json

import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from draw_directional_figures import BLUE, GRAY, GRID, INK, ROOT, RUST, plt, save


STUDY = ROOT / "evidence" / "sampling"
ROBOTS = ("Panda", "UR10e")
STARTS = ("torque", "nominal")
LABELS = {
    "base": "Base",
    "extra_load": "Extra load",
    "extra_direction": "Extra direction",
    "two_direction_ref": "Two-direction reference",
    "sweep": "Complete sweep",
}
PRIMARY = tuple(LABELS)[:3]
MARKS = ("o", "s", "D", "^", "v")
COLORS = (GRAY, BLUE, RUST, "#525688", "#406953")


def load(name):
    return json.loads((STUDY / name).read_text(encoding="utf-8"))


def cases(report, robot):
    selected = [case for case in report["cases"] if case["robot"] == robot]
    assert len(selected) == 12
    return selected


def result(case, design, start):
    matches = [r for r in case["scores"] if (r["design"], r["initialization"]) == (design, start)]
    row, = matches
    assert row["metrics"] is not None
    return row


def matrix_values(selected, design, start):
    return np.asarray([100 * result(case, design, start)["metrics"]["R_M"] for case in selected])


def percents(value):
    return f"{value:.2f}" if abs(value) >= 10 else f"{value:.3f}"


def action_values(selected, design, start, metric):
    return 1000 * np.asarray([result(case, design, start)["metrics"]["torque_error_Nm"][metric]
                              for case in selected])


def conditional(value):
    return f"\\textbf{{{percents(value)}}}" if value >= 1 else percents(value)


def line(*cells):
    return " & ".join(map(str, cells)) + r"\\" + "\n"


def start_table(caption, label, columns, headings, placement="!tbp"):
    return (f"\\begin{{table}}[{placement}]\n\\centering\n\\small\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n{headings}\\midrule\n")


TABLE_END = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def summary_table(original, repeat, plan):
    headings = (
        r"& & \multicolumn{4}{c}{Panda} & \multicolumn{4}{c}{UR10e} & \multicolumn{2}{c}{Below \(1\%\)}\\" + "\n"
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-12}" + "\n"
        r"& & \multicolumn{2}{c}{Torque} & \multicolumn{2}{c}{Nominal}"
        r" & \multicolumn{2}{c}{Torque} & \multicolumn{2}{c}{Nominal} & & \\" + "\n"
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}" + "\n" +
        line("Design", "Reads", *("D1", "D2") * 5)
    )
    reports = (original, repeat)
    table = start_table(
        r"Median held-out matrix error \(R_M\) (\%) for the two initializers and noise draws "
        r"(D1, D2). The read count is per configuration, with Panda/UR10e "
        r"shown for the sweep. The last columns count geometries below \(1\%\) with torque-derived "
        r"starts, out of 24 per draw. All nominal-model fits are below \(1\%\).",
        "tab:sampling-summary", "@{}l" + "r" * 11 + "@{}", headings,
    )
    table = table.replace(r"\small" + "\n", r"\small" + "\n" + r"\setlength{\tabcolsep}{3pt}" + "\n", 1)
    for design, label in LABELS.items():
        reads = []
        medians = []
        for robot in ROBOTS:
            selected = [cases(report, robot) for report in reports]
            configuration, = [
                item for item in plan["robots"][robot]["designs"] if item["name"] == design
            ]
            directions = np.asarray(configuration["directions"]).shape[1]
            reads.append(2 * (configuration["loads"] + 1) * directions)
            medians.extend(percents(np.median(matrix_values(geometries, design, start)))
                           for start in STARTS for geometries in selected)
        counts = [np.count_nonzero(matrix_values(report["cases"], design, "torque") < 1)
                  for report in reports]
        reads_label = str(reads[0]) if reads[0] == reads[1] else f"{reads[0]}/{reads[1]}"
        table += line(label, reads_label, *medians, *counts)
    return table + TABLE_END


def paired_matrix_table(report, draw_label, suffix, robot):
    headings = line("Geometry", "Base", "+Load", "+Direction", r"\(\Delta_L\)", r"\(\Delta_k\)",
                    r"\(4\times2\)", "Sweep")
    table = start_table(
        rf"{robot}, {draw_label} noise draw. Paired \(R_M\) in percent with torque-derived starts. "
        r"\(\Delta_L\) and \(\Delta_k\) are changes from the base fit in percentage points. "
        r"The \(4\times2\) column is the independently designed reference. Bold marks \(R_M\geq1\%\).",
        f"tab:case-matrix-{robot}-{suffix}", "rrrrrrrr", headings,
    )
    for case in cases(report, robot):
        values = [100 * result(case, design, "torque")["metrics"]["R_M"] for design in LABELS]
        base, more_load, more_direction, sparse, sweep = values
        table += line(case["trial"] + 1, conditional(base), conditional(more_load),
                      conditional(more_direction), percents(more_load - base),
                      percents(more_direction - base), conditional(sparse), conditional(sweep))
    return table + TABLE_END


def paired_action_table(report, robot):
    headings = (
        r" & \multicolumn{3}{c}{Base} & \multicolumn{3}{c}{Extra load} & \multicolumn{3}{c}{Extra direction}\\" + "\n"
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}" + "\n" +
        line("Geometry", *("Median", "P95", "Max.") * 3)
    )
    table = start_table(
        rf"{robot} action errors \(E_\tau\) in \(\mathrm{{mN{{\cdot}}m}}\) for the three primary "
        r"torque-derived fits in the original draw. Columns summarize the same held-out actions within each geometry.",
        f"tab:case-actions-{robot}", "rrrrrrrrrr", headings,
    )
    for case in cases(report, robot):
        row = [case["trial"] + 1]
        for design in PRIMARY:
            error = result(case, design, "torque")["metrics"]["torque_error_Nm"]
            row.extend(f"{1000 * error[stat]:.2f}" for stat in ("median", "p95", "max"))
        table += line(*row)
    return table + TABLE_END


def action_summary_table(report, draw_label, suffix):
    headings = line("Robot", "Design", "Start", r"\(R_A\) (\%)", "Median", "P95", "Max.")
    table = start_table(
        rf"Action errors for the {draw_label} noise draw. The \(R_A\) column gives the geometry median. "
        r"For \(E_\tau\) (in \(\mathrm{mN{\cdot}m}\)), Median and P95 are medians of the corresponding "
        r"within-geometry summaries, and Max. is the largest individual action error in any of the 12 geometries.",
        f"tab:action-summary-{suffix}", "lllrrrr", headings,
    )
    for robot in ROBOTS:
        selected = cases(report, robot)
        for design, label in LABELS.items():
            for start in STARTS:
                rel = [100 * result(case, design, start)["metrics"]["R_A"] for case in selected]
                median, p95, worst = (action_values(selected, design, start, metric)
                                      for metric in ("median", "p95", "max"))
                table += line(robot, label, start.capitalize(), percents(np.median(rel)),
                              f"{np.median(median):.2f}", f"{np.median(p95):.2f}", f"{worst.max():.2f}")
        table += "\\addlinespace\n"
    return table + TABLE_END


def plot_matrix(report):
    fig, axes = plt.subplots(2, 2, figsize=(6.35, 3.70), sharex=True, sharey="row")
    fig.subplots_adjust(left=.125, right=.985, bottom=.15, top=.88, hspace=.44, wspace=.13)
    x = np.arange(len(PRIMARY))
    for row, start in enumerate(STARTS):
        for col, robot in enumerate(ROBOTS):
            ax = axes[row, col]
            vals = np.array([matrix_values(cases(report, robot), d, start) for d in PRIMARY]).T
            ax.set_yscale("log" if row == 0 else "linear")
            ax.set_ylim((.15, 160) if row == 0 else (.19, .68))
            ax.set_title(f"{robot}  |  {'Torque-derived starts' if start == 'torque' else 'Nominal-model starts'}",
                         fontsize=8.1, loc="left", pad=5)
            ax.grid(axis="y", color=GRID, linewidth=.55)
            if row == 0:
                ax.axhline(1., color=GRAY, linewidth=.72, linestyle=(0, (3, 3)), zorder=2)
            for case in vals:
                ax.plot(x, case, linewidth=.60, color=GRAY, alpha=.41, zorder=3)
            for i, (marker, color) in enumerate(zip(MARKS[:3], COLORS[:3], strict=True)):
                y = vals[:, i]
                ax.scatter(np.full(y.size, i), y, s=14, marker=marker,
                           color=color, edgecolors="white", linewidth=.25, zorder=4)
                ax.text(i, .94, f"{sum(y < 1)}/12", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7.4, color=color)
            ax.set_xticks(x, [LABELS[d] for d in PRIMARY])
            ax.set_xlim(-.32, 2.32)
            ax.yaxis.set_major_locator(FixedLocator([.2, 1, 10, 50] if row == 0 else [.2, .3, .4, .5, .6]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.tick_params(axis="both", which="minor", length=0)
            if col == 0:
                ax.set_ylabel(r"$R_M$ (%)")
    save(fig, "paired_probe_errors", None)


def plot_action(report):
    fig, axes = plt.subplots(2, 2, figsize=(6.35, 4.1), sharex=True)
    fig.subplots_adjust(left=.235, right=.985, bottom=.16, top=.92, hspace=.40, wspace=.16)
    for row, start in enumerate(STARTS):
        for col, robot in enumerate(ROBOTS):
            ax = axes[row, col]
            selected = cases(report, robot)
            reference = 1000 * np.median([case["reference_Nm"]["median"] for case in selected])
            ax.set_xscale("log")
            ax.set_xlim(.66, 390)
            ax.set_ylim(4.55, -.72)
            ax.set_title(f"{robot}  |  {'Torque-derived starts' if start == 'torque' else 'Nominal-model starts'}",
                         fontsize=8.1, loc="left", pad=5)
            ax.grid(axis="x", color=GRID, linewidth=.55)
            ax.axvline(reference, color=INK, linewidth=.75, linestyle=(0, (3, 3)), zorder=3)
            for i, (design, marker, color) in enumerate(zip(LABELS, MARKS, COLORS, strict=True)):
                values = np.sort(action_values(selected, design, start, "p95"))
                jitter = np.tile([-.12, 0., .12], 4)
                ax.scatter(values, i + jitter, s=13, marker=marker, color=color, alpha=.82,
                           edgecolors="white", linewidth=.28, zorder=4)
                ax.vlines(np.median(values), i - .23, i + .23, color=INK, linewidth=1.35, zorder=5)
            ax.xaxis.set_major_locator(FixedLocator([1, 3, 10, 30, 100, 300]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_yticks(range(len(LABELS)), list(LABELS.values()) if col == 0 else [])
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", which="minor", length=0)
    fig.text(.61, .043, r"Within-geometry P95 of $E_\tau$ (mN$\cdot$m)", ha="center")
    save(fig, "unprobed_action_errors", None)


if __name__ == "__main__":
    original = load("sampling_metrics.json")
    repeat = load("replicate_metrics.json")
    (ROOT / "sampling_summary.tex").write_text(
        summary_table(original, repeat, load("probe_design.json")), encoding="utf-8",
    )
    lines = []
    for robot in ROBOTS:
        lines.append(paired_matrix_table(original, "original", "original", robot))
        lines.append(paired_matrix_table(repeat, "second", "second", robot))
        lines.append(paired_action_table(original, robot))
    lines.append(action_summary_table(original, "original", "original"))
    lines.append(action_summary_table(repeat, "second", "second"))
    (STUDY / "case_level_tables.tex").write_text("\n".join(lines), encoding="utf-8")
    plot_matrix(original)
    plot_action(original)
