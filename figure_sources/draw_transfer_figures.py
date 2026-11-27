"""Draw the held-out transfer and fixed-budget comparisons."""

import argparse
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patheffects import withStroke
from matplotlib.ticker import FuncFormatter
import numpy as np

import generate_sampling_artifacts as sampling
from generate_validation_artifacts import read_allocation_records
from draw_figure1 import BLUE, GRAY, INK, ROOT, RUST, chart_grid, plt, save_manuscript_figure


DESIGNS = ("base", "extra_load", "extra_direction")
LABELS = ("Base", "+1 load", "+1 direction")
START_STYLES = (("torque", BLUE, -.17), ("nominal", RUST, .17))
DRAW_STYLES = (("Draw 1", True, "-", -.074), ("Draw 2", False, (0, (1.0, 1.05)), .074))


def sample_points(ax, center, values, color, *, filled=True, spread=.065, marker="o",
                  median_half_width=.064, median_style="-", median_color=None):
    """Spread sorted cases horizontally so coincident results remain readable."""
    values = np.sort(np.asarray(values))
    assert np.all(np.isfinite(values) & (values > 0))
    pattern = np.array([0., -.72, .67, -.28, .28, -1., 1.])
    offsets = np.resize(pattern, len(values)) * spread
    ax.scatter(center + offsets, values, s=9, marker=marker,
               facecolors=color if filled else "none", edgecolors=color,
               linewidths=.52, alpha=.73, zorder=3)
    center_line = np.median(values)
    ax.plot((center - median_half_width, center + median_half_width), (center_line, center_line),
            color=median_color or color, ls=median_style, lw=1.5,
            solid_capstyle="butt", dash_capstyle="butt", zorder=5,
            path_effects=[withStroke(linewidth=3.0, foreground="white")])


def prepare_error_axis(ax, ticks, limits):
    ax.set(yscale="log", ylim=limits, yticks=ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.tick_params(axis="y", which="minor", length=0)
    chart_grid(ax)


def matrix_panel(ax, reports, robot):
    for (start, color, start_shift) in START_STYLES:
        for report, (_, filled, linestyle, draw_shift) in zip(reports, DRAW_STYLES, strict=True):
            cases = sampling.cases(report, robot)
            for position, design in enumerate(DESIGNS):
                values = sampling.matrix_values(cases, design, start)
                assert np.all((values > .17) & (values < 90))
                sample_points(ax, position + start_shift + draw_shift, values, color,
                              filled=filled, spread=.045, median_style=linestyle)
    ax.axhline(1, color=GRAY, ls=(0, (4, 3)), lw=.78, zorder=1)
    ax.set(xlim=(-.46, 2.46), xticks=range(len(DESIGNS)), xticklabels=LABELS)
    prepare_error_axis(ax, [.2, 1, 5, 20, 60], (.17, 90))


def action_panel(ax, original):
    for robot, shift, marker in (("Panda", -.13, "o"), ("UR10e", .13, "D")):
        cases = sampling.cases(original, robot)
        for position, design in enumerate(DESIGNS):
            values = sampling.action_values(cases, design, "torque", "p95")
            assert np.all((values > .8) & (values < 520))
            sample_points(ax, position + shift, values, GRAY, marker=marker,
                          spread=.055, median_half_width=.09, median_color=INK)
    ax.set(xlim=(-.46, 2.46), xticks=range(len(DESIGNS)), xticklabels=LABELS,
           ylabel=r"95th percentile of $E_\tau$ (mN·m)")
    prepare_error_axis(ax, [1, 3, 10, 30, 100, 300], (.8, 520))


def transfer_figure(reports, output, preview):
    original, repeat = reports
    assert original["replicate_seed"] is None and repeat["replicate_seed"] is not None
    assert [(c["robot"], c["trial"]) for c in original["cases"]] == [
        (c["robot"], c["trial"]) for c in repeat["cases"]
    ]

    fig = plt.figure(figsize=(7.18, 2.90))
    panda = fig.add_axes((.095, .240, .253, .572))
    ur10e = fig.add_axes((.385, .240, .253, .572))
    actions = fig.add_axes((.745, .240, .239, .572))
    matrix_panel(panda, reports, "Panda")
    matrix_panel(ur10e, reports, "UR10e")
    action_panel(actions, original)
    panda.set_ylabel(r"Matrix error, $R_M$ (%)")
    ur10e.tick_params(axis="y", labelleft=False)
    for ax, title in ((panda, "(a) Panda"), (ur10e, "(b) UR10e"),
                      (actions, "(c) Unprobed torques")):
        ax.set_title(title, loc="left", weight="bold", fontsize=9.2, y=1.10)
    starts = [
        Line2D([], [], color=color, linewidth=2, label=name)
        for (_, color, _), name in zip(START_STYLES, ("Torque starts", "Nominal starts"), strict=True)
    ]
    draws = [
        Line2D([], [], ls=linestyle, lw=1.25, marker="o", color=INK, markersize=3.2,
               markerfacecolor=INK if filled else "white", label=label)
        for label, filled, linestyle, _ in DRAW_STYLES
    ]
    robots = [
        Line2D([], [], ls="none", marker=marker, color=GRAY, markersize=3.6,
               markerfacecolor=GRAY, label=robot)
        for robot, marker in (("Panda", "o"), ("UR10e", "D"))
    ]
    legend_args = dict(loc="lower center", frameon=False, fontsize=7.3, handlelength=1.55,
                       handletextpad=.35, columnspacing=1.02)
    fig.legend(handles=starts + draws, bbox_to_anchor=(.365, .029), ncols=4, **legend_args)
    fig.legend(handles=robots, bbox_to_anchor=(.862, .029), ncols=2, **legend_args)
    save_manuscript_figure(fig, "figure2_probe_design_comparison", output, preview)


def fixed_budget_figure(allocations, output, preview):
    fig, (counts, magnitudes) = plt.subplots(1, 2, figsize=(7.18, 2.88))
    fig.subplots_adjust(left=.093, right=.976, bottom=.213, top=.775, wspace=.33)
    x = np.arange(len(allocations))
    labels = [f"{configurations} × {sets}" for configurations, sets, _ in allocations]
    draws = (("Draw 1", BLUE, "o", "-", -.16),
             ("Draw 2", RUST, "D", (0, (4, 2)), .16))
    for index, (label, color, marker, linestyle, offset) in enumerate(draws):
        errors = np.asarray([values[index] for _, _, values in allocations])
        assert errors.shape == (5, 24)
        assert np.all((errors > .14) & (errors < 290))
        success = np.count_nonzero(errors < 1, axis=1)
        counts.plot(x + offset / 2, success, color=color, marker=marker, ms=4.8,
                    mec="white", mew=.5, lw=1.2, ls=linestyle, label=label)
        for position, values in enumerate(errors):
            sample_points(magnitudes, position + offset, values, color, marker=marker,
                          spread=.102, median_half_width=.127, median_color=INK)

    counts.set_title("(a) Geometries below 1%", loc="left", weight="bold", fontsize=9.3, pad=7)
    counts.set(ylabel="Geometries (of 24)", ylim=(-1.35, 25.7), yticks=[0, 6, 12, 18, 24])
    magnitudes.set_title("(b) Error across geometries", loc="left", weight="bold",
                         fontsize=9.3, pad=7)
    magnitudes.set_ylabel(r"Matrix error, $R_M$ (%)")
    prepare_error_axis(magnitudes, [.2, 1, 3, 10, 30, 100], (.14, 290))
    magnitudes.axhline(1, ls=(0, (4, 3)), color=GRAY, lw=.8, zorder=1)
    for ax in (counts, magnitudes):
        ax.set(xlim=(-.37, len(labels) - .63), xticks=x, xticklabels=labels)
    chart_grid(counts)
    handles, names = counts.get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, ncols=2, loc="upper center",
               bbox_to_anchor=(.533, .987), fontsize=8.4, handletextpad=.45)
    fig.text(.534, .064, "Calibration configurations × sets at each configuration",
             ha="center", va="center", fontsize=8.5)
    save_manuscript_figure(fig, "figure3_fixed_read_budget", output, preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    original = sampling.load("sampling_metrics.json")
    repeat = sampling.load("replicate_metrics.json")
    transfer_figure((original, repeat), args.output_dir, args.preview_dir)
    fixed_budget_figure(read_allocation_records(), args.output_dir, args.preview_dir)


if __name__ == "__main__":
    main()

