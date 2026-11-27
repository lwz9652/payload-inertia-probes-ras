"""Render Figure 1 from the archived MuJoCo Panda states and torque reads."""

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#202b33"
GRAY = "#616e77"
GRID = "#e4e9ed"
BLUE = "#235a78"
RUST = "#a0422d"
ROOT = Path(__file__).resolve().parents[1]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "mathtext.fontset": "dejavusans",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.labelsize": 8.15,
    "axes.edgecolor": GRAY,
    "axes.linewidth": 0.65,
    "xtick.color": INK,
    "ytick.color": INK,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def arm_panel(fig):
    """Use the native MuJoCo views cached with their projected marker positions."""
    with np.load(ROOT / "figure_sources/panda_mujoco_views.npz", allow_pickle=False) as views:
        overview = views["overview"]
        joint7 = views["joint7"]
        wrists = views["wrists"]
        wrist_marks = views["wrist_marks"]

    left, right = .022, .275
    arm = fig.add_axes((left, .387, right - left, .503))
    arm.imshow(overview, extent=(0, 1, 0, 1), interpolation="none",
               aspect=overview.shape[0] / overview.shape[1])
    arm.set_axis_off()
    arm.annotate("Joint 7", xy=joint7, xytext=(.968, .825),
                 ha="right", va="bottom", fontsize=7.8, color=INK,
                 arrowprops={"arrowstyle": "-", "color": GRAY, "lw": .7,
                             "shrinkA": 2.5, "shrinkB": 2.0})

    fig.text((left + right) / 2, .372, r"Wrist turn: $+0.50$ rad",
             ha="center", va="center", fontsize=7.8)
    width = .116
    height = width * fig.get_figwidth() / fig.get_figheight()
    for x, image, points, color, label in zip(
            (left, right - width), wrists, wrist_marks, (BLUE, RUST),
            ("Configuration 1", "Configuration 2"), strict=True):
        wrist = fig.add_axes((x, .063, width, height))
        wrist.imshow(image, extent=(0, 1, 0, 1), interpolation="none", aspect="equal")
        wrist.set_axis_off()
        origin, direction, center = points
        wrist.annotate("", xy=direction, xytext=origin,
                       arrowprops={"arrowstyle": "-|>", "color": color, "lw": .9,
                                   "mutation_scale": 6, "shrinkA": 0, "shrinkB": 0})
        wrist.plot(*origin, "o", color=color, markeredgecolor="white", markeredgewidth=.45,
                   markersize=2.9)
        wrist.plot(*center, "o", color=INK, markeredgecolor="white", markeredgewidth=.55,
                   markersize=3.8)
        wrist.annotate("$x$", xy=direction, xytext=(1.5, 2.0), textcoords="offset points",
                       color=color, fontsize=7.45)
        fig.text(x + width / 2, .036, label, ha="center", va="center", fontsize=7.4,
                 color=color)


def chart_grid(ax):
    ax.grid(axis="y", color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="both", length=2.8, pad=3, labelsize=7.8)


def reads_panel(reads_ax, contrast_ax, reads):
    joints = np.arange(1, 8)
    unloaded, loaded = .5 * (reads[:, 0] - reads[:, 1])
    contrast = 1000 * (loaded - unloaded)
    for j, v0, v1 in zip(joints, unloaded, loaded, strict=True):
        reads_ax.plot([j - .12, j + .12], [v0, v1], color=GRAY, lw=.85, zorder=2)
    reads_ax.scatter(joints - .12, unloaded, label="Unloaded", s=17, linewidth=.85,
                     facecolor="white", edgecolor=GRAY, zorder=3)
    reads_ax.scatter(joints + .12, loaded, label="Loaded", s=18, marker="D",
                     linewidth=.4, facecolor=BLUE, edgecolor="white", zorder=4)
    reads_ax.set(ylabel="Half diff. (N·m)", xlim=(.55, 7.45),
                 ylim=(-.86, .79), yticks=[-.8, -.4, 0, .4], xticks=joints)
    reads_ax.tick_params(axis="x", labelbottom=False)
    reads_ax.legend(loc="lower left", bbox_to_anchor=(0., 1.015), ncols=2,
                    frameon=False, columnspacing=.7,
                    labelspacing=.2, borderpad=.15, handlelength=1.05,
                    handletextpad=.3, fontsize=7.9)
    contrast_ax.bar(joints, contrast, width=.58, color=BLUE, zorder=4)
    contrast_ax.set(ylabel="Contrast (mN·m)", xlabel="Joint",
                    xlim=(.55, 7.45), ylim=(-177, 130), yticks=[-150, -75, 0, 75],
                    xticks=joints)
    for ax in (reads_ax, contrast_ax):
        chart_grid(ax)
        ax.axhline(0, color=GRAY, lw=.7, zorder=1)
        ax.yaxis.set_label_coords(-.205, .50)


def candidates_panel(ax, results):
    # Both candidates use the first configuration alone.
    assert np.max(results[:, :, 0]) < 1e-6  # mN·m
    selected, other = np.sort(results[:, :, 1], axis=1).T
    assert np.max(selected) < 1e-6
    assert 24.2 < np.min(other) < 24.4 and 88.8 < np.max(other) < 89.1
    splits = np.arange(1, len(results) + 1)
    ax.scatter(splits, selected, marker="o", s=21, c=BLUE, edgecolor="white", lw=.45,
               label="Selected chain", zorder=5)
    ax.scatter(splits, other, marker="D", s=23, c=RUST, edgecolor="white", lw=.45,
               label="Other chain", zorder=5)
    for split, value in zip(splits, other, strict=True):
        ax.annotate(f"{value:.1f}", (split, value), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    color=RUST, fontsize=7.1)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncols=2,
              fontsize=7.35, handlelength=1, handletextpad=.28, columnspacing=.64,
              borderaxespad=0)
    ax.set(xlim=(.52, 4.48), ylim=(-6, 106), yticks=[0, 25, 50, 75, 100],
           ylabel="Whitened RMS error (mN·m)", xlabel="Calibration split", xticks=splits)
    chart_grid(ax)


def load_experiment(source):
    from directional_recovery import recover_initial, shared_reference_whitener
    from robot_geometry import reverse_axes, serial_body

    scores = []
    for fold in range(4):
        filename = source / f"evidence/directional/mujoco_terminal_check/fold{fold}_probes.npz"
        with np.load(filename, allow_pickle=False) as archive:
            q, basis, torques, inertia = [archive[name] for name in
                                         ("q", "applied_directions", "torques", "inertias")]
        if fold == 0:
            example_reads = torques[0, [0, 1], 0]
        np.testing.assert_allclose(q[1, :-1], q[0, :-1], atol=0, rtol=0)
        np.testing.assert_allclose(abs(q[1, -1] - q[0, -1]), .5, atol=1e-12)
        half = .5 * (torques[..., 0, :] - torques[..., 1, :])
        observed = (half[:, 1:] - half[:, :1]).swapaxes(-1, -2)
        local, _ = recover_initial(inertia, basis, observed[0])
        whitener = shared_reference_whitener(len(inertia))
        error = []
        for sign in (+1, -1):
            axes, points = reverse_axes(q[:1], (sign * local)[None])
            jacobian = serial_body(q, axes, points)
            prediction = jacobian[:, None].swapaxes(-1, -2) @ inertia @ jacobian[:, None] @ basis
            whitened = np.einsum("ml,qlik->qmik", whitener, prediction - observed)
            error.append(1000 * np.sqrt(np.mean(whitened**2, axis=(1, 2, 3))))
        scores.append(error)
    return example_reads, np.asarray(scores)


def save_manuscript_figure(fig, name, output, preview):
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", facecolor="white")
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}.png", dpi=270, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "evidence/directional"))
    reads, scores = load_experiment(ROOT)
    fig = plt.figure(figsize=(7.18, 2.92), layout=None, facecolor="white")
    reads_ax = fig.add_axes((.357, .566, .250, .265))
    contrast = fig.add_axes((.357, .146, .250, .331))
    candidates = fig.add_axes((.717, .146, .255, .685))
    for left, label in ((.024, "(a) Panda in MuJoCo"),
                        (.352, "(b) Paired torque reads"),
                        (.713, "(c) Second-configuration test")):
        fig.text(left, .939, label, fontweight="bold", fontsize=9.2, va="center")
    arm_panel(fig)
    reads_panel(reads_ax, contrast, reads)
    candidates_panel(candidates, scores)
    save_manuscript_figure(fig, "figure1_directional_calibration",
                           args.output_dir, args.preview_dir)
    print("Whitened RMS errors (mN·m), [split, candidate, configuration]:")
    print(np.array2string(scores, precision=5))


if __name__ == "__main__":
    main()
