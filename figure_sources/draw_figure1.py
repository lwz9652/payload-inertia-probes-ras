"""Render Figure 1 from the archived MuJoCo Panda states and torque reads."""

import argparse
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation


INK = "#202b33"
GRAY = "#616e77"
GRID = "#e4e9ed"
BLUE = "#235a78"
RUST = "#a0422d"
PALE = "#f1f4f5"
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


def panda_frames(xml, q):
    """Traverse the archived bodies, including the actual payload frame."""
    parent = ET.parse(xml).getroot().find("worldbody")
    position = np.zeros(3)
    orientation = Rotation.identity()
    frames = []
    for index, name in enumerate([f"link{i}" for i in range(8)] + ["hand", "end_effector"]):
        body = parent.find(f"body[@name='{name}']")
        if body is None:
            raise ValueError(f"Missing archived body {name}")
        position = position + orientation.apply(np.fromstring(body.get("pos", "0 0 0"), sep=" "))
        quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        orientation = orientation * Rotation.from_quat(np.roll(quat, -1))
        if 1 <= index <= 7:
            assert body.find("joint").get("name") == f"joint{index}"
            orientation = orientation * Rotation.from_rotvec([0, 0, q[index - 1]])
        frames.append((position, orientation))
        parent = body
    return frames


def view_projection(azimuth, elevation):
    az, el = np.deg2rad([azimuth, elevation])
    look = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    horizontal = np.array([-np.sin(az), np.cos(az), 0.])
    return np.column_stack((horizontal, np.cross(look, horizontal)))


def link_surfaces(ax, frames, projection, meshes):
    from draw_directional_figures import mesh_triangles

    look = np.cross(projection[:, 0], projection[:, 1])
    view = np.column_stack((projection, look))
    mapped = np.concatenate([
        (origin + orientation.apply(mesh_triangles(meshes / f"link{index}.obj").reshape(-1, 3)))
        .reshape(-1, 3, 3) @ view
        for index, (origin, orientation) in enumerate(frames[:8])
    ])
    normal = np.cross(mapped[:, 1] - mapped[:, 0], mapped[:, 2] - mapped[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    front = normal[:, 2] > 0
    visible = mapped[front]
    order = np.argsort(visible.mean(axis=1)[:, 2])
    light = np.array([-.36, .48, .8])
    shade = np.clip(normal[front] @ light, 0, 1)
    dark, pale = np.array([.38, .42, .45]), np.array([.87, .89, .91])
    colors = dark + (pale - dark) * shade[:, None]
    ax.add_collection(PolyCollection(visible[order, :, :2], facecolors=colors[order],
                                    edgecolors="face", linewidth=.08, zorder=2))
    return mapped[..., :2].reshape(-1, 2)


def arm_panel(ax, frames_a, frames_b, center, projection, meshes):
    outlines = link_surfaces(ax, frames_a, projection, meshes)
    uv = np.array([p for p, _ in frames_a]) @ projection
    ax.plot(*uv[7], "o", mfc="white", mec=RUST, mew=1, ms=4.6, zorder=5)
    joint = uv[7]
    axis = frames_a[7][1].apply([0., 0., 1.])
    axis_line = (frames_a[7][0] + np.array([-.075, .16])[:, None] * axis) @ projection
    ax.plot(*axis_line.T, color=RUST, lw=.85, ls=(0, (3, 2)), zorder=4)
    ax.annotate("Joint 7\n0.50 rad turn", xy=joint,
                xytext=(.197, .131), ha="left", va="top", fontsize=8.0, color=RUST,
                arrowprops={"arrowstyle": "-", "lw": .7, "color": RUST,
                            "connectionstyle": "arc3,rad=-.04"})

    origin, rotation_a = frames_a[-1]
    origin_b, rotation_b = frames_b[-1]
    np.testing.assert_allclose(origin_b, origin, atol=1e-13)
    np.testing.assert_allclose((rotation_a.inv() * rotation_b).magnitude(), .5, atol=1e-13)
    frame_uv = origin @ projection
    ax.plot(*uv[[7, 9]].T, color=GRAY, lw=.7, ls=(0, (2.3, 2.3)), zorder=3)
    ax.plot(*frame_uv, "o", mfc=BLUE, mec="white", mew=.45, ms=4, zorder=7)
    for rotation, color, linestyle in ((rotation_b, RUST, (0, (2, 2))),
                                        (rotation_a, BLUE, "-")):
        ends = (origin + rotation.apply(.105 * np.eye(3))) @ projection
        for end, symbol in zip(ends, "xyz", strict=True):
            if color == RUST and symbol == "z":
                continue
            ax.annotate("", xy=end, xytext=frame_uv,
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": .75,
                                    "linestyle": linestyle, "mutation_scale": 7,
                                    "shrinkA": 0, "shrinkB": 0}, zorder=5)
            if color == BLUE:
                direction = end - frame_uv
                extra = .016 * direction / np.linalg.norm(direction)
                ax.text(*(end + extra), symbol, color=BLUE, ha="center", va="center",
                        fontsize=7.5, fontstyle="italic")
    ax.annotate("Attachment\nframe", xy=frame_uv, xytext=(.198, .519),
                fontsize=8, ha="left", va="bottom",
                arrowprops={"arrowstyle": "-", "lw": .65, "color": GRAY, "shrinkB": 4,
                            "connectionstyle": "angle,angleA=0,angleB=72,rad=0"})

    center_uv = (origin + rotation_a.apply(center)) @ projection
    # The archive specifies spatial inertia. Show a point and offset, not a CAD load shape.
    ax.plot([frame_uv[0], center_uv[0]], [frame_uv[1], center_uv[1]],
            color=GRAY, lw=.7, ls=(0, (1.5, 2.4)), zorder=3)
    ax.plot(*center_uv, "o", mfc=INK, mec="white", mew=.6, ms=4.6, zorder=8)
    ax.annotate("Load COM", xy=center_uv, xytext=(.476, .265),
                fontsize=8, ha="right", va="top",
                arrowprops={"arrowstyle": "-", "lw": .65, "color": GRAY, "shrinkB": 3})
    ax.set(aspect="equal", xlim=(outlines[:, 0].min() - .014, .485),
           ylim=(outlines[:, 1].min() - .012, max(outlines[:, 1].max() + .022, .57)))
    ax.set_axis_off()


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
            example_q = q
            example_reads = torques[0, [0, 1], 0]
            cross = inertia[0, :3, 3:]
            center = np.array([cross[2, 1], cross[0, 2], cross[1, 0]]) / inertia[0, 3, 3]
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
    return example_q, example_reads, center, np.asarray(scores)


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
    q, reads, center, scores = load_experiment(ROOT)
    xml = ROOT / "evidence/followup/panda_contact_free.xml"
    meshes = ROOT / "figure_sources/panda_meshes"
    projection = view_projection(-65, 16)
    frames = [panda_frames(xml, v) for v in q]
    fig = plt.figure(figsize=(7.18, 2.92), layout=None, facecolor="white")
    arm = fig.add_axes((.022, .115, .255, .775))
    reads_ax = fig.add_axes((.357, .566, .250, .265))
    contrast = fig.add_axes((.357, .146, .250, .331))
    candidates = fig.add_axes((.717, .146, .255, .685))
    for left, label in ((.024, "(a) Panda calibration setup"),
                        (.352, "(b) Paired torque reads"),
                        (.713, "(c) Second-configuration test")):
        fig.text(left, .939, label, fontweight="bold", fontsize=9.2, va="center")
    arm_panel(arm, *frames, center, projection, meshes)
    reads_panel(reads_ax, contrast, reads)
    candidates_panel(candidates, scores)
    frames_legend = [
        Line2D([], [], color=BLUE, lw=1.05, label="First frame"),
        Line2D([], [], color=RUST, lw=1.05, ls=(0, (3, 2)), label="Second frame"),
    ]
    fig.legend(handles=frames_legend, loc="lower center", bbox_to_anchor=(.150, .040),
               frameon=False, ncols=2, columnspacing=.75, handlelength=1.35,
               handletextpad=.40, fontsize=7.9, borderaxespad=0)
    save_manuscript_figure(fig, "figure1_directional_calibration",
                           args.output_dir, args.preview_dir)
    print("Whitened RMS errors (mN·m), [split, candidate, configuration]:")
    print(np.array2string(scores, precision=5))


if __name__ == "__main__":
    main()
