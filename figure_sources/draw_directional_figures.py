"""Draw the manuscript figures from the archived geometry and case records."""

import argparse
from itertools import product
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
MESHES = Path(__file__).resolve().parent / "panda_meshes"
sys.path.insert(0, str(EVIDENCE / "directional"))
from probe_benchmark import BOXES
sys.path.insert(0, str(EVIDENCE / 'followup'))
from verify_controls import assess_alignment, pools, select

INK = "#23323c"
GRAY = "#596671"
GRID = "#e0e5e8"
BLUE = "#245e7b"
RUST = "#a34f36"
CENTER_COLORS = ("#275e7b", "#a34f36", "#555976", "#3d725e")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "mathtext.fontset": "dejavusans",
    "axes.labelcolor": INK,
    "axes.edgecolor": GRAY,
    "axes.linewidth": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def read_json(name):
    return json.loads((EVIDENCE / "review" / name).read_text(encoding="utf-8"))


def save(fig, name, preview):
    output = ROOT / "figures"
    output.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg"):
        fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.045)
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def mesh_triangles(file):
    vertices, triangles = [], []
    for line in file.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(value) for value in fields[1:4]])
        if fields[0] == "f":
            face = [int(token.split("/")[0]) for token in fields[1:]]
            indices = [k - 1 if k > 0 else len(vertices) + k for k in face]
            triangles.extend((indices[0], indices[j], indices[j + 1]) for j in range(1, len(indices) - 1))
    return np.asarray(vertices)[np.asarray(triangles)]


def arm_meshes(model, q):
    points, rotation = np.zeros(3), np.eye(3)
    triangles = [mesh_triangles(MESHES / "link0.obj")]
    for local_rotation, displacement, axis in model["chain"]:
        points = points + rotation @ displacement
        rotation = rotation @ local_rotation
        if axis is not None:
            joint_axis = np.asarray(axis) / np.linalg.norm(axis)
            rotation = rotation @ Rotation.from_rotvec(q[len(triangles)-1] * joint_axis).as_matrix()
            mesh = mesh_triangles(MESHES / f"link{len(triangles)}.obj")
            triangles.append(mesh @ rotation.T + points)
    return np.concatenate(triangles), points, rotation


def drawing_view(triangles, azimuth, elevation):
    azimuth, elevation = np.deg2rad([azimuth, elevation])
    look = np.array([np.cos(azimuth) * np.cos(elevation), np.sin(azimuth) * np.cos(elevation),
                     np.sin(elevation)])
    side = np.array([-np.sin(azimuth), np.cos(azimuth), 0])
    up = np.cross(look, side)
    view = np.column_stack((side, up, look))
    mapped = triangles @ view
    normal = np.cross(mapped[:, 1] - mapped[:, 0], mapped[:, 2] - mapped[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    front = normal[:, 2] > 0
    visible = mapped[front]
    order = np.argsort(visible.mean(axis=1)[:, 2])
    light = np.array([-0.38, 0.55, 0.74])
    shade = np.clip(normal[front] @ light, 0, 1)
    dark = np.array([0.38, 0.42, 0.45])
    pale = np.array([0.87, 0.89, 0.91])
    face = dark + (pale - dark) * shade[:, None]
    return view, visible[order, :, :2], face[order]


def geometry_figure(preview):
    panda = read_json("kinematic_inputs.json")[0]
    q = [0., -0.48, 0., -1.95, 0., 1.63, 0.62]
    triangles, origin, orientation = arm_meshes(panda, q)
    view, visible, colors = drawing_view(triangles, -63, 17)
    fig, (arm, loads) = plt.subplots(1, 2, figsize=(6.4, 2.60),
                                     gridspec_kw={"width_ratios": (1.08, 1)})
    fig.subplots_adjust(left=.045, right=.985, bottom=.165, top=.87, wspace=.20)
    arm.add_collection(PolyCollection(visible, facecolors=colors, edgecolors="face", linewidth=.10))
    frame = (origin @ view)[:2]
    axes = ((origin + .14 * orientation.T) @ view)[:, :2]
    for end, name in zip(axes, ("x", "y", "z"), strict=True):
        arm.annotate("", xy=end, xytext=frame,
                     arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0, shrinkA=0, shrinkB=0))
        arrow = end - frame
        arm.text(*(end + .025 * arrow / np.linalg.norm(arrow)), name, color=BLUE,
                 va="center", ha="center", fontsize=9.5, fontstyle="italic")
    arm.scatter(*frame, s=13, color=BLUE, edgecolor="white", lw=.35, zorder=8)
    arm.annotate("Attachment\nframe", xy=frame, xycoords="data",
                 xytext=(.90, .31), textcoords="axes fraction", ha="left", va="center",
                 fontsize=8,
                 arrowprops=dict(arrowstyle="-", color=GRAY, lw=.55, shrinkA=4, shrinkB=4))
    arm.set_aspect("equal")
    arm.set_xlim(visible[..., 0].min()-.075, max(visible[..., 0].max()+.23, axes[:, 0].max()+.10))
    arm.set_ylim(visible[..., 1].min()-.024, visible[..., 1].max()+.028)
    arm.set_axis_off()
    arm.set_title("(a) Panda model", loc="left", fontsize=9, fontweight="bold", pad=8)

    signs = np.asarray(list(product((-1, 1), repeat=3)))
    labels = ((-.020, .013), (.013, -.024), (.012, .014), (-.036, -.020))
    for number, ((mass, center, sides, angles), color, shift) in enumerate(
            zip(BOXES, CENTER_COLORS, labels, strict=True), 1):
        rotation = Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        footprint = (signs * np.asarray(sides) * .5) @ rotation.T + center
        xy = footprint[:, :2]
        xy = xy[ConvexHull(xy).vertices]
        loads.add_patch(Polygon(xy, facecolor=matplotlib.colors.to_rgba(color, .10),
                                edgecolor=color, lw=1.05,
                                linestyle="--" if number == 4 else "-", zorder=2))
        c = np.asarray(center)[:2]
        loads.scatter(*c, color=color, s=20, linewidth=.5, edgecolor="white", zorder=4)
        loads.text(*(c+shift), rf"$\ell_{number}$", va="center", ha="center",
                   color=color, fontsize=9, zorder=5,
                   bbox=dict(facecolor="white", edgecolor="none", pad=.14, alpha=.78))
    loads.axhline(0, color=GRID, linewidth=.65, zorder=0)
    loads.axvline(0, color=GRID, linewidth=.65, zorder=0)
    loads.scatter(0, 0, marker="+", s=30, color=GRAY, linewidth=.9, zorder=3)
    loads.set(xlabel=r"$x$ (m)", ylabel=r"$y$ (m)", xlim=(-.235,.235), ylim=(-.225,.245),
              xticks=[-.2,-.1,0,.1,.2], yticks=[-.2,-.1,0,.1,.2])
    loads.set_aspect("equal")
    loads.tick_params(length=2.2, pad=2)
    loads.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    loads.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    loads.set_title("(b) Load footprints", loc="left", fontsize=9, fontweight="bold", pad=8)
    save(fig, "panda_probe_geometry", preview)


def make_error_panels(height, limits, ticks, *, bottom, top):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.4, height))
    fig.subplots_adjust(left=.115, right=.985, top=top, bottom=bottom, wspace=.18)
    for label, axis in zip(("(a) Panda", "(b) UR10e"), axes, strict=True):
        axis.set_title(label, loc="left", fontweight="bold", fontsize=9, pad=6)
        axis.set_yscale("log")
        axis.set_ylim(limits)
        axis.set_yticks(ticks)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        axis.tick_params(axis="y", which="both", length=2.7, width=.6)
        axis.tick_params(axis="y", which="minor", left=False)
        axis.tick_params(axis="x", which="both", length=0, pad=5)
        axis.grid(False, axis="x", which="both")
        axis.grid(axis="y", color=GRID, linewidth=.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].set_ylabel(r"$R_M$ (%)", labelpad=5)
    return fig, axes


def dots(ax, position, values, color, marker):
    offsets = np.linspace(-.082, .082, len(values))
    ax.scatter(position + offsets, 100 * np.asarray(values), s=15, color=color,
               marker=marker, alpha=.78, linewidth=.2, edgecolor="white", zorder=3)
    md = 100 * np.median(values)
    ax.plot([position-.094,position+.094],[md,md],color=color,lw=1.6,zorder=4)


def stages_figure(preview):
    records = read_json("matched_cases.json")
    fig, axes = make_error_panels(2.55, (.012, 25000), [0.02, .1, 1, 10, 100, 1000, 10000],
                                  bottom=.20, top=.79)
    for robot, ax in zip(("Panda", "UR10e"), axes, strict=True):
        rows = [r for r in records if (r["robot"], r["design"], r["sigma"]) == (robot, "3xfull", .001)]
        rows.sort(key=lambda row: row["trial"])
        assert len(rows) == 12
        dots(ax, 0, [r["independent_calibration_matrices_R_M"] for r in rows], GRAY, "s")
        for position, kind, outcome in ((1., "local", "spectral_local"), (2., "kernel", "spectral")):
            dots(ax, position-.14, [r[kind]["R_M"] for r in rows], BLUE, "o")
            dots(ax, position+.14, [r[outcome]["R_M"] for r in rows], RUST, "D")
        ax.set_xlim(-.29, 2.34)
        ax.set_xticks([0,1,2], ["Observed\nmatrices", "Pointwise\nrecovery", "Chain\ntransfer"])
    symbols = [
        Line2D([], [], color=color, marker=shape, ms=4.2, linestyle="none", label=label)
        for color, shape, label in ((GRAY,"s","Independent projection"), (BLUE,"o","Nullspace"),
                                    (RUST,"D","Spectral"))
    ]
    fig.legend(handles=symbols, loc="upper center", bbox_to_anchor=(.535, .984),
               ncol=3, frameon=False, handletextpad=.34, columnspacing=1.6, borderpad=0)
    save(fig, "complete_sweep_stages", preview)


def selection_figure(preview):
    rows = json.loads((EVIDENCE / "solver_controls" / "moving_candidate_cases.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(6.4, 3.50))
    fig.subplots_adjust(left=.106, right=.986, top=.78, bottom=.17, wspace=.17)
    rules = {
        "sigma6": (r"$\sigma_6(D_w)$", "#445d8c", "^"),
        "operator": (r"$\eta$", "#a34f36", "o"),
        "d_opt": ("D-optimal", "#6a527c", "x"),
        "product": (r"$\sigma_6(D_w)\eta$", "#1c6b59", "D"),
    }
    offsets = (-.285, -.095, .095, .285)
    for index, (robot, ax) in enumerate(zip(("Panda", "UR10e"), axes, strict=True)):
        for trial, candidates in enumerate(pools(rows, robot), 1):
            spread = np.linspace(-.06, .06, len(candidates))
            ax.scatter([100*row["fit"]["final"]["R_M"] for row in candidates], trial + spread,
                       s=9, color="#78848b", alpha=.48, linewidths=0, zorder=2)
            for offset, (key, (_, color, marker)) in zip(offsets, rules.items(), strict=True):
                chosen = select(candidates, key)
                x = chosen["fit"]["final"]["R_M"] * 100
                ax.plot(x, trial+offset, linestyle="none", marker=marker, ms=4.0,
                        color=color, markeredgewidth=.8, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(.145, 80)
        ax.set_xticks([.2,.5,1,2,5,10,20,50])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_ylim(12.6, .4)
        ax.set_yticks(range(1,13))
        ax.set_xlabel(r"$R_M$ (%)", labelpad=3)
        ax.set_title(f"({'ab'[index]}) {robot}", loc="left", fontweight="bold", fontsize=9, pad=6)
        ax.grid(axis="y", lw=.55, color=GRID)
        ax.grid(False, axis="x")
        ax.axvline(1., linestyle=(0,(4,2)), color=GRAY, linewidth=.7)
        ax.tick_params(axis="both", width=.6, length=2.5, labelsize=7.3)
        ax.tick_params(axis="x", which="minor", length=0)
        ax.spines[["right","top"]].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Geometry case")
    symbols = [Line2D([], [], color="#78848b", marker=".", ms=5.5, linestyle="none",
                      label="All candidates")]
    symbols += [Line2D([], [], color=color, marker=shape, ms=4.1, linestyle="none", label=label)
                for label,color,shape in rules.values()]
    fig.legend(handles=symbols, loc="upper center", bbox_to_anchor=(.555, .97),
               ncol=5, frameon=False, handletextpad=.28, columnspacing=1.25, borderpad=0)
    save(fig, "fourth_load_selection", preview)


def alignment_figure(preview):
    data = read_json_followup("alignment_cases.json")
    modes = (
        ("cross_load_q","(a) Position across loads","Position difference (mrad)",1000),
        ("sign_pair_q","(b) Position within a pair","Position difference (mrad)",1000),
        ("sign_pair_v","(c) Velocity within a pair","Velocity difference (mrad/s)",1000),
        ("off_space","(d) Acceleration outside span","Orthogonal acceleration (%)",100),
    )
    fig, axes = plt.subplots(2, 2, sharey=True, figsize=(6.4, 4.33))
    fig.subplots_adjust(left=.119, right=.982, top=.83, bottom=.105, hspace=.61, wspace=.18)
    for ax, (mode,title,xlabel,unit) in zip(axes.flat,modes,strict=True):
        levels = data["levels"][mode]
        values = [assess_alignment(data["rows"], mode, level)[1] for level in levels]
        med, _, worst = np.asarray(values).T
        x = unit * np.asarray(levels)
        ax.set(xscale="log", yscale="log", ylim=(.029, 75), xlim=(x[0]/1.24,x[-1]*1.24),
               xlabel=xlabel)
        ax.set_yticks([.03,.1,.3,1,3,10,30])
        ax.set_xticks(x)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(title, loc="left", fontweight="bold", fontsize=8.6, pad=5)
        ax.plot(x, med, marker="o", ms=3.3, color=BLUE, lw=1.15, zorder=3)
        ax.plot(x, worst, marker="^", ms=3.6, color=RUST, lw=1.05, zorder=3)
        ax.axhline(1., linestyle=(0,(4,2)), color=GRAY, linewidth=.75, zorder=2)
        ax.grid(axis="y", color=GRID, linewidth=.55)
        ax.grid(False, axis="x")
        ax.tick_params(axis="both", which="major", width=.6, length=2.5, labelsize=7.4)
        ax.tick_params(which="minor", length=0)
        ax.spines[["right","top"]].set_visible(False)
        ax.set_axisbelow(True)
    for ax in axes[:,0]:
        ax.set_ylabel(r"$R_A$ (%)")
    symbols = [Line2D([],[],color=color,marker=shape,ms=4,lw=1.1,label=name)
               for name,color,shape in (("Median",BLUE,"o"),("Worst of 12",RUST,"^"))]
    fig.legend(handles=symbols, loc="upper center", bbox_to_anchor=(.55,.973),
               ncol=2, frameon=False, handletextpad=.5, columnspacing=1.5)
    save(fig, "alignment_sensitivity", preview)


def read_json_followup(name):
    return json.loads((EVIDENCE / "followup" / name).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    geometry_figure(args.preview_dir)
    stages_figure(args.preview_dir)
    selection_figure(args.preview_dir)
    alignment_figure(args.preview_dir)


if __name__ == "__main__":
    main()
