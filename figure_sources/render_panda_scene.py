
"""Render the recorded Panda configurations for Figure 1(a) in MuJoCo.

Use the Panda model directory from the pinned CaDeLaC revision in the README.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
from xml.etree import ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evidence/followup"))
from panda_engine import add_payload, source_xml

OVERVIEW_SIZE = (1260, 1000)
WRIST_SIZE = 660


def make_view_model(loaded_xml, model_dir):
    """Attach the upstream display meshes to the archived dynamics model."""
    source = ET.parse(model_dir / "panda_no_collision.xml").getroot()
    root = ET.fromstring(loaded_xml)
    root.find("compiler").attrib.pop("meshdir")
    world = root.find("worldbody")
    used_meshes = set()
    visual_class = root.find(".//default[@class='visual']")
    ET.SubElement(visual_class, "geom", type="mesh", contype="0", conaffinity="0", group="2")
    for body in world.iter("body"):
        upstream = source.find(f".//body[@name='{body.get('name')}']")
        if upstream is None:
            continue
        for geom in upstream.findall("geom[@class='visual']"):
            body.append(deepcopy(geom))
            used_meshes.add(geom.attrib["mesh"])

    assets = {}
    asset = root.find("asset")
    for mesh in source.findall("asset/mesh"):
        file = mesh.attrib["file"]
        if mesh.get("name", Path(file).stem) not in used_meshes:
            continue
        assets[file] = (model_dir / "assets" / file).read_bytes()
        asset.append(deepcopy(mesh))

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(OVERVIEW_SIZE[0]), offheight=str(OVERVIEW_SIZE[1]))
    ET.SubElement(visual, "quality", offsamples="8", shadowsize="2048")
    ET.SubElement(visual, "headlight", active="0")
    ET.SubElement(asset, "texture", type="skybox", builtin="flat",
                  rgb1=".97 .979 .988", rgb2=".97 .979 .988", width="32", height="32")
    ET.SubElement(asset, "material", name="paper_floor", rgba=".93 .95 .97 1",
                  specular="0", shininess="0", emission=".26")
    ET.SubElement(world, "geom", name="paper_floor", type="plane", pos="0 0 -.001",
                  size="3 3 .1", material="paper_floor", contype="0", conaffinity="0")
    ET.SubElement(world, "light", name="paper_key", directional="true", dir=".34 .45 -.83",
                  ambient=".20 .20 .20", diffuse=".54 .54 .54", specular=".14 .14 .14")
    ET.SubElement(world, "light", name="paper_fill", directional="true", dir="-.55 -.5 -.56",
                  ambient=".17 .17 .17", diffuse=".12 .12 .12", specular=".05 .05 .05",
                  castshadow="false")
    ET.SubElement(world, "camera", name="paper_view", pos="2 2 2",
                  orthographic="true", fovy="1")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), assets)
    model.opt.disableflags |= (
        int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    )
    return model


def camera_axes(direction):
    eye = direction / np.linalg.norm(direction)
    right = np.cross([0, 0, 1], eye)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(eye, right), eye))


def set_camera(model, cam, look_at, axes, span):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, axes.ravel())
    model.cam_quat[cam] = quat
    model.cam_pos[cam] = look_at + 1.45 * axes[:, 2]
    model.cam_fovy[cam] = span


def world_mesh_vertices(model, data):
    vertices = []
    for geom in np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_MESH):
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        points = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
        vertices.append(points @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    return np.concatenate(vertices)


def project(renderer, points):
    """Map positions to lower-left normalized image coordinates."""
    cam = renderer.scene.camera[0]
    assert cam.orthographic
    height = cam.frustum_top - cam.frustum_bottom
    width = height * renderer.width / renderer.height
    relative = points - cam.pos
    right = np.cross(cam.forward, cam.up)
    return np.column_stack((
        .5 + (relative @ right - cam.frustum_center) / width,
        (relative @ cam.up - cam.frustum_bottom) / height,
    ))


def render_views(model_dir, output):
    archive = ROOT / "evidence/directional/mujoco_terminal_check/fold0_probes.npz"
    with np.load(archive, allow_pickle=False) as bundle:
        q = bundle["q"]
        qd = bundle["qd"]
        acceleration = bundle["applied_directions"][:, 0]
        torque = bundle["torques"][:, 1, 0, 0]
        loaded_xml = add_payload(source_xml(), bundle["inertias"][0])
    np.testing.assert_array_equal(q[0, :-1], q[1, :-1])
    np.testing.assert_allclose(q[1, -1] - q[0, -1], .5, atol=1e-12, rtol=0)
    model = make_view_model(loaded_xml, model_dir)
    data = mujoco.MjData(model)
    cam = model.camera("paper_view").id
    for pose, velocity, trace in zip(q, qd, torque, strict=True):
        data.qpos[:] = pose
        data.qvel[:] = velocity
        data.qacc[:] = acceleration
        mujoco.mj_inverse(model, data)
        np.testing.assert_allclose(data.qfrc_inverse, trace, rtol=2e-10, atol=2e-10)

    data.qpos[:] = q[0]
    mujoco.mj_forward(model, data)
    azimuth, elevation = np.deg2rad([-81.0, 20.0])
    overview_axes = camera_axes(np.array([
        np.cos(azimuth) * np.cos(elevation),
        np.sin(azimuth) * np.cos(elevation),
        np.sin(elevation),
    ]))
    projected = world_mesh_vertices(model, data) @ overview_axes
    center = .5 * (projected.min(axis=0) + projected.max(axis=0))
    width, height = OVERVIEW_SIZE
    extent = np.ptp(projected, axis=0)
    span = 1.27 * max(extent[1], extent[0] * height / width)
    set_camera(model, cam, overview_axes @ center, overview_axes, span)
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(model, width=width, height=height) as renderer:
        renderer.update_scene(data, camera="paper_view")
        overview = renderer.render()
        joint7 = project(renderer, data.body("link7").xpos[None])[0]

    tool = data.body("end_effector")
    position = tool.xpos.copy()
    x, y, z = tool.xmat.copy().reshape(3, 3).T
    wrist_axes = camera_axes(z - .44 * x + .23 * y)
    set_camera(model, cam, position + .02 * z, wrist_axes, .242)
    wrist_views, wrist_marks = [], []
    with mujoco.Renderer(model, width=WRIST_SIZE, height=WRIST_SIZE) as renderer:
        for pose in q:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="paper_view")
            wrist_views.append(renderer.render())
            x_axis = tool.xmat.reshape(3, 3)[:, 0]
            points = np.stack((
                tool.xpos,
                tool.xpos + .065 * x_axis,
                data.body("calibration_payload").xipos,
            ))
            wrist_marks.append(project(renderer, points))
    wrist_marks = np.stack(wrist_marks)
    np.testing.assert_allclose(wrist_marks[0, 0], wrist_marks[1, 0], atol=1e-6)
    assert np.all((wrist_marks > 0) & (wrist_marks < 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, overview=overview, joint7=joint7,
                        wrists=np.stack(wrist_views), wrist_marks=wrist_marks)
    print("Rendered the archived pose pair and verified both inverse-dynamics reads.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panda_model_dir", type=Path, help="CaDeLaC's cadelac/robots/panda directory")
    parser.add_argument("--output", type=Path, default=ROOT / "figure_sources/panda_mujoco_views.npz")
    args = parser.parse_args()
    render_views(args.panda_model_dir, args.output)


if __name__ == "__main__":
    main()

