"""
生成済みケース (cases/{case}/) の地形・森林(tree.glb)・観測ポール/カメラ・
建物・農場境界をまとめてGenesisビューアで閲覧する。

閲覧専用 (カメラ移動やPTZ制御などのシミュレーションは行わない)。
マウスドラッグで視点回転、スクロールでズーム。ウィンドウを閉じるかCtrl+Cで終了。

使い方:
  python view_scene.py --case ranch_01
  python view_scene.py --case ranch_01 --real-trees --n-trees 2000
  python view_scene.py --case ranch_01 --no-trees --no-frustums
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import genesis as gs

from scripts.genesis_renderer import resolve_backend, BACKENDS
from scripts.terrain_loader import load_terrain_as_mesh, ply_to_obj, ground_z_at
from scripts.scene_populate import load_farm_boundary, sample_tree_positions, place_tree_entities

COLOR_POLE       = (0.45, 0.35, 0.25, 1.0)
COLOR_BUILDING   = (0.55, 0.35, 0.30, 1.0)
COLOR_FIXED_CAM  = (0.2, 0.4, 1.0, 1.0)
COLOR_PTZ_CAM    = (1.0, 0.3, 0.1, 1.0)
COLOR_BOUNDARY   = (0.2, 1.0, 0.2, 1.0)
COLOR_FOREST     = (0.1, 0.6, 0.1, 0.8)


def _circle_points(center, radius, z, n=28):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=True)
    return [
        (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a), z)
        for a in angles
    ]


def _draw_polyline(scene, points, radius, color):
    for a, b in zip(points[:-1], points[1:]):
        scene.draw_debug_line(start=a, end=b, radius=radius, color=color)


def _camera_marker_offset(cam: dict, offset: float = 0.25):
    """カメラマーカーをポール軸から水平にずらす方向。lookat方向を優先し、
    水平方向差がほぼ無いPTZ直下視の場合のみ azimuth_deg にフォールバックする。"""
    pos  = np.array(cam["position_enu"], dtype=float)
    look = np.array(cam["lookat_enu"], dtype=float)
    horiz = look[:2] - pos[:2]
    norm = np.linalg.norm(horiz)
    if norm > 1e-3:
        dir_xy = horiz / norm
    else:
        az = math.radians(cam["azimuth_deg"])
        dir_xy = np.array([math.sin(az), math.cos(az)])
    return pos + np.array([dir_xy[0] * offset, dir_xy[1] * offset, -0.05])


def build_scene(case: str, device: str, n_trees: int, real_trees: bool,
                 tree_glb: str, show_trees: bool, show_frustums: bool,
                 show_buildings: bool, show_boundary: bool, seed: int):
    case_dir = Path("cases") / case
    terrain_ply = case_dir / "terrain.ply"
    poses_json  = case_dir / "camera_poses.json"
    if not terrain_ply.exists():
        raise FileNotFoundError(f"Terrain file not found: {terrain_ply}")
    if not poses_json.exists():
        raise FileNotFoundError(f"Camera poses file not found: {poses_json}")

    mesh, info = load_terrain_as_mesh(str(terrain_ply))
    verts = np.array(mesh.vertices)
    center_x = (info["x_min"] + info["x_max"]) / 2
    center_y = (info["y_min"] + info["y_max"]) / 2
    center_z = (info["z_min"] + info["z_max"]) / 2
    span = max(info["width"], info["height"])

    camera_pos    = (center_x, center_y - span * 0.8, info["z_max"] + span * 0.3)
    camera_lookat = (center_x, center_y, center_z)

    gs.init(backend=resolve_backend(device))
    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            res=(1280, 720),
            camera_pos=camera_pos,
            camera_lookat=camera_lookat,
            camera_fov=40,
        ),
    )

    terrain_obj = ply_to_obj(terrain_ply)
    scene.add_entity(gs.morphs.Mesh(file=str(terrain_obj), fixed=True))

    poses = json.loads(poses_json.read_text())
    poles   = poses.get("poles", [])
    cameras = poses.get("cameras", [])
    farm = load_farm_boundary(case_dir / "farm_boundary.json")

    if show_trees:
        forests = farm.get("forests") if farm else None
        positions = sample_tree_positions(verts, forests, n_trees=n_trees, seed=seed)
        rng = np.random.default_rng(seed)
        glb_path = Path(tree_glb) if tree_glb else None
        n = place_tree_entities(scene, glb_path, positions, rng, use_proxy=not real_trees)
        print(f"木を配置: {n} 本 "
              f"({'森林パッチ限定' if forests else '地形全体ランダム'}, "
              f"{'tree.glb' if real_trees else 'プロキシ'})")

    for pole in poles:
        base = pole["position_ground"]
        h    = pole["height"]
        scene.add_entity(
            gs.morphs.Cylinder(
                pos=(base[0], base[1], base[2] + h / 2),
                height=h, radius=0.07, fixed=True,
            ),
            surface=gs.surfaces.Default(color=COLOR_POLE),
        )

    if show_buildings and farm:
        for b in farm.get("buildings", []):
            cx, cy = b["center"]
            w, d, h = b["size_wdh"]
            gz = ground_z_at(verts, (cx, cy))
            scene.add_entity(
                gs.morphs.Box(
                    pos=(cx, cy, gz + h / 2),
                    size=(w, d, h),
                    euler=(0, 0, b["rot_deg"]),
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=COLOR_BUILDING),
            )

    gs_cameras = []
    for cam in cameras:
        pos, lookat = cam["position_enu"], cam["lookat_enu"]
        fov_h, fov_v = cam["fov_h"], cam["fov_v"]
        aspect = math.tan(math.radians(fov_h / 2)) / math.tan(math.radians(fov_v / 2))
        gs_cam = scene.add_camera(
            res=(max(round(100 * aspect), 1), 100),
            pos=pos, lookat=lookat, up=(0, 0, 1),
            fov=fov_v, near=0.1, far=150.0,
            GUI=False,
        )
        gs_cameras.append((cam, gs_cam))

    scene.build()

    if show_boundary and farm and farm.get("boundary"):
        pts = [(x, y, ground_z_at(verts, (x, y))) for x, y in farm["boundary"]]
        _draw_polyline(scene, pts, radius=0.05, color=COLOR_BOUNDARY)
        for f in farm.get("forests", []):
            z = ground_z_at(verts, f["center"])
            circle = _circle_points(f["center"], f["radius"], z)
            _draw_polyline(scene, circle, radius=0.04, color=COLOR_FOREST)

    for cam, gs_cam in gs_cameras:
        color = COLOR_FIXED_CAM if cam["type"] == "fixed" else COLOR_PTZ_CAM
        marker_pos = _camera_marker_offset(cam)
        scene.draw_debug_sphere(pos=marker_pos, radius=0.09, color=color)
        if show_frustums:
            scene.draw_debug_frustum(gs_cam, color=(*color[:3], 0.25))

    return scene


def main():
    parser = argparse.ArgumentParser(
        description="生成済みケースをGenesisビューアで一括閲覧する"
                     " (地形+森林+ポール/カメラ+建物+農場境界)")
    parser.add_argument("--case",   required=True, help="ケース名 (cases/ 以下のフォルダ名)")
    parser.add_argument("--device", default="auto", choices=BACKENDS,
                         help="Genesisバックエンド (auto=CUDA/Metal/CPUを自動選択)")
    parser.add_argument("--no-trees", action="store_true", help="森林の描画を省略する")
    parser.add_argument("--n-trees", type=int, default=400,
                         help="配置する木の本数 (既定400。学習用は2000本)")
    parser.add_argument("--real-trees", action="store_true",
                         help="軽量プロキシではなく実際のtree.glbを使う (重い)")
    parser.add_argument("--tree-glb", default="assets/tree.glb",
                         help="tree.glbのパス")
    parser.add_argument("--no-frustums", action="store_true", help="カメラFOVフラスタムを省略する")
    parser.add_argument("--no-buildings", action="store_true", help="建物の描画を省略する")
    parser.add_argument("--no-boundary", action="store_true",
                         help="農場境界・森林パッチ輪郭線の描画を省略する")
    parser.add_argument("--seed", type=int, default=42, help="木配置の乱数シード")
    args = parser.parse_args()

    scene = build_scene(
        case=args.case, device=args.device,
        n_trees=args.n_trees, real_trees=args.real_trees, tree_glb=args.tree_glb,
        show_trees=not args.no_trees, show_frustums=not args.no_frustums,
        show_buildings=not args.no_buildings, show_boundary=not args.no_boundary,
        seed=args.seed,
    )

    print(f"シーンを表示中: cases/{args.case} "
          f"(マウスドラッグで回転、スクロールでズーム、ウィンドウを閉じるかCtrl+Cで終了)")
    try:
        while scene.viewer.is_alive():
            scene.step()
    except KeyboardInterrupt:
        scene.viewer.stop()


if __name__ == "__main__":
    main()
