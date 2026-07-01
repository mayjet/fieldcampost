"""
生成済み地形 (cases/{case}/terrain.ply) をGenesisのインタラクティブビューアで閲覧する。
マウスドラッグで視点回転、スクロールでズーム。ウィンドウを閉じるかCtrl+Cで終了。

使い方:
  python view_terrain.py --case ranch01
  python view_terrain.py --case ranch01 --device cpu
"""
import argparse
from pathlib import Path

import genesis as gs

from scripts.genesis_renderer import resolve_backend, BACKENDS
from scripts.terrain_loader import load_terrain_as_mesh, ply_to_obj


def view_terrain(case: str, device: str = "auto"):
    terrain_ply = Path("cases") / case / "terrain.ply"
    if not terrain_ply.exists():
        raise FileNotFoundError(f"Terrain file not found: {terrain_ply}")

    _, info = load_terrain_as_mesh(str(terrain_ply))
    center_x = (info["x_min"] + info["x_max"]) / 2
    center_y = (info["y_min"] + info["y_max"]) / 2
    center_z = (info["z_min"] + info["z_max"]) / 2
    span = max(info["width"], info["height"])

    camera_pos = (center_x, center_y - span * 0.8, info["z_max"] + span * 0.3)
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

    scene.build()

    print(f"地形を表示中: {terrain_ply} (マウスドラッグで回転、スクロールでズーム、ウィンドウを閉じるかCtrl+Cで終了)")
    try:
        while scene.viewer.is_alive():
            scene.step()
    except KeyboardInterrupt:
        scene.viewer.stop()


def main():
    parser = argparse.ArgumentParser(description="生成済み地形をGenesisビューアで閲覧する")
    parser.add_argument("--case",   required=True, help="ケース名 (cases/ 以下のフォルダ名)")
    parser.add_argument("--device", default="auto", choices=BACKENDS,
                         help="Genesisバックエンド (auto=CUDA/Metal/CPUを自動選択)")
    args = parser.parse_args()

    view_terrain(case=args.case, device=args.device)


if __name__ == "__main__":
    main()
