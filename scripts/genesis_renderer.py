"""
Genesis でカメラ画像を一括レンダリング。
地形PLY + tree.glb(オプション) + camera_poses.json → images/*.png
"""
import genesis as gs
import json, math, imageio
import numpy as np
from pathlib import Path

from scripts.terrain_loader import ply_to_obj
from scripts.scene_populate import load_farm_boundary, sample_tree_positions, place_tree_entities

BACKENDS = ("auto", "gpu", "cuda", "metal", "amdgpu", "cpu")


def resolve_backend(device: str = "auto"):
    """device名 → Genesisバックエンド定数。'auto'/'gpu' はGenesis自身に
    cuda→amdgpu→metal→cpu の順で自動選択させる (backend=None と等価)。"""
    device = (device or "auto").lower()
    if device not in BACKENDS:
        raise ValueError(f"Unknown Genesis backend '{device}'. Choose from: {', '.join(BACKENDS)}")
    if device in ("auto", "gpu"):
        return None
    return getattr(gs, device)


class GenesisRenderer:
    def __init__(self, terrain_ply, tree_glb, poses_json, output_dir,
                 image_w=1920, image_h=1080, device="auto"):
        self.terrain_ply = Path(terrain_ply)
        self.tree_glb    = Path(tree_glb) if tree_glb else None
        self.poses_json  = Path(poses_json)
        self.output_dir  = Path(output_dir)
        self.image_w     = image_w
        self.image_h     = image_h
        self.device      = device

    def render_all(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        gs.init(backend=resolve_backend(self.device))
        scene = gs.Scene(show_viewer=False)

        terrain_obj = ply_to_obj(self.terrain_ply)
        scene.add_entity(gs.morphs.Mesh(file=str(terrain_obj), fixed=True))

        if self.tree_glb and self.tree_glb.exists():
            self._place_trees(scene)

        with open(self.poses_json) as f:
            poses = json.load(f)["cameras"]

        cameras = []
        for cam in poses:
            ext  = np.array(cam["extrinsic_4x4"])
            pos  = ext[:3, 3].tolist()
            intr = cam["intrinsic"]
            fov_v = 2 * math.degrees(math.atan(intr["cy"] / intr["fy"]))
            gs_cam = scene.add_camera(
                res=(self.image_w, self.image_h),
                pos=pos, lookat=cam["lookat_enu"], up=(0, 0, 1),
                fov=fov_v,
            )
            cameras.append((cam["id"], gs_cam))

        scene.build()

        rendered = 0
        total    = len(cameras)
        for cam_id, gs_cam in cameras:
            out_path = self.output_dir / f"cam_{cam_id:04d}.png"
            if out_path.exists():
                continue
            rgb = gs_cam.render(rgb=True)["rgb"]
            imageio.imwrite(str(out_path), (rgb * 255).astype("uint8"))
            rendered += 1
            if rendered % 10 == 0:
                print(f"  {rendered}/{total} rendered")

        print(f"Rendering done: {rendered} new images → {self.output_dir}/")

    def _place_trees(self, scene, n_trees: int = 2000, seed: int = 42):
        """
        tree.glb を大量配置する。farm_boundary.json があれば forests(森林パッチ)
        内に限定して配置し、無ければ従来どおり地形全体からランダムに配置する
        (後方互換)。
        """
        import trimesh
        mesh  = trimesh.load(str(ply_to_obj(self.terrain_ply)), force='mesh')
        verts = np.array(mesh.vertices)

        farm    = load_farm_boundary(self.terrain_ply.parent / "farm_boundary.json")
        forests = farm.get("forests") if farm else None

        positions = sample_tree_positions(verts, forests, n_trees=n_trees, seed=seed)
        rng = np.random.default_rng(seed)
        print(f"  木を配置中: {len(positions)} 本 "
              f"({'森林パッチ限定' if forests else '地形全体ランダム'})...")
        n = place_tree_entities(scene, self.tree_glb, positions, rng, use_proxy=False)
        print(f"    {n} 本配置済み")
