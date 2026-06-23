"""
Genesis でカメラ画像を一括レンダリング。
地形PLY + tree.glb(オプション) + camera_poses.json → images/*.png
"""
import genesis as gs
import json, math, imageio
import numpy as np
from pathlib import Path


class GenesisRenderer:
    def __init__(self, terrain_ply, tree_glb, poses_json, output_dir,
                 image_w=1920, image_h=1080):
        self.terrain_ply = Path(terrain_ply)
        self.tree_glb    = Path(tree_glb) if tree_glb else None
        self.poses_json  = Path(poses_json)
        self.output_dir  = Path(output_dir)
        self.image_w     = image_w
        self.image_h     = image_h

    def render_all(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)

        gs.init(backend=gs.cuda)
        scene = gs.Scene(show_viewer=False)

        terrain_obj = self._ply_to_obj(self.terrain_ply)
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
                pos=pos, fov=fov_v,
            )
            cameras.append((cam["id"], gs_cam))

        scene.build()

        rendered = 0
        total    = len(cameras)
        for cam_id, gs_cam in cameras:
            out_path = self.output_dir / f"cam_{cam_id:04d}.png"
            if out_path.exists():
                continue
            rgb, *_ = gs_cam.render(rgb=True)  # Genesis 1.1.2: タプル (rgb, depth, seg, normal) を返す
            rgb = np.asarray(rgb)
            if np.issubdtype(rgb.dtype, np.floating):
                rgb = (rgb * 255).astype("uint8")
            else:
                rgb = rgb.astype("uint8")
            imageio.imwrite(str(out_path), rgb)
            rendered += 1
            if rendered % 10 == 0:
                print(f"  {rendered}/{total} rendered")

        print(f"Rendering done: {rendered} new images → {self.output_dir}/")

    def _ply_to_obj(self, ply_path: Path) -> Path:
        """PLY → OBJ 変換 (Genesis はOBJが安定)"""
        import trimesh
        obj_path = ply_path.with_suffix(".obj")
        if obj_path.exists():
            return obj_path
        mesh = trimesh.load(str(ply_path), force='mesh')
        mesh.export(str(obj_path))
        return obj_path

    def _place_trees(self, scene, n_trees: int = 2000, seed: int = 42):
        """
        tree.glb を地形全体にランダム大量配置する (森林想定)。
        配置制限は一切なし — 地形頂点からランダムサンプリングして均一に散布する。
        """
        import trimesh
        rng   = np.random.default_rng(seed)
        mesh  = trimesh.load(str(self._ply_to_obj(self.terrain_ply)), force='mesh')
        verts = np.array(mesh.vertices)

        indices = rng.integers(0, len(verts), size=n_trees)
        print(f"  木を配置中: {n_trees} 本 (地形全体ランダム)...")

        for i, idx in enumerate(indices):
            v     = verts[idx]
            pos   = [float(v[0]), float(v[1]), float(v[2]) + 0.2]
            scale = float(rng.uniform(0.6, 3.0))
            scene.add_entity(
                gs.morphs.Mesh(
                    file=str(self.tree_glb),
                    fixed=True,
                    pos=pos,
                    scale=scale,
                ),
            )
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{n_trees} 本配置済み")
