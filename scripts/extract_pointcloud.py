"""
3DGS .ply から Gaussian 中心を不透明度フィルタで抽出して点群として保存
"""
import numpy as np
from plyfile import PlyData, PlyElement
from pathlib import Path


def extract_pointcloud(model_dir: Path, output_ply: Path,
                       opacity_threshold: float = 0.1):
    ply_path = _find_latest_ply(model_dir)
    plydata  = PlyData.read(str(ply_path))
    v        = plydata['vertex']

    xyz     = np.stack([v['x'], v['y'], v['z']], axis=1)
    opacity = 1.0 / (1.0 + np.exp(-v['opacity']))
    mask    = opacity > opacity_threshold
    fxyz    = xyz[mask]

    print(f"Gaussians: {len(xyz):,} → filtered: {len(fxyz):,} "
          f"(opacity>{opacity_threshold})")

    el = PlyElement.describe(
        np.array([tuple(p) for p in fxyz],
                 dtype=[('x','f4'),('y','f4'),('z','f4')]),
        'vertex'
    )
    PlyData([el]).write(str(output_ply))
    print(f"Saved: {output_ply}")


def _find_latest_ply(model_dir: Path) -> Path:
    candidates = sorted(
        (model_dir / "point_cloud").glob("iteration_*/point_cloud.ply")
    )
    if not candidates:
        raise FileNotFoundError(f"No point_cloud.ply in {model_dir}")
    return candidates[-1]
