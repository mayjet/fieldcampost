"""
terrain.ply → trimesh.Trimesh への変換。
PLYが点群の場合はConvex Hull / Poisson再構成でメッシュ化。
"""
import trimesh
import numpy as np
from pathlib import Path


def ply_to_obj(ply_path: str) -> Path:
    """PLY → OBJ 変換 (Genesis はOBJが安定なため)。既に変換済みならキャッシュを再利用する。"""
    ply_path = Path(ply_path)
    obj_path = ply_path.with_suffix(".obj")
    if obj_path.exists():
        return obj_path
    mesh = trimesh.load(str(ply_path), force='mesh')
    mesh.export(str(obj_path))
    return obj_path


def load_terrain_as_mesh(ply_path: str) -> tuple[trimesh.Trimesh, dict]:
    """
    PLYファイルを読み込みtrimesh.Trimeshとして返す。
    点群PLYの場合はPoisson再構成でメッシュ化する。

    Returns:
        mesh: trimesh.Trimesh
        info: dict with x_min, x_max, y_min, y_max, z_min, z_max, width, height
    """
    path = Path(ply_path)
    if not path.exists():
        raise FileNotFoundError(f"Terrain file not found: {ply_path}")

    loaded = trimesh.load(str(path), force='mesh')

    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
        mesh = loaded
    else:
        print(f"点群として読み込み → Poisson再構成中...")
        import open3d as o3d  # 点群再構成時のみ必要 (lazy import でメモリ節約)
        pcd = o3d.io.read_point_cloud(str(path))
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(100)

        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        density_arr = np.asarray(densities)
        keep = density_arr > np.percentile(density_arr, 10)
        mesh_o3d = mesh_o3d.select_by_index(
            np.where(keep)[0]
        ) if keep.sum() > 0 else mesh_o3d

        verts = np.asarray(mesh_o3d.vertices)
        faces = np.asarray(mesh_o3d.triangles)
        mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        print(f"再構成完了: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    verts = np.array(mesh.vertices)
    info = {
        "x_min": verts[:,0].min(), "x_max": verts[:,0].max(),
        "y_min": verts[:,1].min(), "y_max": verts[:,1].max(),
        "z_min": verts[:,2].min(), "z_max": verts[:,2].max(),
        "width":  verts[:,0].max() - verts[:,0].min(),
        "height": verts[:,1].max() - verts[:,1].min(),
    }
    return mesh, info


def mesh_to_pointcloud(mesh: trimesh.Trimesh, n_points: int = 1_000_000,
                        return_numpy: bool = False):
    """
    trimesh メッシュ表面から均一サンプリングして点群を返す。
    open3d 不要 (trimesh のみで完結、メモリ節約のため lazy import もしない)。

    Args:
        return_numpy: True の場合 (N,3) ndarray を返す。
                      False の場合 open3d.geometry.PointCloud を返す (要 open3d)。
    """
    pts, _ = trimesh.sample.sample_surface(mesh, n_points)

    if return_numpy:
        return np.asarray(pts)

    import open3d as o3d  # PointCloud オブジェクトが必要な場合のみ lazy import
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd
