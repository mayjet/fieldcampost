"""
camera_poses.json + terrain.ply → COLMAPフォーマット
COLMAP のインストール不要。Jetsonでも動作。

extrinsic_4x4 規約: cam→world (カメラ位置・姿勢を表す行列)
COLMAPは world→cam なので R_cw=R.T, t_cw=-R.T@t で変換する
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import trimesh


def export_colmap_format(
    poses_json:  Path,
    terrain_ply: Path,
    images_dir:  Path,
    output_dir:  Path,
    n_points3d:  int = 100_000,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(poses_json) as f:
        data = json.load(f)
    cameras = data["cameras"]

    _write_cameras_txt(cameras, output_dir / "cameras.txt")
    _write_images_txt(cameras, images_dir, output_dir / "images.txt")
    _write_points3d_txt(Path(terrain_ply), output_dir / "points3D.txt", n_points3d)

    for fname in ["cameras.txt", "images.txt", "points3D.txt"]:
        size = (output_dir / fname).stat().st_size
        print(f"  {fname}: {size:,} bytes")


def _cam_to_world_to_colmap(ext4x4: list):
    """
    cam→world 行列 → COLMAP (QW,QX,QY,QZ,TX,TY,TZ)
    COLMAP は world→cam なので逆変換する
    """
    E   = np.array(ext4x4, dtype=float)
    R_c2w = E[:3, :3]
    t_c2w = E[:3,  3]

    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t_c2w

    q    = Rotation.from_matrix(R_w2c).as_quat()  # [x,y,z,w]
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return float(qw), float(qx), float(qy), float(qz), \
           float(t_w2c[0]), float(t_w2c[1]), float(t_w2c[2])


def _write_cameras_txt(cameras: list, out: Path):
    lines = ["# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n"]
    for cam in cameras:
        i = cam["intrinsic"]
        cid = cam["id"] + 1
        lines.append(
            f"{cid} PINHOLE {i['width']} {i['height']} "
            f"{i['fx']:.6f} {i['fy']:.6f} {i['cx']:.6f} {i['cy']:.6f}\n"
        )
    out.write_text("".join(lines))


def _write_images_txt(cameras: list, images_dir: Path, out: Path):
    lines = ["# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n"]
    for cam in cameras:
        img_name = f"cam_{cam['id']:04d}.png"
        if images_dir.exists() and not (images_dir / img_name).exists():
            continue
        qw,qx,qy,qz,tx,ty,tz = _cam_to_world_to_colmap(cam["extrinsic_4x4"])
        iid = cam["id"] + 1
        lines.append(
            f"{iid} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{tx:.6f} {ty:.6f} {tz:.6f} {iid} {img_name}\n\n"
        )
    out.write_text("".join(lines))


def _write_points3d_txt(terrain_ply: Path, out: Path, n: int):
    if not terrain_ply.exists():
        out.write_text("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        print("WARNING: terrain.ply not found — points3D.txt is empty (random init)")
        return
    mesh   = trimesh.load(str(terrain_ply), force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    lines  = ["# POINT3D_ID X Y Z R G B ERROR TRACK[]\n"]
    for i, (x,y,z) in enumerate(pts):
        lines.append(f"{i+1} {x:.6f} {y:.6f} {z:.6f} 128 128 128 0.0\n")
    out.write_text("".join(lines))
    print(f"points3D.txt: {n:,} pts from terrain mesh (→ better 3DGS init)")
