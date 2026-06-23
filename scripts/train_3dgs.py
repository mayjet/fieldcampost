"""
3DGSトレーニング: gsplat コアAPI (rasterization) のみを使った最小構成トレーナー。

gsplat公式の examples/simple_trainer.py は PyPI配布に含まれず、別途
GitHubから取得する必要があり、しかも viser / nerfview (fork) / pycolmap (fork) /
fused_ssim / fused_bilagrid といったCUDAビルドを要する重い依存関係を要求する。
ここではそれらを避け、COLMAPテキスト形式 (cameras.txt/images.txt/points3D.txt)
を自前で読み込み、L1損失のみで学習するシンプルな実装にする。
densification（学習中の点の追加・削除）は行わず、points3D.txt の初期点群を
そのまま最適化する。
"""
import numpy as np
import torch
import imageio.v2 as imageio
from pathlib import Path
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def train_3dgs(colmap_dir: Path, output_dir: Path, iterations: int = 30000,
                lr: float = 1e-2, log_every: int = 500):
    colmap_dir = Path(colmap_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sparse_dir = colmap_dir / "sparse" / "0"
    images_dir = colmap_dir / "images"
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    cams   = _read_cameras_txt(sparse_dir / "cameras.txt")
    images = _read_images_txt(sparse_dir / "images.txt")
    xyz, rgb = _read_points3d_txt(sparse_dir / "points3D.txt")

    if not images:
        raise RuntimeError(f"images.txt に画像が1件もありません: {sparse_dir / 'images.txt'}")

    # ── カメラ姿勢・GT画像をロード ────────────────────────────────────────────
    viewmats, Ks, gt_images = [], [], []
    width = height = None
    for img in images:
        img_path = images_dir / img["name"]
        if not img_path.exists():
            continue
        cam = cams[img["camera_id"]]
        width, height = cam["width"], cam["height"]
        viewmats.append(_build_viewmat(img))
        Ks.append(_build_K(cam))
        gt = imageio.imread(str(img_path)).astype(np.float32) / 255.0
        if gt.shape[-1] == 4:
            gt = gt[..., :3]
        gt_images.append(gt)

    if not gt_images:
        raise RuntimeError(f"レンダリング画像が見つかりません: {images_dir}")

    viewmats  = torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device)
    Ks        = torch.tensor(np.stack(Ks), dtype=torch.float32, device=device)
    gt_images = torch.tensor(np.stack(gt_images), dtype=torch.float32, device=device)

    # ── Gaussian パラメータ初期化 (points3D.txt の点群をそのまま使う) ────────────
    n = xyz.shape[0]
    bbox_extent = float(np.linalg.norm(xyz.max(0) - xyz.min(0)))
    init_scale  = max(bbox_extent / (n ** (1 / 3)), 1e-3)

    means     = torch.tensor(xyz, dtype=torch.float32, device=device, requires_grad=True)
    scales    = torch.full((n, 3), float(np.log(init_scale)),
                            dtype=torch.float32, device=device, requires_grad=True)
    quats0    = torch.zeros((n, 4), dtype=torch.float32, device=device)
    quats0[:, 0] = 1.0
    quats     = quats0.clone().requires_grad_(True)
    opacities = torch.full((n,), _logit(0.1), dtype=torch.float32,
                            device=device, requires_grad=True)
    colors    = torch.tensor(rgb, dtype=torch.float32, device=device, requires_grad=True)

    optimizer = torch.optim.Adam([means, scales, quats, opacities, colors], lr=lr)

    from gsplat.rendering import rasterization

    n_cams = viewmats.shape[0]
    print(f"3DGSトレーニング開始: {n:,} Gaussians, {n_cams} カメラ, {iterations} iter "
          f"(device={device})")
    for step in range(1, iterations + 1):
        idx = np.random.randint(n_cams)
        rendered, _, _ = rasterization(
            means=means,
            quats=quats / quats.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales),
            opacities=torch.sigmoid(opacities),
            colors=torch.sigmoid(colors),
            viewmats=viewmats[idx:idx + 1],
            Ks=Ks[idx:idx + 1],
            width=width, height=height,
            sh_degree=None,
        )
        loss = torch.abs(rendered[0] - gt_images[idx]).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0 or step == iterations:
            print(f"  iter {step}/{iterations}  loss={loss.item():.4f}")

    out_dir = output_dir / "point_cloud" / f"iteration_{iterations}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_ply(out_dir / "point_cloud.ply", means, opacities)
    print(f"Training done: {out_dir}/point_cloud.ply")


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _read_cameras_txt(path: Path) -> dict:
    cams = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        cid = int(parts[0])
        width, height = int(parts[2]), int(parts[3])
        fx, fy, cx, cy = map(float, parts[4:8])
        cams[cid] = dict(width=width, height=height, fx=fx, fy=fy, cx=cx, cy=cy)
    return cams


def _read_images_txt(path: Path) -> list:
    images = []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz     = map(float, parts[5:8])
        images.append(dict(
            qw=qw, qx=qx, qy=qy, qz=qz, tx=tx, ty=ty, tz=tz,
            camera_id=int(parts[8]), name=parts[9],
        ))
    return images


def _read_points3d_txt(path: Path):
    xyz, rgb = [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([float(parts[4]), float(parts[5]), float(parts[6])])
    if not xyz:
        raise RuntimeError(f"points3D.txt に点が1つもありません: {path}")
    return np.array(xyz, dtype=np.float32), np.array(rgb, dtype=np.float32) / 255.0


def _build_viewmat(img: dict) -> np.ndarray:
    """COLMAPの world→cam quaternion/translation から4x4 viewmatを構築"""
    R = Rotation.from_quat([img["qx"], img["qy"], img["qz"], img["qw"]]).as_matrix()
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R
    M[:3, 3]  = [img["tx"], img["ty"], img["tz"]]
    return M


def _build_K(cam: dict) -> np.ndarray:
    K = np.eye(3, dtype=np.float32)
    K[0, 0], K[1, 1] = cam["fx"], cam["fy"]
    K[0, 2], K[1, 2] = cam["cx"], cam["cy"]
    return K


def _save_ply(path: Path, means: torch.Tensor, opacities: torch.Tensor):
    """extract_pointcloud.py が読む point_cloud.ply 形式 (x,y,z,opacity) で保存"""
    xyz_np = means.detach().cpu().numpy()
    op_np  = opacities.detach().cpu().numpy()
    verts  = np.array(
        [tuple(p) + (o,) for p, o in zip(xyz_np, op_np)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('opacity', 'f4')],
    )
    el = PlyElement.describe(verts, 'vertex')
    PlyData([el]).write(str(path))
