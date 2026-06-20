"""
GT点群 vs 3DGS再構成点群の精度評価。
Chamfer Distance / DSM RMSE / F-Score の3指標。
自動精度改善: 誤差の大きい領域を特定してカメラ追加候補を生成。
"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

AUTO_IMPROVE_THRESHOLD_CM = 15.0


def evaluate_accuracy(gt_ply: Path, recon_ply: Path,
                      case_dir: Path) -> dict:
    gt    = o3d.io.read_point_cloud(str(gt_ply))
    recon = o3d.io.read_point_cloud(str(recon_ply))
    print(f"GT={len(gt.points):,}  Recon={len(recon.points):,}")

    gt_ds, recon_ds = _align(gt, recon, voxel=0.10)

    cd, d_a2b, d_b2a = _chamfer_distance(gt_ds, recon_ds)
    rmse, mae, diff  = _dsm_rmse(gt, recon, grid=0.50)

    results = {
        "chamfer_distance_cm": round(cd*100, 3),
        "dsm_rmse_cm":         round(rmse*100, 3),
        "dsm_mae_cm":          round(mae*100, 3),
    }
    for thr in [0.05, 0.10, 0.20]:
        f, p, r = _fscore(d_a2b, d_b2a, thr)
        results[f"fscore_{int(thr*100)}cm"] = round(f, 4)
        print(f"F-Score@{thr*100:.0f}cm  F={f:.4f}  P={p:.4f}  R={r:.4f}")

    print(f"Chamfer Distance : {results['chamfer_distance_cm']:.2f} cm")
    print(f"DSM RMSE         : {results['dsm_rmse_cm']:.2f} cm")

    out_json = case_dir / "evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    _save_error_map(gt_ds, d_a2b, case_dir)

    return results


def add_cameras_to_error_regions(
    case_dir: Path, poses_json: Path,
    terrain_ply: Path, error_map_ply: Path,
) -> int:
    """
    誤差の大きい領域を特定し、そこをカバーするカメラを追加。
    追加したカメラ数を返す。
    """
    if not error_map_ply.exists():
        return 0

    error_pcd = o3d.io.read_point_cloud(str(error_map_ply))
    if len(error_pcd.points) == 0:
        return 0

    error_pts  = np.asarray(error_pcd.points)
    centroid   = error_pts.mean(axis=0)

    import trimesh
    mesh = trimesh.load(str(terrain_ply), force='mesh')
    pole_h = 8.0

    new_cam_pos = centroid.copy()
    new_cam_pos[2] += pole_h + 5.0
    lookat = centroid.copy()
    lookat[2] -= 10.0

    with open(poses_json) as f:
        data = json.load(f)

    max_id  = max(c["id"] for c in data["cameras"]) + 1
    intr    = data["cameras"][0]["intrinsic"]

    z_cam = new_cam_pos - lookat
    z_cam /= np.linalg.norm(z_cam)
    up    = np.array([0.0, 0.0, 1.0])
    x_cam = np.cross(up, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    ext   = np.eye(4)
    ext[:3,0]=x_cam; ext[:3,1]=y_cam; ext[:3,2]=z_cam; ext[:3,3]=new_cam_pos

    new_cam = {
        "id": max_id, "type": "fixed",
        "extrinsic_4x4": ext.tolist(),
        "intrinsic": intr,
        "position_enu": new_cam_pos.tolist(),
        "lookat_enu":   lookat.tolist(),
        "zone": 3,
    }
    data["cameras"].append(new_cam)
    with open(poses_json, "w") as f:
        json.dump(data, f, indent=2)

    return 1


# ── 内部関数 ──────────────────────────────────────────────────────────────────

def _align(gt, recon, voxel):
    gt_ds    = gt.voxel_down_sample(voxel)
    recon_ds = recon.voxel_down_sample(voxel)
    reg = o3d.pipelines.registration.registration_icp(
        recon_ds, gt_ds, max_correspondence_distance=2.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )
    recon_ds.transform(reg.transformation)
    recon.transform(reg.transformation)
    print(f"ICP  fitness={reg.fitness:.4f}  RMSE={reg.inlier_rmse:.4f}m")
    return gt_ds, recon_ds

def _chamfer_distance(a, b):
    ta = o3d.geometry.KDTreeFlann(a)
    tb = o3d.geometry.KDTreeFlann(b)
    pa = np.asarray(a.points)
    pb = np.asarray(b.points)
    d_a2b = np.array([np.sqrt(tb.search_knn_vector_3d(p,1)[2][0]) for p in pa])
    d_b2a = np.array([np.sqrt(ta.search_knn_vector_3d(p,1)[2][0]) for p in pb])
    return (d_a2b.mean()+d_b2a.mean())/2, d_a2b, d_b2a

def _dsm_rmse(gt_pcd, recon_pcd, grid):
    gp = np.asarray(gt_pcd.points)
    rp = np.asarray(recon_pcd.points)
    all_p = np.vstack([gp, rp])
    xm,ym = all_p[:,0].min(), all_p[:,1].min()
    nx = int((all_p[:,0].max()-xm)/grid)+1
    ny = int((all_p[:,1].max()-ym)/grid)+1
    def to_dsm(pts):
        d=np.full((nx,ny),np.nan)
        ix=((pts[:,0]-xm)/grid).astype(int).clip(0,nx-1)
        iy=((pts[:,1]-ym)/grid).astype(int).clip(0,ny-1)
        for xi,yi,z in zip(ix,iy,pts[:,2]):
            if np.isnan(d[xi,yi]) or z>d[xi,yi]: d[xi,yi]=z
        return d
    gd,rd=to_dsm(gp),to_dsm(rp)
    mask=~np.isnan(gd)&~np.isnan(rd)
    diff=gd[mask]-rd[mask]
    return np.sqrt(np.mean(diff**2)), np.mean(np.abs(diff)), diff

def _fscore(d_a2b, d_b2a, thr):
    p=float(np.mean(d_a2b<thr)); r=float(np.mean(d_b2a<thr))
    f=2*p*r/(p+r) if (p+r)>0 else 0.0
    return f,p,r

def _save_error_map(gt_ds, d_a2b, case_dir: Path):
    pts  = np.asarray(gt_ds.points)
    vmax = float(np.percentile(d_a2b, 95))
    fig, ax = plt.subplots(figsize=(10,10))
    sc = ax.scatter(pts[:,0], pts[:,1], c=d_a2b*100,
                    cmap='plasma', s=0.3, vmin=0, vmax=vmax*100)
    plt.colorbar(sc, label='Error [cm]', ax=ax)
    ax.set_title('DSM Error Map'); ax.set_aspect('equal')
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    plt.tight_layout()
    plt.savefig(str(case_dir/"error_map.png"), dpi=150, bbox_inches='tight')
    plt.close()

    high_err = d_a2b > np.percentile(d_a2b, 80)
    if high_err.sum() > 0:
        err_pcd = o3d.geometry.PointCloud()
        err_pcd.points = o3d.utility.Vector3dVector(pts[high_err])
        o3d.io.write_point_cloud(str(case_dir/"error_map_pts.ply"), err_pcd)
