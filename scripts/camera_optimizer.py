"""
3次元カメラ配置最適化モジュール — 観測点(ポール)モデル + 農場境界ゾーン

ゾーン定義:
  A: 農場境界の外側          → ポール配置禁止
  B: 境界から内側 fence_zone_width m の帯 → 外向き+内向き混合配置
  C: 農場内部 (B・D 以外)   → 全方位均等配置 (デフォルト)
  D: 建物周辺 building_zone_radius m 以内 → 建物外向き配置

farm_boundary.json がない場合は BBox 相対距離による "1"/"2"/"3" ゾーン判定に
フォールバックし、既存動作を完全に維持する。

使い方:
    from scripts.camera_optimizer import CameraOptimizer, CameraConfig, FarmBoundary

    farm, buildings = FarmBoudary.load_from_json(Path("cases/x/farm_boundary.json"))
    config  = CameraConfig(n_poles=15, fixed_per_pole=4, ptz_per_pole=2)
    optimizer = CameraOptimizer(mesh, config, farm=farm, buildings=buildings)
    poles = optimizer.optimize()
"""

import json
import numpy as np
import trimesh
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from pathlib import Path

try:
    from shapely.geometry import Polygon, Point
    _HAS_SHAPELY = True
except ImportError:
    _HAS_SHAPELY = False

try:
    from trimesh.ray.ray_pyembree import RayMeshIntersector
    _HAS_EMBREE = True
except ImportError:
    _HAS_EMBREE = False


# ── 農場境界 ──────────────────────────────────────────────────────────────────

class FarmBoundary:
    """
    農場外周ポリゴンを保持し、ゾーン判定・外向き方位計算を提供する。
    Shapely が未インストールの場合は ImportError を送出する。
    """

    def __init__(self, polygon_xy: np.ndarray):
        if not _HAS_SHAPELY:
            raise ImportError("shapely が必要です: pip install shapely>=2.0")
        pts = np.asarray(polygon_xy)
        # 閉じていなければ自動的に閉じる
        if not np.allclose(pts[0], pts[-1]):
            pts = np.vstack([pts, pts[0]])
        self._poly     = Polygon(pts)
        self._centroid = np.array(self._poly.centroid.coords[0])
        self._coords   = np.array(self._poly.exterior.coords)  # (N+1, 2)

    # ── 基本判定 ──────────────────────────────────────────────────────────────

    def contains(self, point: np.ndarray) -> bool:
        return self._poly.contains(Point(point[0], point[1]))

    def distance_to_boundary(self, point: np.ndarray) -> float:
        """
        境界までの距離を返す。
        農場内部: 正値、農場外部: 負値。
        """
        d = self._poly.exterior.distance(Point(point[0], point[1]))
        return d if self.contains(point) else -d

    # ── 外向き方位計算 ────────────────────────────────────────────────────────

    def outward_azimuth(self, point: np.ndarray) -> float:
        """
        point に最も近い境界セグメントの外向き法線を北基準時計回り方位角 [deg] で返す。
        ポリゴンの巻き方向に依存しない (centroid との内積で符号補正する)。
        """
        px, py = float(point[0]), float(point[1])
        coords = self._coords  # shape (N+1, 2)

        best_dist   = np.inf
        best_seg    = None  # (A, B) として (2,) np.ndarray のペア

        for i in range(len(coords) - 1):
            A  = coords[i]
            B  = coords[i + 1]
            AB = B - A
            ab_len = np.linalg.norm(AB)
            if ab_len < 1e-9:
                continue
            t  = np.clip(np.dot(np.array([px, py]) - A, AB) / ab_len ** 2, 0.0, 1.0)
            closest = A + t * AB
            d = np.linalg.norm(np.array([px, py]) - closest)
            if d < best_dist:
                best_dist = d
                best_seg  = (A, B)

        if best_seg is None:
            return 0.0

        A, B  = best_seg
        seg   = B - A
        seg  /= np.linalg.norm(seg) + 1e-9

        # 外向き法線候補 (右手回転)
        n = np.array([seg[1], -seg[0]])

        # centroid から point へのベクトルと同方向なら外向き、逆なら反転
        to_point = np.array([px, py]) - self._centroid
        if np.dot(n, to_point) < 0:
            n = -n

        return float(np.degrees(np.arctan2(n[0], n[1]))) % 360.0

    # ── JSON 読み込み ─────────────────────────────────────────────────────────

    @classmethod
    def load_from_json(
        cls, path: Path
    ) -> Tuple["FarmBoundary", Optional["BuildingMap"]]:
        """
        farm_boundary.json を読み込み (FarmBoundary, BuildingMap|None) を返す。
        "buildings" キーがなければ BuildingMap は None。
        """
        data      = json.loads(Path(path).read_text())
        boundary  = np.array(data["boundary"])
        farm      = cls(boundary)

        buildings = None
        if data.get("buildings"):
            fps       = [np.array(b["footprint"]) for b in data["buildings"]]
            buildings = BuildingMap(fps)

        return farm, buildings


# ── 建物マップ ────────────────────────────────────────────────────────────────

class BuildingMap:
    """建物フットプリント群を保持し、建物ゾーン判定と外向き方位計算を提供する。"""

    def __init__(self, footprints: List[np.ndarray]):
        if not _HAS_SHAPELY:
            raise ImportError("shapely が必要です: pip install shapely>=2.0")
        self._polys     = [Polygon(fp) for fp in footprints]
        self._centroids = [np.array(p.centroid.coords[0]) for p in self._polys]

    def is_in_building_zone(self, point: np.ndarray, radius: float) -> bool:
        p = Point(point[0], point[1])
        return any(poly.distance(p) <= radius for poly in self._polys)

    def nearest_building_azimuth(self, point: np.ndarray) -> float:
        """
        最近傍建物の重心から point への方向 = 建物外向き方位を返す。
        """
        p     = Point(point[0], point[1])
        dists = [poly.distance(p) for poly in self._polys]
        idx   = int(np.argmin(dists))
        vec   = np.array([point[0], point[1]]) - self._centroids[idx]
        norm  = np.linalg.norm(vec)
        if norm < 1e-6:
            return 0.0
        vec /= norm
        return float(np.degrees(np.arctan2(vec[0], vec[1]))) % 360.0


# ── データクラス ──────────────────────────────────────────────────────────────

@dataclass
class CameraConfig:
    # ── 観測点(ポール)構成 ──
    n_poles: int = 15
    fixed_per_pole: int = 4
    ptz_per_pole: int = 2
    pole_height_min: float = 2.5      # ポール最低高さ [m] (急斜面・木取付等)
    pole_height_max: float = 3.0      # ポール最大高さ [m] (平地標準)

    # ── 固定カメラ マウント設定 ──
    fixed_tilt_deg: float = -20.0
    fixed_fov_h: float = 90.0
    fixed_fov_v: float = 60.0

    # ── PTZカメラ マウント設定 ──
    ptz_fov_h: float = 60.0
    ptz_fov_v: float = 40.0
    ptz_z_offsets: List[float] = field(default_factory=lambda: [0.0, -1.5])

    # ── カメラ共通パラメータ ──
    image_width: int = 1920
    image_height: int = 1080
    fx: float = 1200.0
    fy: float = 1200.0

    # ── 農場境界ゾーン設定 ──
    fence_zone_width: float = 20.0       # Zone B: 境界から内側の幅 [m]
    building_zone_radius: float = 10.0   # Zone D: 建物フットプリントからの半径 [m]

    # ── 地形解析パラメータ ──
    slope_threshold_deg: float = 25.0
    ridge_percentile: float = 85.0
    valley_percentile: float = 15.0

    # ── 最適化パラメータ ──
    n_ray_samples: int = 64
    sa_iterations: int = 500
    sa_initial_temp: float = 10.0
    sa_cooling: float = 0.95
    zone_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "A": 0.0,   # 農場外: 配置禁止
            "B": 1.5,   # フェンスゾーン
            "C": 2.0,   # 内部ゾーン
            "D": 1.8,   # 建物ゾーン
            # フォールバック用 (farm_boundary.json なし)
            "1": 1.0, "2": 1.5, "3": 2.0,
        }
    )

    # ── 可視距離 ──
    max_visibility_range: float = 150.0


@dataclass
class CameraPose:
    id: int
    type: str            # "fixed" or "ptz"
    pole_id: int
    position: np.ndarray
    lookat: np.ndarray
    azimuth_deg: float
    fov_h: float
    fov_v: float
    zone: str = "C"      # "A"/"B"/"C"/"D" or "1"/"2"/"3" (fallback)
    coverage_score: float = 0.0

    def to_extrinsic_4x4(self) -> np.ndarray:
        pos    = self.position
        target = self.lookat
        up     = np.array([0., 0., 1.])

        z_cam = pos - target
        n = np.linalg.norm(z_cam)
        z_cam = z_cam / n if n > 1e-6 else np.array([0., 0., 1.])

        x_cam = np.cross(up, z_cam)
        xn = np.linalg.norm(x_cam)
        if xn < 1e-6:
            x_cam = np.cross(np.array([0., 1., 0.]), z_cam)
            xn = np.linalg.norm(x_cam)
        x_cam /= xn
        y_cam = np.cross(z_cam, x_cam)

        ext = np.eye(4)
        ext[:3, 0] = x_cam
        ext[:3, 1] = y_cam
        ext[:3, 2] = z_cam
        ext[:3, 3] = pos
        return ext

    def to_dict(self, intrinsic: dict) -> dict:
        ext = self.to_extrinsic_4x4()
        return {
            "id":             self.id,
            "pole_id":        self.pole_id,
            "type":           self.type,
            "extrinsic_4x4":  ext.tolist(),
            "intrinsic":      intrinsic,
            "position_enu":   self.position.tolist(),
            "lookat_enu":     self.lookat.tolist(),
            "azimuth_deg":    self.azimuth_deg,
            "fov_h":          self.fov_h,
            "fov_v":          self.fov_v,
            "zone":           self.zone,
            "coverage_score": self.coverage_score,
        }


@dataclass
class Pole:
    id: int
    position_ground: np.ndarray
    height: float
    zone: str = "C"      # "A"/"B"/"C"/"D" or "1"/"2"/"3" (fallback)
    pole_score: float = 0.0
    cameras: List[CameraPose] = field(default_factory=list)

    @property
    def top_position(self) -> np.ndarray:
        pos = self.position_ground.copy()
        pos[2] += self.height
        return pos

    def mount_cameras(
        self,
        config: CameraConfig,
        cam_id_start: int,
        terrain_centroid: np.ndarray,
        farm: Optional[FarmBoundary] = None,
        buildings: Optional[BuildingMap] = None,
    ) -> int:
        """
        ゾーンに応じたカメラ方位戦略でカメラをマウントする。

        Zone B: 外向き(n//2台) + 内向き(n//2台)の扇状配置
        Zone D: 建物外向き扇状配置
        Zone C / フォールバック: 360°均等配置
        """
        self.cameras = []
        cid  = cam_id_start
        top  = self.top_position
        step = config.fixed_fov_h * 0.9  # 10% オーバーラップ

        # ── 固定カメラ方位の決定 ──────────────────────────────────────────────
        zone = self.zone

        if zone == "B" and farm is not None:
            az_out  = farm.outward_azimuth(self.position_ground)
            az_in   = (az_out + 180.0) % 360.0
            n_fixed = config.fixed_per_pole
            n_out   = n_fixed // 2
            n_in    = n_fixed - n_out

            azimuths_out = [
                (az_out + (i - (n_out - 1) / 2) * step) % 360.0
                for i in range(n_out)
            ]
            azimuths_in = [
                (az_in + (i - (n_in - 1) / 2) * step) % 360.0
                for i in range(n_in)
            ]
            azimuths = azimuths_out + azimuths_in

        elif zone == "D" and buildings is not None:
            az_center = buildings.nearest_building_azimuth(self.position_ground)
            n_fixed   = config.fixed_per_pole
            azimuths  = [
                (az_center + (i - (n_fixed - 1) / 2) * step) % 360.0
                for i in range(n_fixed)
            ]

        else:
            # Zone C またはフォールバック: 全方位均等割り
            azimuths = list(
                np.linspace(0, 360, config.fixed_per_pole, endpoint=False)
            )

        # ── 固定カメラをマウント ──────────────────────────────────────────────
        tilt_rad = np.radians(config.fixed_tilt_deg)
        dist_h   = config.max_visibility_range * 0.5 * np.cos(tilt_rad)
        dz       = config.max_visibility_range * 0.5 * np.sin(tilt_rad)

        for az in azimuths:
            az_rad = np.radians(az)
            lookat = top + np.array([
                dist_h * np.sin(az_rad),
                dist_h * np.cos(az_rad),
                dz,
            ])
            self.cameras.append(CameraPose(
                id=cid, type="fixed", pole_id=self.id,
                position=top.copy(), lookat=lookat,
                azimuth_deg=az,
                fov_h=config.fixed_fov_h, fov_v=config.fixed_fov_v,
                zone=self.zone,
            ))
            cid += 1

        # ── PTZカメラをマウント ───────────────────────────────────────────────
        n_ptz     = config.ptz_per_pole
        z_offsets = (config.ptz_z_offsets + [0.0] * n_ptz)[:n_ptz]

        for i, z_off in enumerate(z_offsets):
            cam_pos = top.copy()
            cam_pos[2] += z_off

            if zone == "B" and farm is not None and i == 0:
                # PTZ[0] は外向き監視
                az_out = farm.outward_azimuth(self.position_ground)
                az_rad = np.radians(az_out)
                lookat = cam_pos + np.array([
                    dist_h * np.sin(az_rad),
                    dist_h * np.cos(az_rad),
                    dz,
                ])
                az = az_out
            elif zone == "D" and buildings is not None:
                az     = buildings.nearest_building_azimuth(self.position_ground)
                az_rad = np.radians(az)
                lookat = cam_pos + np.array([
                    dist_h * np.sin(az_rad),
                    dist_h * np.cos(az_rad),
                    dz,
                ])
            else:
                # 重心方向
                lookat = terrain_centroid.copy()
                dx = lookat[0] - cam_pos[0]
                dy = lookat[1] - cam_pos[1]
                az = float(np.degrees(np.arctan2(dx, dy))) % 360.0

            self.cameras.append(CameraPose(
                id=cid, type="ptz", pole_id=self.id,
                position=cam_pos, lookat=lookat,
                azimuth_deg=az,
                fov_h=config.ptz_fov_h, fov_v=config.ptz_fov_v,
                zone=self.zone,
            ))
            cid += 1

        return cid

    def all_camera_dicts(self, config: CameraConfig) -> List[dict]:
        cx   = config.image_width  / 2.0
        cy   = config.image_height / 2.0
        intr = {"fx": config.fx, "fy": config.fy, "cx": cx, "cy": cy,
                "width": config.image_width, "height": config.image_height}
        return [c.to_dict(intr) for c in self.cameras]


# ── 地形解析 ──────────────────────────────────────────────────────────────────

class TerrainAnalyzer:
    def __init__(self, mesh: trimesh.Trimesh, config: CameraConfig):
        self.mesh     = mesh
        self.config   = config
        self.vertices = np.array(mesh.vertices)

    def compute_vertex_slopes(self) -> np.ndarray:
        normals    = np.array(self.mesh.vertex_normals)
        cos_angles = np.clip(normals @ np.array([0., 0., 1.]), -1., 1.)
        return np.degrees(np.arccos(cos_angles))

    def detect_ridges(self) -> np.ndarray:
        z = self.vertices[:, 2]
        return z >= np.percentile(z, self.config.ridge_percentile)

    def detect_valleys(self) -> np.ndarray:
        z = self.vertices[:, 2]
        return z <= np.percentile(z, self.config.valley_percentile)

    def detect_steep_slopes(self) -> np.ndarray:
        return self.compute_vertex_slopes() >= self.config.slope_threshold_deg

    def effective_pole_height(self, pos: np.ndarray) -> float:
        """設置点の局所傾斜に応じてポール高さを決定 (平地=max, 急斜面=min)。"""
        dists  = np.linalg.norm(self.vertices[:, :2] - pos[:2], axis=1)
        nearby = dists < 10.0
        if nearby.sum() < 3:
            return self.config.pole_height_max

        local_slope = self.compute_vertex_slopes()[nearby].mean()
        lo  = self.config.pole_height_min
        hi  = self.config.pole_height_max
        thr = self.config.slope_threshold_deg

        if local_slope >= thr:
            return lo
        t = local_slope / thr
        return hi - t * (hi - lo)

    def get_pole_candidate_positions(self, n_candidates: int = 500) -> np.ndarray:
        pts, _    = trimesh.sample.sample_surface_even(self.mesh, n_candidates)
        ridge_idx = np.where(self.detect_ridges())[0]
        # 稜線頂点は最大200点にサブサンプル (全量追加すると54k点以上に膨張するため)
        if len(ridge_idx) > 200:
            ridge_idx = np.random.choice(ridge_idx, 200, replace=False)
        ridge_pts = self.vertices[ridge_idx]
        return np.vstack([pts, ridge_pts])

    def get_valley_exit_positions(self) -> np.ndarray:
        valley_mask = self.detect_valleys()
        if valley_mask.sum() < 3:
            return np.empty((0, 3))
        valley_pts = self.vertices[valley_mask]
        centroid   = valley_pts.mean(axis=0)
        cov        = np.cov((valley_pts - centroid).T)
        _, eigvecs = np.linalg.eigh(cov)
        axis       = eigvecs[:, -1]
        proj       = (valley_pts - centroid) @ axis
        extent     = proj.max() - proj.min()
        exits = np.array([
            centroid + axis * extent * 0.5,
            centroid - axis * extent * 0.5,
        ])
        exits[:, 2] += self.config.pole_height_max + 1.0
        return exits


# ── 可視性計算 ────────────────────────────────────────────────────────────────

class VisibilityCalculator:
    def __init__(self, mesh: trimesh.Trimesh, config: CameraConfig):
        self.config = config
        try:
            if _HAS_EMBREE:
                self.intersector = RayMeshIntersector(mesh)
            else:
                self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        except Exception:
            self.intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    def compute_pole_score(self, pole_top: np.ndarray, lookat: np.ndarray,
                           fov_h: float, fov_v: float) -> float:
        n      = self.config.n_ray_samples
        half_h = np.radians(fov_h / 2)
        half_v = np.radians(fov_v / 2)

        fwd = lookat - pole_top
        fn  = np.linalg.norm(fwd)
        if fn < 1e-6:
            return 0.0
        fwd /= fn
        up    = np.array([0., 0., 1.])
        right = np.cross(fwd, up)
        rn    = np.linalg.norm(right)
        if rn < 1e-6:
            return 0.0
        right /= rn
        up_cam = np.cross(right, fwd)

        h_a  = np.random.uniform(-half_h, half_h, n)
        v_a  = np.random.uniform(-half_v, half_v, n)
        dirs = (fwd[None] + np.tan(h_a)[:, None] * right[None]
                           + np.tan(v_a)[:, None] * up_cam[None])
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= np.where(norms < 1e-9, 1.0, norms)

        try:
            locs, idx_ray, _ = self.intersector.intersects_location(
                ray_origins=np.tile(pole_top, (n, 1)),
                ray_directions=dirs,
                multiple_hits=False,
            )
        except Exception:
            return 0.0

        hit_ratio   = len(idx_ray) / n
        avg_dist    = fn if len(locs) == 0 else np.mean(
            np.linalg.norm(locs - pole_top, axis=1))
        solid_angle = 4 * np.tan(half_h) * np.tan(half_v)
        return float(hit_ratio * avg_dist ** 2 * solid_angle)


# ── 配置最適化 ────────────────────────────────────────────────────────────────

class CameraOptimizer:
    """
    ポール(観測点)位置を最適化し、各ポールにカメラを自動マウントする。
    farm / buildings を渡すことで農場境界ゾーン制御が有効になる。
    """

    def __init__(
        self,
        mesh: trimesh.Trimesh,
        config: CameraConfig,
        farm: Optional[FarmBoundary] = None,
        buildings: Optional[BuildingMap] = None,
    ):
        self.mesh      = mesh
        self.config    = config
        self.farm      = farm
        self.buildings = buildings
        self.analyzer  = TerrainAnalyzer(mesh, config)
        self.vis_calc  = VisibilityCalculator(mesh, config)
        self._centroid = np.array(mesh.vertices).mean(axis=0)

    # ── ゾーン割り当て ────────────────────────────────────────────────────────

    def _assign_zone(self, pos: np.ndarray, bbox: np.ndarray) -> str:
        """
        farm がある場合: A/B/C/D を返す。
        farm がない場合: BBox 相対距離による "1"/"2"/"3" を返す (既存動作)。
        """
        if self.farm is None:
            # フォールバック: BBox 相対距離
            cx = (bbox[0] + bbox[3]) / 2
            cy = (bbox[1] + bbox[4]) / 2
            w  = bbox[3] - bbox[0]
            h  = bbox[4] - bbox[1]
            dx = abs(pos[0] - cx) / (w / 2 + 1e-9)
            dy = abs(pos[1] - cy) / (h / 2 + 1e-9)
            r  = max(dx, dy)
            return "1" if r > 0.7 else ("2" if r > 0.4 else "3")

        if not self.farm.contains(pos):
            return "A"
        if (self.buildings is not None
                and self.buildings.is_in_building_zone(
                    pos, self.config.building_zone_radius)):
            return "D"
        if self.farm.distance_to_boundary(pos) <= self.config.fence_zone_width:
            return "B"
        return "C"

    # ── スコアリング ──────────────────────────────────────────────────────────

    def _score_pole_position(
        self, pos: np.ndarray, existing: List[Pole],
        precomputed_height: Optional[float] = None,
    ) -> float:
        # precomputed_height が渡された場合は36万頂点走査をスキップ
        height = precomputed_height if precomputed_height is not None \
                 else self.analyzer.effective_pole_height(pos)
        top    = pos.copy()
        top[2] += height
        lookat = self._centroid.copy()

        score = self.vis_calc.compute_pole_score(
            top, lookat,
            self.config.fixed_fov_h,
            self.config.fixed_fov_v,
        )

        for p in existing:
            d = np.linalg.norm(pos[:2] - p.position_ground[:2])
            if d < 40.0:
                score -= (40.0 - d) * 0.15

        return score

    # ── 最適化メイン ──────────────────────────────────────────────────────────

    def optimize(self) -> List[Pole]:
        """
        ポール位置を最適化して返す。各ポールにはカメラが自動マウント済み。
        """
        config = self.config
        n_cand = max(config.n_poles * 15, 400)
        bbox   = np.array(self.mesh.bounds).flatten()

        candidates   = self.analyzer.get_pole_candidate_positions(n_cand)
        valley_exits = self.analyzer.get_valley_exit_positions()

        # Zone A (農場外) をポール候補から除外
        # shapely.vectorized で一括判定 (Pythonループより大幅に高速)
        if self.farm is not None:
            try:
                from shapely.vectorized import contains as shp_contains
                inside_mask = shp_contains(
                    self.farm._poly,
                    candidates[:, 0], candidates[:, 1],
                )
            except ImportError:
                inside_mask = np.array([self.farm.contains(p) for p in candidates])
            candidates = candidates[inside_mask]

        # ── 候補点のゾーンと高さを事前一括キャッシュ ────────────────────────
        # 貪欲法ループ内で毎回計算すると candidates数×ポール数 回の重複になるため
        # ここで1回だけ全候補を走査してキャッシュする
        cand_zones   = [self._assign_zone(p, bbox) for p in candidates]
        cand_heights = [self.analyzer.effective_pole_height(p) for p in candidates]

        # Zone A (農場外) をあらかじめ除外したインデックスリストを作る
        valid_indices = [i for i, z in enumerate(cand_zones) if z != "A"]

        selected: List[Pole] = []
        pole_id = 0

        # ── 谷出口に優先配置 ─────────────────────────────────────────────────
        n_valley = min(len(valley_exits), config.n_poles // 4)
        for i in range(n_valley):
            pos    = valley_exits[i].copy()
            pos[2] -= config.pole_height_max + 1.0
            # Zone A なら谷出口への強制配置もスキップ
            if self.farm is not None and not self.farm.contains(pos):
                continue
            height = self.analyzer.effective_pole_height(pos)
            zone   = self._assign_zone(pos, bbox)
            score  = self._score_pole_position(pos, selected)
            selected.append(Pole(id=pole_id, position_ground=pos,
                                 height=height, zone=zone, pole_score=score))
            pole_id += 1

        # ── 貪欲法: 残りポールを配置 ─────────────────────────────────────────
        used = set()
        for _ in range(config.n_poles - len(selected)):
            best_score = -np.inf
            best_idx   = -1

            for idx in valid_indices:          # Zone A 除外済みリストを走査
                if idx in used:
                    continue
                zone = cand_zones[idx]         # キャッシュから取得 (Shapely呼び出しなし)
                s = self._score_pole_position(candidates[idx], selected, cand_heights[idx])
                # ゾーン重みを掛けてスコアを調整
                s *= config.zone_weights.get(zone, 1.0)
                if s > best_score:
                    best_score = s
                    best_idx   = idx

            if best_idx < 0:
                break

            used.add(best_idx)
            pos    = candidates[best_idx]
            height = cand_heights[best_idx]    # キャッシュから取得
            zone   = cand_zones[best_idx]      # キャッシュから取得
            selected.append(Pole(id=pole_id, position_ground=pos,
                                 height=height, zone=zone,
                                 pole_score=best_score))
            pole_id += 1

        # ── 焼きなまし ────────────────────────────────────────────────────────
        selected = self._simulated_annealing(
            selected, candidates, bbox, cand_zones, cand_heights)

        # ── 各ポールにカメラをマウント ────────────────────────────────────────
        cam_id = 0
        for pole in selected:
            cam_id = pole.mount_cameras(
                config, cam_id, self._centroid,
                farm=self.farm, buildings=self.buildings,
            )

        return selected

    def _simulated_annealing(
        self, poles: List[Pole], candidates: np.ndarray, bbox: np.ndarray,
        cand_zones: List[str], cand_heights: List[float],
    ) -> List[Pole]:
        config = self.config
        T      = config.sa_initial_temp

        for _ in range(config.sa_iterations):
            if not poles:
                break
            idx      = np.random.randint(len(poles))
            pole     = poles[idx]
            cand_idx = np.random.randint(len(candidates))
            new_pos  = candidates[cand_idx]

            new_zone = cand_zones[cand_idx]    # キャッシュから取得
            if new_zone == "A":
                T *= config.sa_cooling
                continue

            others        = [p for i, p in enumerate(poles) if i != idx]
            old_score     = pole.pole_score * config.zone_weights.get(str(pole.zone), 1.0)
            new_score_raw = self._score_pole_position(
                new_pos, others, cand_heights[cand_idx])  # 高さもキャッシュ利用
            new_score     = new_score_raw * config.zone_weights.get(new_zone, 1.0)

            delta = new_score - old_score
            if delta > 0 or np.random.rand() < np.exp(delta / (T + 1e-9)):
                poles[idx] = Pole(
                    id=pole.id, position_ground=new_pos,
                    height=cand_heights[cand_idx], zone=new_zone,
                    pole_score=new_score_raw,
                )
            T *= config.sa_cooling

        return poles

    # ── JSON 出力 ─────────────────────────────────────────────────────────────

    def to_json(self, poles: List[Pole], case_name: str) -> dict:
        all_cameras = []
        poles_info  = []
        for pole in poles:
            all_cameras.extend(pole.all_camera_dicts(self.config))
            poles_info.append({
                "id":              pole.id,
                "position_ground": pole.position_ground.tolist(),
                "height":          pole.height,
                "zone":            pole.zone,
                "pole_score":      pole.pole_score,
                "n_cameras":       len(pole.cameras),
                "camera_ids":      [c.id for c in pole.cameras],
            })

        return {
            "case_name":   case_name,
            "coord_frame": "ENU",
            "unit":        "meters",
            "poles":       poles_info,
            "cameras":     all_cameras,
        }
