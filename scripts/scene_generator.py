"""
日本の農牧場想定シーン自動生成モジュール

想定スケール:
  地形総面積  : 3km × 3km  (農場外縁の地形も含む)
  農場敷地    : デフォルト 2km × 2km 長方形 (boundary_irregularity=0)
               boundary_irregularity > 0 にすると不整形ポリゴンに切り替わる
  高低差      : 0〜150m  (山・谷が複雑に入り組む日本的地形)
  解像度      : デフォルト 5m/グリッド → 600×600 = 36万頂点

生成物:
  cases/{name}/terrain.ply          地形メッシュ
  cases/{name}/farm_boundary.json   農場境界 + 建物FP + 森林パッチ
  cases/{name}/scene_preview.png    俯瞰確認図 (左:散布図 / 右:等高線)

farm_boundary.json 構造:
  {
    "coord_frame": "ENU",
    "unit": "meters",
    "boundary": [[x,y], ...],          ← 不整形ポリゴン(閉じている)
    "buildings": [{"id":0, "footprint":[[x,y],...], ...}],
    "forests":   [{"id":0, "center":[x,y], "radius":r}]
  }
"""

import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon, Circle
from matplotlib.collections import PatchCollection
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from noise import pnoise2
    _HAS_NOISE = True
except ImportError:
    _HAS_NOISE = False


# ── 設定 ──────────────────────────────────────────────────────────────────────

@dataclass
class SceneConfig:
    # ── 地形 ──────────────────────────────────────────────────────────────────
    terrain_size_x: float = 3000.0      # 東西 [m]
    terrain_size_y: float = 3000.0      # 南北 [m]
    terrain_resolution: float = 5.0     # グリッド間隔 [m]
    elevation_max: float = 120.0        # 最大標高 [m]

    # Perlinノイズ: 複数スケールを重ねて日本的な山・谷地形を再現
    # 大スケール: 山の骨格 / 中スケール: 尾根・谷 / 小スケール: 細かい起伏
    noise_layers: List[dict] = field(default_factory=lambda: [
        {"scale": 0.0005, "octaves": 4, "weight": 1.00},  # 大スケール (山体)
        {"scale": 0.0015, "octaves": 3, "weight": 0.40},  # 中スケール (尾根谷)
        {"scale": 0.0050, "octaves": 2, "weight": 0.15},  # 小スケール (細起伏)
    ])

    # ── 農場境界 ──────────────────────────────────────────────────────────────
    farm_size_x: float = 2000.0         # 敷地 東西幅 [m]  (長方形モード)
    farm_size_y: float = 2000.0         # 敷地 南北幅 [m]  (長方形モード)
    # boundary_irregularity = 0 → 長方形 (デフォルト)
    # boundary_irregularity > 0 → 不整形ポリゴン (farm_diameter を半径として使用)
    farm_diameter: float = 2000.0       # 不整形モード時の概略直径 [m]
    boundary_n_vertices: int = 10       # 不整形モード: 境界ポリゴン頂点数
    boundary_irregularity: float = 0.0  # 0=長方形, 0より大=不整形ポリゴン
    boundary_spikiness: float = 0.15    # 不整形モード: 凹みの深さ

    # ── 建物クラスタ ──────────────────────────────────────────────────────────
    n_buildings: int = 6
    cluster_radius: float = 80.0        # クラスタ半径 [m]
    building_gap: float = 6.0           # 建物間隔 [m]

    # 小屋〜大型施設のサイズプール
    building_sizes: List[Tuple[float, float]] = field(default_factory=lambda: [
        ( 5.0,  8.0),   # 小屋
        ( 6.0, 10.0),   # 小屋(大)
        ( 8.0, 15.0),   # 中型施設
        (10.0, 20.0),   # 中型施設(大)
        (12.0, 25.0),   # 大型施設
        (15.0, 30.0),   # 大型倉庫
        (20.0, 40.0),   # 畜舎/主屋
    ])
    building_height_range: Tuple[float, float] = (3.5, 8.0)

    # ── 森林パッチ ────────────────────────────────────────────────────────────
    n_forest_patches: int = 3           # 森林パッチ数
    forest_radius_range: Tuple[float, float] = (80.0, 250.0)  # 半径 [m]

    # ── 平坦エリア探索 ────────────────────────────────────────────────────────
    flat_slope_threshold_deg: float = 15.0
    cluster_prefer_low: bool = True     # 建物は低い・平坦な場所に配置


# ── 不整形ポリゴン生成 ────────────────────────────────────────────────────────

def _random_polygon(
    cx: float, cy: float, radius: float,
    n_verts: int, irregularity: float, spikiness: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    中心 (cx, cy) 周囲に不整形な閉じたポリゴンを生成する。
    irregularity : 角度間隔のランダム度 (0=均等, 1=最大ランダム)
    spikiness    : 半径のランダム度 (凹凸の深さ)
    戻り値: shape (n_verts+1, 2)  最後の点は先頭と同じ (閉じている)
    """
    # 角度間隔をランダム化
    angles = np.linspace(0, 2 * np.pi, n_verts, endpoint=False)
    d_angle = 2 * np.pi / n_verts
    angles += rng.uniform(-d_angle * irregularity,
                           d_angle * irregularity, n_verts)
    angles = np.sort(angles)

    # 各頂点の半径をランダム化
    radii = radius * (1 + rng.uniform(-spikiness, spikiness, n_verts))
    radii = np.clip(radii, radius * 0.5, radius * 1.4)

    xs = cx + radii * np.cos(angles)
    ys = cy + radii * np.sin(angles)

    # 地形範囲にクリップ
    pts = np.column_stack([xs, ys])
    pts = np.vstack([pts, pts[0]])   # 閉じる
    return pts


def make_farm_boundary(
    config: SceneConfig, rng: np.random.Generator
) -> np.ndarray:
    """
    農場境界ポリゴンを生成する。

    boundary_irregularity == 0 (デフォルト):
        地形中央に farm_size_x × farm_size_y の長方形を生成する。
    boundary_irregularity > 0:
        farm_diameter を半径とした不整形ポリゴンを生成する。
    """
    sx, sy = config.terrain_size_x, config.terrain_size_y
    cx     = sx / 2
    cy     = sy / 2
    margin = 50.0

    if config.boundary_irregularity == 0.0:
        # ── 長方形モード ──────────────────────────────────────────────────────
        hw = config.farm_size_x / 2
        hh = config.farm_size_y / 2

        # 地形からはみ出さないようクリップ
        hw = min(hw, cx - margin)
        hh = min(hh, cy - margin)

        pts = np.array([
            [cx - hw, cy - hh],
            [cx + hw, cy - hh],
            [cx + hw, cy + hh],
            [cx - hw, cy + hh],
            [cx - hw, cy - hh],   # 閉じる
        ])
        return pts

    # ── 不整形ポリゴンモード ──────────────────────────────────────────────────
    r   = config.farm_diameter / 2
    cx += float(rng.uniform(-r * 0.2, r * 0.2))
    cy += float(rng.uniform(-r * 0.2, r * 0.2))

    pts = _random_polygon(
        cx, cy, r,
        n_verts=config.boundary_n_vertices,
        irregularity=config.boundary_irregularity,
        spikiness=config.boundary_spikiness,
        rng=rng,
    )
    pts[:, 0] = np.clip(pts[:, 0], margin, sx - margin)
    pts[:, 1] = np.clip(pts[:, 1], margin, sy - margin)
    return pts


# ── 地形生成 ──────────────────────────────────────────────────────────────────

def generate_terrain(config: SceneConfig, seed: int = 0) -> trimesh.Trimesh:
    """
    Perlinノイズ多層重ね合わせで山・谷が複雑な日本的地形を生成する。
    """
    sx, sy = config.terrain_size_x, config.terrain_size_y
    res    = config.terrain_resolution
    nx     = int(sx / res) + 1
    ny     = int(sy / res) + 1

    xs = np.linspace(0, sx, nx)
    ys = np.linspace(0, sy, ny)
    XX, YY = np.meshgrid(xs, ys)   # shape (ny, nx)

    rng = np.random.default_rng(seed)
    ZZ  = np.zeros((ny, nx))

    if _HAS_NOISE:
        for layer in config.noise_layers:
            sc   = layer["scale"]
            octs = layer["octaves"]
            w    = layer["weight"]
            ox   = float(rng.uniform(0, 10000))
            oy   = float(rng.uniform(0, 10000))
            layer_z = np.array([
                [pnoise2(ox + x * sc, oy + y * sc, octaves=octs,
                         persistence=0.5, lacunarity=2.0)
                 for x in xs]
                for y in ys
            ])
            ZZ += w * layer_z
    else:
        # noise 未インストール時の代替 (多周波 sin/cos 重ね合わせ)
        for layer in config.noise_layers:
            sc = layer["scale"]
            w  = layer["weight"]
            for _ in range(layer.get("octaves", 3)):
                freq = sc * 10
                ph_x = float(rng.uniform(0, 2 * np.pi))
                ph_y = float(rng.uniform(0, 2 * np.pi))
                ZZ += w * np.sin(freq * XX + ph_x) * np.cos(freq * YY + ph_y)
                sc *= 2.0
                w  *= 0.5

    # 0〜elevation_max に正規化
    ZZ -= ZZ.min()
    ZZ  = ZZ / (ZZ.max() + 1e-9) * config.elevation_max

    # 地形端部をフェードアウト (境界ギザギザ防止)
    fade_frac = 0.06   # 端から 6% をフェード
    fade_px_x = int(nx * fade_frac)
    fade_px_y = int(ny * fade_frac)
    if fade_px_x > 0:
        fade_x = np.ones(nx)
        fade_x[:fade_px_x]  = np.linspace(0, 1, fade_px_x)
        fade_x[-fade_px_x:] = np.linspace(1, 0, fade_px_x)
        ZZ *= fade_x[np.newaxis, :]
    if fade_px_y > 0:
        fade_y = np.ones(ny)
        fade_y[:fade_px_y]  = np.linspace(0, 1, fade_px_y)
        fade_y[-fade_px_y:] = np.linspace(1, 0, fade_px_y)
        ZZ *= fade_y[:, np.newaxis]

    # メッシュ構築
    verts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

    faces = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            i00 = iy * nx + ix
            i10 = iy * nx + ix + 1
            i01 = (iy + 1) * nx + ix
            i11 = (iy + 1) * nx + ix + 1
            faces.append([i00, i10, i11])
            faces.append([i00, i11, i01])

    mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.array(faces),
        process=True,
    )
    return mesh


# ── 農場内かどうかの判定 (Shapely なし版 / あり版) ───────────────────────────

def _point_in_polygon(pt: np.ndarray, polygon: np.ndarray) -> bool:
    """Ray casting 法による点-ポリゴン内包判定。"""
    x, y = float(pt[0]), float(pt[1])
    n     = len(polygon) - 1   # 最後は先頭の複製
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


# ── 平坦エリア探索 ─────────────────────────────────────────────────────────────

def find_cluster_site(
    mesh: trimesh.Trimesh,
    boundary: np.ndarray,
    config: SceneConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    農場境界内で平坦かつ低い(または中程度の)標高エリアを探して
    建物クラスタ中心を決定する。
    """
    verts  = np.array(mesh.vertices)
    norms  = np.array(mesh.vertex_normals)
    slopes = np.degrees(np.arccos(np.clip(norms @ [0, 0, 1], -1, 1)))
    z      = verts[:, 2]

    # 農場内 かつ 緩斜面 の頂点を候補とする
    in_farm = np.array([_point_in_polygon(v[:2], boundary) for v in verts])
    flat    = slopes < config.flat_slope_threshold_deg
    cands   = verts[in_farm & flat]

    if len(cands) < 20:
        # 平坦エリアが少ない場合: 農場内の低い場所を選ぶ
        farm_verts = verts[in_farm]
        if len(farm_verts) == 0:
            return np.array([config.terrain_size_x / 2,
                             config.terrain_size_y / 2])
        low_idx = np.argsort(farm_verts[:, 2])
        pool    = farm_verts[low_idx[:max(10, len(farm_verts)//5)]]
        cands   = pool

    # 低標高 優先 (農場内の標高分布に対して下位 40% から選ぶ)
    z_cands = cands[:, 2]
    threshold = np.percentile(z_cands, 40)
    low_cands = cands[z_cands <= threshold]
    if len(low_cands) == 0:
        low_cands = cands

    idx = rng.integers(0, len(low_cands))
    return low_cands[idx, :2]


# ── 建物配置 ──────────────────────────────────────────────────────────────────

@dataclass
class BuildingDef:
    id: int
    cx: float
    cy: float
    w: float
    d: float
    h: float
    rot_deg: float

    @property
    def footprint(self) -> List[List[float]]:
        hw, hd = self.w / 2, self.d / 2
        corners = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
        r   = np.radians(self.rot_deg)
        rot = np.array([[np.cos(r), -np.sin(r)],
                        [np.sin(r),  np.cos(r)]])
        rotated = (rot @ corners.T).T
        return (rotated + np.array([self.cx, self.cy])).tolist()


def place_buildings(
    config: SceneConfig,
    cluster_center: np.ndarray,
    boundary: np.ndarray,
    rng: np.random.Generator,
) -> List[BuildingDef]:
    """
    クラスタ中心付近に重なりなく建物を配置する。
    農場境界内に収まることも確認する。
    """
    sizes = config.building_sizes
    n     = min(config.n_buildings, len(sizes))
    chosen_sizes = [sizes[i] for i in rng.choice(len(sizes), n, replace=False)]
    h_min, h_max = config.building_height_range

    buildings: List[BuildingDef] = []

    for bid, (w, d) in enumerate(chosen_sizes):
        placed = False
        for _ in range(300):
            angle  = float(rng.uniform(0, 2 * np.pi))
            radius = float(rng.uniform(0, config.cluster_radius * 0.8))
            cx = cluster_center[0] + radius * np.cos(angle)
            cy = cluster_center[1] + radius * np.sin(angle)
            rot = float(rng.uniform(0, 90))

            # 農場内かチェック
            if not _point_in_polygon(np.array([cx, cy]), boundary):
                continue

            # 既存建物との重なりチェック
            margin  = config.building_gap
            overlap = False
            for b in buildings:
                dist = np.hypot(cx - b.cx, cy - b.cy)
                if dist < (max(w, d) + max(b.w, b.d)) / 2 + margin:
                    overlap = True
                    break

            if not overlap:
                h = float(rng.uniform(h_min, h_max))
                buildings.append(BuildingDef(
                    id=bid, cx=cx, cy=cy, w=w, d=d, h=h, rot_deg=rot
                ))
                placed = True
                break

        if not placed:
            print(f"      警告: B{bid} ({w:.0f}×{d:.0f}m) の配置に失敗 (スキップ)")

    return buildings


# ── 森林パッチ配置 ────────────────────────────────────────────────────────────

@dataclass
class ForestPatch:
    id: int
    cx: float
    cy: float
    radius: float


def place_forests(
    config: SceneConfig,
    boundary: np.ndarray,
    buildings: List[BuildingDef],
    rng: np.random.Generator,
) -> List[ForestPatch]:
    """
    農場内の建物から離れた場所に森林パッチを配置する。
    """
    farms_center = boundary[:-1].mean(axis=0)   # 農場の重心
    r_min, r_max = config.forest_radius_range
    # 農場の概略半径
    farm_r       = np.linalg.norm(boundary[:-1] - farms_center, axis=1).mean()

    forests: List[ForestPatch] = []

    for fid in range(config.n_forest_patches):
        for _ in range(200):
            angle  = float(rng.uniform(0, 2 * np.pi))
            radius = float(rng.uniform(farm_r * 0.1, farm_r * 0.75))
            cx = farms_center[0] + radius * np.cos(angle)
            cy = farms_center[1] + radius * np.sin(angle)

            if not _point_in_polygon(np.array([cx, cy]), boundary):
                continue

            fr = float(rng.uniform(r_min, r_max))

            # 建物クラスタから十分離れているか
            too_close = any(
                np.hypot(cx - b.cx, cy - b.cy) < fr + config.cluster_radius * 0.5
                for b in buildings
            )
            # 他の森林パッチと重なりすぎていないか
            overlap = any(
                np.hypot(cx - f.cx, cy - f.cy) < (fr + f.radius) * 0.5
                for f in forests
            )
            if not too_close and not overlap:
                forests.append(ForestPatch(id=fid, cx=cx, cy=cy, radius=fr))
                break

    return forests


# ── JSON エクスポート ──────────────────────────────────────────────────────────

def export_farm_boundary_json(
    case_dir: Path,
    boundary: np.ndarray,
    buildings: List[BuildingDef],
    forests: List[ForestPatch],
) -> Path:
    data = {
        "coord_frame": "ENU",
        "unit":        "meters",
        "boundary":    boundary.tolist(),
        "buildings": [
            {
                "id":        b.id,
                "center":    [b.cx, b.cy],
                "size_wdh":  [b.w, b.d, b.h],
                "rot_deg":   b.rot_deg,
                "footprint": b.footprint,
            }
            for b in buildings
        ],
        "forests": [
            {"id": f.id, "center": [f.cx, f.cy], "radius": f.radius}
            for f in forests
        ],
    }
    out = case_dir / "farm_boundary.json"
    out.write_text(json.dumps(data, indent=2))
    return out


# ── プレビュー図 ──────────────────────────────────────────────────────────────

def save_scene_preview(
    case_dir: Path,
    mesh: trimesh.Trimesh,
    boundary: np.ndarray,
    buildings: List[BuildingDef],
    forests: List[ForestPatch],
    config: SceneConfig,
    seed: int,
) -> Path:
    verts = np.array(mesh.vertices)
    sx, sy = config.terrain_size_x, config.terrain_size_y
    res    = config.terrain_resolution
    nx     = int(sx / res) + 1
    ny     = int(sy / res) + 1

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # ── 左: 散布図 + 配置図 ──────────────────────────────────────────────────
    ax = axes[0]
    # 間引いてプロット (全頂点は重い)
    step = max(1, len(verts) // 50000)
    sv   = verts[::step]
    sc   = ax.scatter(sv[:, 0], sv[:, 1], c=sv[:, 2],
                      cmap='terrain', s=0.3, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='Elevation [m]', shrink=0.7)

    # 農場境界
    bnd_patch = MplPolygon(boundary[:-1], closed=True,
                            fill=False, edgecolor='lime',
                            linewidth=2.0, linestyle='--')
    ax.add_patch(bnd_patch)
    ax.plot([], [], color='lime', linestyle='--', linewidth=2, label='Farm boundary')

    # 森林パッチ
    for f in forests:
        circ = Circle((f.cx, f.cy), f.radius,
                      alpha=0.25, facecolor='forestgreen',
                      edgecolor='darkgreen', linewidth=1.2)
        ax.add_patch(circ)
    if forests:
        ax.plot([], [], color='forestgreen', linewidth=0,
                marker='o', markersize=10, alpha=0.4, label='Forest')

    # 建物フットプリント
    for b in buildings:
        fp = np.array(b.footprint + [b.footprint[0]])
        ax.fill(fp[:, 0], fp[:, 1], alpha=0.7, color='tomato', zorder=4)
        ax.plot(fp[:, 0], fp[:, 1], 'darkred', linewidth=1.0, zorder=5)
        ax.text(b.cx, b.cy, f"B{b.id}\n{b.w:.0f}x{b.d:.0f}",
                ha='center', va='center', fontsize=5.5,
                color='white', fontweight='bold', zorder=6)
    if buildings:
        ax.plot([], [], color='tomato', linewidth=0,
                marker='s', markersize=8, alpha=0.7, label='Buildings')

    ax.set_aspect('equal')
    ax.set_xlim(0, sx); ax.set_ylim(0, sy)
    ax.set_title(f'Terrain + Farm layout  (seed={seed})', fontsize=11)
    ax.set_xlabel('East [m]'); ax.set_ylabel('North [m]')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)

    # ── 右: 等高線図 ─────────────────────────────────────────────────────────
    ax2 = axes[1]
    ZZ  = verts[:, 2].reshape(ny, nx)
    XX  = verts[:, 0].reshape(ny, nx)
    YY  = verts[:, 1].reshape(ny, nx)

    # 等高線数を標高範囲に応じて調整
    n_levels = max(15, int(config.elevation_max / 5))
    cf = ax2.contourf(XX, YY, ZZ, levels=n_levels, cmap='terrain', alpha=0.85)
    plt.colorbar(cf, ax=ax2, label='Elevation [m]', shrink=0.7)
    ax2.contour(XX, YY, ZZ, levels=n_levels // 2,
                colors='white', linewidths=0.3, alpha=0.4)

    bnd2 = MplPolygon(boundary[:-1], closed=True,
                       fill=False, edgecolor='lime',
                       linewidth=2.0, linestyle='--')
    ax2.add_patch(bnd2)

    for f in forests:
        circ2 = Circle((f.cx, f.cy), f.radius,
                       alpha=0.3, facecolor='forestgreen',
                       edgecolor='lime', linewidth=1.0)
        ax2.add_patch(circ2)

    for b in buildings:
        fp = np.array(b.footprint + [b.footprint[0]])
        ax2.fill(fp[:, 0], fp[:, 1], alpha=0.8, color='tomato', zorder=4)

    ax2.set_aspect('equal')
    ax2.set_xlim(0, sx); ax2.set_ylim(0, sy)
    ax2.set_title('Elevation contour map', fontsize=11)
    ax2.set_xlabel('East [m]'); ax2.set_ylabel('North [m]')
    ax2.grid(True, alpha=0.2)

    # ── 情報テキスト ──────────────────────────────────────────────────────────
    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    info = (
        f"Terrain {sx/1000:.1f}x{sy/1000:.1f} km  "
        f"elev {z_min:.0f}-{z_max:.0f} m  "
        f"{len(mesh.vertices):,} verts / {len(mesh.faces):,} faces\n"
        f"Farm {config.farm_diameter/1000:.1f} km dia  "
        f"Buildings: {len(buildings)}  "
        f"Forests: {len(forests)}  "
        f"seed={seed}"
    )
    fig.suptitle(info, fontsize=9, y=0.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = case_dir / "scene_preview.png"
    plt.savefig(str(out), dpi=130, bbox_inches='tight')
    plt.close()
    return out


# ── メイン生成関数 ────────────────────────────────────────────────────────────

def generate_scene(
    case_name: str,
    config: Optional[SceneConfig] = None,
    seed: int = 0,
    cases_root: Path = Path("cases"),
) -> dict:
    """
    シーンを生成し case_dir に全ファイルを書き出す。
    戻り値: 生成物のパス辞書
    """
    if config is None:
        config = SceneConfig()

    rng      = np.random.default_rng(seed)
    case_dir = cases_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    sx, sy = config.terrain_size_x, config.terrain_size_y
    res    = config.terrain_resolution
    nx     = int(sx / res) + 1
    ny     = int(sy / res) + 1
    n_verts = nx * ny

    print(f"[1/5] 地形生成  {sx/1000:.1f}x{sy/1000:.1f} km  "
          f"高低差 0-{config.elevation_max:.0f}m  "
          f"解像度 {res:.0f}m  ({nx}x{ny}={n_verts:,} verts)  seed={seed}")
    mesh = generate_terrain(config, seed=seed)

    terrain_ply = case_dir / "terrain.ply"
    mesh.export(str(terrain_ply))
    print(f"      -> {terrain_ply}  ({len(mesh.vertices):,} verts)")

    print("[2/5] 農場境界ポリゴン生成 (不整形)")
    boundary = make_farm_boundary(config, rng)
    print(f"      -> {len(boundary)-1} 頂点ポリゴン")

    print("[3/5] 建物クラスタ配置")
    cluster_center = find_cluster_site(mesh, boundary, config, rng)
    print(f"      クラスタ中心: ({cluster_center[0]:.0f}, {cluster_center[1]:.0f}) m")
    buildings = place_buildings(config, cluster_center, boundary, rng)
    print(f"      -> {len(buildings)} 棟配置")
    for b in buildings:
        print(f"         B{b.id}: {b.w:.0f}x{b.d:.0f}m  h={b.h:.1f}m  "
              f"rot={b.rot_deg:.0f}deg  @ ({b.cx:.0f},{b.cy:.0f})")

    print("[4/5] 森林パッチ配置")
    forests = place_forests(config, boundary, buildings, rng)
    print(f"      -> {len(forests)} パッチ配置")
    for f in forests:
        print(f"         F{f.id}: r={f.radius:.0f}m @ ({f.cx:.0f},{f.cy:.0f})")

    print("[5/5] farm_boundary.json + プレビュー図生成")
    json_path = export_farm_boundary_json(case_dir, boundary, buildings, forests)
    preview   = save_scene_preview(
        case_dir, mesh, boundary, buildings, forests, config, seed
    )
    print(f"      -> {json_path}")
    print(f"      -> {preview}")

    print(f"\n完了: {case_dir}")
    return {
        "case_name":     case_name,
        "case_dir":      str(case_dir),
        "terrain_ply":   str(terrain_ply),
        "boundary_json": str(json_path),
        "preview_png":   str(preview),
        "n_buildings":   len(buildings),
        "n_forests":     len(forests),
        "seed":          seed,
        "config":        config,
    }
