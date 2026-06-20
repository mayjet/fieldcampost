"""
日本の農牧場 テストケース生成スクリプト

デフォルト:
  地形総面積  3km x 3km  / 農場敷地 ~2km 不整形ポリゴン
  高低差 0-120m / グリッド 5m / 建物6棟 / 森林3パッチ

使い方:
  # デフォルト (3km x 3km, seed=42)
  python generate_case.py --case test01 --seed 42

  # 地形サイズ・起伏を調整
  python generate_case.py --case ranch_hilly --seed 7 \\
      --elev-max 150 --noise-large 0.0004 --n-buildings 8 --n-forests 4

  # 小さめ・平坦な農場
  python generate_case.py --case small_flat --seed 3 \\
      --terrain-x 1500 --terrain-y 1500 --farm-diameter 900 \\
      --elev-max 40 --n-buildings 5 --n-forests 2

生成物:
  cases/{name}/terrain.ply         地形メッシュ
  cases/{name}/farm_boundary.json  農場境界 + 建物FP + 森林パッチ
  cases/{name}/scene_preview.png   確認図

次のステップ:
  pipeline.ipynb の CASE_NAME を合わせて Cell 1-10 を実行
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.scene_generator import SceneConfig, generate_scene


def parse_args():
    p = argparse.ArgumentParser(
        description="日本農牧場 シミュレーション テストケース生成",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case",   required=True,
                   help="ケース名 (cases/ 以下のフォルダ名)")
    p.add_argument("--seed",   type=int, default=0,
                   help="乱数シード (同値で完全再現)")

    # ── 地形 ──────────────────────────────────────────────────────────────────
    g = p.add_argument_group("地形")
    g.add_argument("--terrain-x",    type=float, default=3000.0,
                   help="地形 東西サイズ [m]")
    g.add_argument("--terrain-y",    type=float, default=3000.0,
                   help="地形 南北サイズ [m]")
    g.add_argument("--resolution",   type=float, default=5.0,
                   help="グリッド間隔 [m]  (小=精細・重い / 大=粗い・軽い)")
    g.add_argument("--elev-max",     type=float, default=120.0,
                   help="最大高低差 [m]  (推奨: 30-200)")
    g.add_argument("--noise-large",  type=float, default=0.0005,
                   help="大スケールノイズ周波数 (山体骨格)  小=大きな山")
    g.add_argument("--noise-mid",    type=float, default=0.0015,
                   help="中スケールノイズ周波数 (尾根・谷)")
    g.add_argument("--noise-small",  type=float, default=0.0050,
                   help="小スケールノイズ周波数 (細かい起伏)")

    # ── 農場境界 ──────────────────────────────────────────────────────────────
    g2 = p.add_argument_group("農場境界")
    g2.add_argument("--farm-size-x",        type=float, default=2000.0,
                    help="敷地 東西幅 [m]  (長方形モード, boundary-irregular=0 時に有効)")
    g2.add_argument("--farm-size-y",        type=float, default=2000.0,
                    help="敷地 南北幅 [m]  (長方形モード)")
    g2.add_argument("--farm-diameter",      type=float, default=2000.0,
                    help="不整形モード時の概略直径 [m]  (boundary-irregular>0 時に有効)")
    g2.add_argument("--boundary-verts",     type=int,   default=10,
                    help="不整形モード: 境界ポリゴン頂点数")
    g2.add_argument("--boundary-irregular", type=float, default=0.0,
                    help="0=長方形(デフォルト) / 0より大=不整形ポリゴン (最大1.0)")
    g2.add_argument("--boundary-spiky",     type=float, default=0.15,
                    help="不整形モード: 凹凸深さ 0=なめらか 1=ギザギザ")

    # ── 建物 ──────────────────────────────────────────────────────────────────
    g3 = p.add_argument_group("建物")
    g3.add_argument("--n-buildings",    type=int,   default=6,
                    help="建物棟数 (推奨: 3-10)")
    g3.add_argument("--cluster-radius", type=float, default=80.0,
                    help="建物クラスタ半径 [m]")
    g3.add_argument("--building-gap",   type=float, default=6.0,
                    help="建物間最小距離 [m]")

    # ── 森林 ──────────────────────────────────────────────────────────────────
    g4 = p.add_argument_group("森林")
    g4.add_argument("--n-forests",       type=int,   default=3,
                    help="森林パッチ数")
    g4.add_argument("--forest-r-min",    type=float, default=80.0,
                    help="森林パッチ最小半径 [m]")
    g4.add_argument("--forest-r-max",    type=float, default=250.0,
                    help="森林パッチ最大半径 [m]")

    return p.parse_args()


def main():
    args = parse_args()

    config = SceneConfig(
        terrain_size_x        = args.terrain_x,
        terrain_size_y        = args.terrain_y,
        terrain_resolution    = args.resolution,
        elevation_max         = args.elev_max,
        noise_layers=[
            {"scale": args.noise_large, "octaves": 4, "weight": 1.00},
            {"scale": args.noise_mid,   "octaves": 3, "weight": 0.40},
            {"scale": args.noise_small, "octaves": 2, "weight": 0.15},
        ],
        farm_size_x           = args.farm_size_x,
        farm_size_y           = args.farm_size_y,
        farm_diameter         = args.farm_diameter,
        boundary_n_vertices   = args.boundary_verts,
        boundary_irregularity = args.boundary_irregular,
        boundary_spikiness    = args.boundary_spiky,
        n_buildings           = args.n_buildings,
        cluster_radius        = args.cluster_radius,
        building_gap          = args.building_gap,
        n_forest_patches      = args.n_forests,
        forest_radius_range   = (args.forest_r_min, args.forest_r_max),
    )

    result = generate_scene(
        case_name = args.case,
        config    = config,
        seed      = args.seed,
    )

    print()
    print("=== 生成完了 ===")
    print(f"  地形メッシュ  : {result['terrain_ply']}")
    print(f"  農場境界+建物 : {result['boundary_json']}")
    print(f"  確認図        : {result['preview_png']}")
    print(f"  建物 {result['n_buildings']} 棟  森林 {result['n_forests']} パッチ")
    print()
    print(f'次: pipeline.ipynb の CASE_NAME = "{args.case}" で実行')


if __name__ == "__main__":
    main()
