"""
farm_boundary.json の森林パッチ(forests)を考慮した tree.glb 大量配置。
GenesisRenderer(学習用レンダリング)と view_scene.py(閲覧用ビューア)の
両方から呼ばれる共通ロジック。shapely 依存は増やさない (円判定はnumpyの距離計算で足りる)。
"""
import json
import numpy as np
from pathlib import Path


def load_farm_boundary(path) -> dict | None:
    """farm_boundary.json を素のJSONとして読み込む。存在しなければ None。"""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def sample_tree_positions(verts: np.ndarray, forests, n_trees: int = 2000,
                           seed: int = 42) -> np.ndarray:
    """
    地形頂点から木の配置位置をサンプリングする。

    forests が None/空リストの場合は地形全体から一様サンプリングする
    (farm_boundary.json が無いケースとの完全後方互換)。
    forests が指定されている場合は各パッチ(中心+半径の円)内の頂点だけから
    サンプリングし、本数はパッチ面積 (pi * radius^2) に比例して配分する。

    Returns:
        (N, 3) ndarray の頂点座標 (N は n_trees 以下になることがある:
        半径内に頂点が無いパッチはスキップされるため)
    """
    rng = np.random.default_rng(seed)

    if not forests:
        indices = rng.integers(0, len(verts), size=n_trees)
        return verts[indices]

    xy = verts[:, :2]
    areas = np.array([np.pi * f["radius"] ** 2 for f in forests])
    counts = np.floor(n_trees * areas / areas.sum()).astype(int)
    counts[np.argmax(areas)] += n_trees - counts.sum()  # 端数は最大パッチに寄せる

    picked = []
    for forest, n in zip(forests, counts):
        if n <= 0:
            continue
        center = np.array(forest["center"], dtype=float)
        dists = np.linalg.norm(xy - center, axis=1)
        in_patch = np.where(dists <= forest["radius"])[0]
        if len(in_patch) == 0:
            print(f"    警告: 森林パッチ {forest.get('id')} 内に地形頂点が無いためスキップ")
            continue
        idx = rng.integers(0, len(in_patch), size=n)
        picked.append(verts[in_patch[idx]])

    if not picked:
        return np.empty((0, 3))
    return np.vstack(picked)


def place_tree_entities(scene, tree_glb, positions: np.ndarray,
                         rng: np.random.Generator, use_proxy: bool = False,
                         scale_range=(0.6, 3.0), z_offset: float = 0.2,
                         proxy_radius: float = 0.25,
                         proxy_height_range=(2.0, 5.0)) -> int:
    """
    サンプリング済みの位置に木のエンティティを配置する。

    use_proxy=False: 実際の tree.glb (gs.morphs.Mesh) を1本ずつ配置する
                      (学習用レンダリング向け、高フィデリティ)。
                      tree_glb が None/未存在の場合は何も配置しない。
    use_proxy=True : 軽量な gs.morphs.Cylinder を木の代役として配置する
                      (対話ビューア向け、高速)。tree_glb は不要。

    Returns:
        配置したエンティティ数。
    """
    import genesis as gs

    if use_proxy:
        for pos in positions:
            height = float(rng.uniform(*proxy_height_range))
            scene.add_entity(
                gs.morphs.Cylinder(
                    pos=(float(pos[0]), float(pos[1]), float(pos[2]) + height / 2),
                    height=height,
                    radius=proxy_radius,
                    fixed=True,
                ),
                surface=gs.surfaces.Default(color=(0.15, 0.45, 0.15, 1.0)),
            )
        return len(positions)

    if not tree_glb or not Path(tree_glb).exists():
        return 0

    for pos in positions:
        scale = float(rng.uniform(*scale_range))
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(tree_glb),
                fixed=True,
                pos=[float(pos[0]), float(pos[1]), float(pos[2]) + z_offset],
                scale=scale,
            ),
        )
    return len(positions)
