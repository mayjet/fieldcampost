# field-CamSim

農場害獣監視のための3次元カメラ配置最適化 + 3DGS精度検証シミュレーション。

仮想地形 (Blender/プロシージャル生成) に対して観測点(ポール)とカメラを3次元的に最適配置し、
Genesisでレンダリングした画像から3DGSで地形を再構成、GT点群との精度を比較する。

---

## ディレクトリ構成

```
docker/             コンテナ定義 (x86 / Jetson L4T)
scripts/            コアモジュール (地形読み込み・カメラ配置最適化・レンダリング・学習・評価)
assets/             3Dアセット (tree.glb など)
cases/              シミュレーションケース毎の生成データ (ケースごとにサブディレクトリ)
pipeline.ipynb      地形読み込み〜カメラ配置〜COLMAPフォーマット出力までのNotebook
generate_case.py    地形・農場境界・建物のプロシージャル生成スクリプト
run_simulation.py   Genesisレンダリング〜3DGS学習〜精度評価の一括実行スクリプト
```

---

## セットアップ

### コンテナ起動 (Jetson / L4T)

```bash
bash docker/run_l4t.sh
```

### コンテナ起動 (x86)

```bash
bash docker/run.sh
```

起動後、Jupyter Lab が自動起動する (ポート8888、使用中なら8889にフォールバック)。
ブラウザで表示されたURLを開く。

---

## 実行手順

### 1. 地形ケースの生成 (3km四方 / 中心2km四方の農場敷地)

```bash
python generate_case.py \
    --case ranch_01 \
    --seed 42 \
    --terrain-x 3000 \
    --terrain-y 3000 \
    --resolution 5 \
    --elev-max 120 \
    --farm-size-x 2000 \
    --farm-size-y 2000 \
    --boundary-irregular 0.0 \
    --n-buildings 0 \
    --n-forests 0
```

- `--terrain-x/y`: 地形全体サイズ [m] (デフォルト3000m四方)
- `--resolution`: 地形メッシュの解像度 [m] (デフォルト5m。重い場合は10〜15に変更)
- `--farm-size-x/y`: 農場敷地サイズ [m] (中心に配置される長方形)
- `--boundary-irregular`: 0.0=長方形、>0.0=不整形ポリゴン
- `--n-buildings` / `--n-forests`: 建物・森クラスタ数 (0で生成なし)

出力: `cases/ranch_01/terrain.ply`, `farm_boundary.json`, `scene_preview.png`

### 2. Notebook 実行 (カメラ配置最適化)

```bash
jupyter lab pipeline.ipynb
```

`Cell 1` の `CASE_NAME = "ranch_01"` だけ変更すれば全セルが追従する。
Cell 1〜10 を順に実行し、`camera_poses.json` / `terrain_gt.ply` / COLMAPフォーマットを生成する。

### 3. シミュレーション実行 (Genesisレンダリング〜3DGS学習〜精度評価)

```bash
python run_simulation.py --case ranch_01 --auto-improve
```

出力: `evaluation_results.json` (chamfer_distance_cm, dsm_rmse_cm, fscore_10cm), `error_map.png`

---

## 注意事項

- 地形解像度 (`--resolution`) を5mのまま3km四方で生成すると約36万頂点・72万面のメッシュになり、
  計算機資源 (特にJetsonなどの組み込み機)によっては Notebook 実行時に処理が重くなることがある。
  重い場合は `--resolution 10` または `15` に上げて頂点数を削減すること。
- `extrinsic_4x4` は `cam→world` 規約で統一している。COLMAP変換時のみ `world→cam` に変換する。
- COLMAP のインストールは不要 (`scripts/pose_exporter.py` が直接フォーマットを生成する)。
