"""
Genesis レンダリング → 3DGS学習 → 点群抽出 → 精度評価 の一括実行。
自動精度改善ループ付き。
"""
import argparse
from pathlib import Path
from scripts.genesis_renderer    import GenesisRenderer
from scripts.train_3dgs          import train_3dgs
from scripts.extract_pointcloud  import extract_pointcloud
from scripts.evaluate_accuracy   import evaluate_accuracy, AUTO_IMPROVE_THRESHOLD_CM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",         required=True)
    parser.add_argument("--skip-render",  action="store_true")
    parser.add_argument("--skip-train",   action="store_true")
    parser.add_argument("--auto-improve", action="store_true")
    parser.add_argument("--max-loops",    type=int, default=3)
    parser.add_argument("--device",       default="auto",
                         choices=("auto", "gpu", "cuda", "metal", "amdgpu", "cpu"),
                         help="Genesisバックエンド (auto=CUDA/Metal/CPUを自動選択)")
    args = parser.parse_args()

    case_dir  = Path("cases") / args.case
    poses_json = case_dir / "camera_poses.json"
    terrain_ply = case_dir / "terrain.ply"
    tree_glb    = Path("assets/tree.glb")
    colmap_dir  = case_dir / "colmap_input"
    images_dir  = colmap_dir / "images"  # gsplat は --data_dir 直下に images/ を要求
    model_dir   = case_dir / "gaussian_model"
    recon_ply   = case_dir / "recon.ply"
    gt_ply      = case_dir / "terrain_gt.ply"

    for loop in range(args.max_loops if args.auto_improve else 1):
        print(f"\n{'='*60}")
        print(f"ループ {loop+1}/{args.max_loops if args.auto_improve else 1}")
        print(f"{'='*60}")

        # Step A: Genesis レンダリング
        if not args.skip_render or loop > 0:
            print("\n[Step A] Genesis レンダリング")
            renderer = GenesisRenderer(
                terrain_ply=terrain_ply,
                tree_glb=tree_glb if tree_glb.exists() else None,
                poses_json=poses_json,
                output_dir=images_dir,
                device=args.device,
            )
            renderer.render_all()

        # Step B: 3DGS トレーニング
        if not args.skip_train or loop > 0:
            print("\n[Step B] 3DGS トレーニング")
            train_3dgs(
                colmap_dir=colmap_dir,
                output_dir=model_dir,
                iterations=30000,
            )

        # Step C: 点群抽出
        print("\n[Step C] 点群抽出")
        extract_pointcloud(
            model_dir=model_dir,
            output_ply=recon_ply,
            opacity_threshold=0.1,
        )

        # Step D: 精度評価
        print("\n[Step D] 精度評価")
        results = evaluate_accuracy(
            gt_ply=gt_ply,
            recon_ply=recon_ply,
            case_dir=case_dir,
        )

        dsm_rmse = results.get("dsm_rmse_cm", 999)
        print(f"\nDSM RMSE = {dsm_rmse:.2f} cm  (目標 < {AUTO_IMPROVE_THRESHOLD_CM} cm)")

        # Step E: 自動精度改善判定
        if not args.auto_improve or dsm_rmse < AUTO_IMPROVE_THRESHOLD_CM:
            print("\n✓ 目標精度達成 (または自動改善なし)")
            break

        print(f"\n[Step E] 精度改善: 誤差の大きい領域にカメラ追加...")
        from scripts.evaluate_accuracy import add_cameras_to_error_regions
        added = add_cameras_to_error_regions(
            case_dir=case_dir,
            poses_json=poses_json,
            terrain_ply=terrain_ply,
            error_map_ply=case_dir / "error_map_pts.ply",
        )
        if added == 0:
            print("追加可能なカメラ位置が見つかりませんでした")
            break
        print(f"カメラ {added} 台追加 → 再レンダリング")

    print("\n=== 完了 ===")
    import json
    with open(case_dir / "evaluation_results.json") as f:
        print(json.dumps(json.load(f), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
