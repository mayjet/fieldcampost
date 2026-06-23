"""
3DGSトレーニング: gsplat (primary) → inria gaussian-splatting (fallback)
"""
import sys, subprocess
from pathlib import Path

def train_3dgs(colmap_dir: Path, output_dir: Path, iterations: int = 30000):
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = _find_gsplat_trainer()

    if trainer:
        print(f"gsplat SimpleTrainer: {trainer}")
        cmd = [
            sys.executable, trainer,
            "--data_dir",    str(colmap_dir),
            "--result_dir",  str(output_dir),
            "--max_steps",   str(iterations),
            "--data_factor", "1",
        ]
    else:
        print("WARNING: gsplat trainer not found → trying inria train.py")
        train_py = Path("gaussian-splatting/train.py")
        if not train_py.exists():
            raise RuntimeError(
                "gsplat も inria gaussian-splatting も見つかりません。\n"
                "pip install gsplat または gaussian-splatting を clone してください。"
            )
        cmd = [
            sys.executable, str(train_py),
            "-s", str(colmap_dir),
            "-m", str(output_dir),
            "--iterations", str(iterations),
        ]

    print(f"Training 3DGS ({iterations} iterations)...")
    subprocess.run(cmd, check=True)
    print(f"Training done: {output_dir}/")


def _find_gsplat_trainer() -> str | None:
    try:
        import gsplat
        gsplat_dir = Path(gsplat.__file__).parent.parent
        t = gsplat_dir / "examples" / "simple_trainer.py"
        if t.exists():
            return str(t)
    except ImportError:
        pass

    result = subprocess.run(["pip", "show", "gsplat"],
                            capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Location:"):
            loc = Path(line.split(":", 1)[1].strip())
            t   = loc / "examples" / "simple_trainer.py"
            if t.exists():
                return str(t)
    return None
