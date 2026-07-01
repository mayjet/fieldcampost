#!/usr/bin/env bash
# docker/requirements.txt 相当 (torch/gsplatを除く、CUDA専用のため) の venv を
# プロジェクトルートの .venv に作成する (Mac向け)。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

uv venv --python 3.11 .venv
uv pip install -r build/requirements.txt
