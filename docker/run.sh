#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="field-campole-optimizer"
CONTAINER_NAME="field-campole-optimizer"

# ホスト側ポートチェック: 8888 が使用中なら 8889 にフォールバック
PORT=8888
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    PORT=8889
    echo "[run.sh] port 8888 in use — using 8889"
fi

docker build \
    -t "$IMAGE_NAME" \
    -f "$SCRIPT_DIR/Dockerfile" \
    "$PROJECT_ROOT"

xhost +local:docker 2>/dev/null || true
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=8g \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}" \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e JUPYTER_PORT="${PORT}" \
    -e PYTHONPATH=/workspace \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority:ro" \
    -v "$PROJECT_ROOT/cases":/workspace/cases \
    -v "$PROJECT_ROOT/assets":/workspace/assets \
    -v "$PROJECT_ROOT":/workspace \
    --device /dev/dri:/dev/dri \
    --network host \
    "$IMAGE_NAME"

# Jupyter 起動を待ってURLを表示 (最大60秒)
HOST_IP=$(hostname -I | awk '{print $1}')
echo "Jupyter Lab 起動待ち..."
for i in $(seq 1 30); do
    if docker logs "$CONTAINER_NAME" 2>&1 | grep -q "ServerApp.url"; then
        echo ""
        echo "=== コンテナ起動完了 ==="
        echo "【Jupyter Lab】 http://${HOST_IP}:${PORT}"
        echo "【ターミナル】  docker exec -it ${CONTAINER_NAME} bash"
        exit 0
    fi
    sleep 2
done
echo "WARNING: Jupyter の起動確認がタイムアウトしました (60秒)"
echo "【ブラウザ】 http://${HOST_IP}:${PORT}"
echo "【ターミナル】  docker exec -it ${CONTAINER_NAME} bash"
