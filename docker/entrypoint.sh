#!/usr/bin/env bash
# コンテナ内でポート 8888 が使用中なら 8889 にフォールバック
PORT=${JUPYTER_PORT:-8888}

if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
    PORT=8889
fi

echo "[entrypoint] Jupyter Lab starting on port ${PORT} ..."

exec jupyter lab \
    --ip=0.0.0.0 \
    --port="${PORT}" \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password=''
