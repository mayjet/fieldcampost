FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

# ── System packages ───────────────────────────────────────────────────────────
# Ubuntu 22.04 の標準 python3 (3.10) を使用。3.11 は apt の dist-packages
# （numpy 等）と ABI が合わず ImportError を起こすため避ける。
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-venv python3-pip \
    git wget curl build-essential cmake ninja-build \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    libegl1 libgles2 libopengl0 libglvnd0 \
    libx11-6 libxext6 libxrender1 libxi6 libxfixes3 libxxf86vm1 \
    libgdal-dev gdal-bin \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install uv

# python コマンドを python3 に向ける
RUN ln -sf /usr/bin/python3 /usr/local/bin/python

# ── Blender 4.1 (headless) ────────────────────────────────────────────────────
ARG BLENDER_VERSION=4.1.1
RUN wget -q \
    "https://download.blender.org/release/Blender4.1/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
    -O /tmp/blender.tar.xz \
    && tar -xf /tmp/blender.tar.xz -C /opt \
    && ln -s /opt/blender-${BLENDER_VERSION}-linux-x64/blender /usr/local/bin/blender \
    && rm /tmp/blender.tar.xz

RUN BPYTHON=$(find /opt/blender-${BLENDER_VERSION}-linux-x64/ \
        -name "python3*" -executable -path "*/bin/*" | head -1) \
    && if [ -n "$BPYTHON" ] && "$BPYTHON" --version > /dev/null 2>&1; then \
        "$BPYTHON" -m ensurepip --upgrade \
        && "$BPYTHON" -m pip install --upgrade trimesh numpy; \
    else \
        echo "WARNING: Blender bundled Python not executable — skipping"; \
    fi

WORKDIR /workspace

# ── Python依存 ────────────────────────────────────────────────────────────────
COPY docker/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# open3d: x86 wheel
RUN uv pip install --system --no-cache open3d

# PyTorch (CUDA 12.4)
RUN uv pip install --system --no-cache \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

# gsplat (primary 3DGS backend)
RUN uv pip install --system --no-cache gsplat

# Genesis
RUN uv pip install --system --no-cache genesis-world \
    || (git clone --depth 1 https://github.com/Genesis-Embodied-AI/Genesis.git /tmp/genesis \
        && uv pip install --system --no-cache /tmp/genesis \
        && rm -rf /tmp/genesis)

# I/O
RUN uv pip install --system --no-cache imageio plyfile

# PYTHONPATH: /workspace を常にモジュール検索パスに含める
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Jupyter
RUN python3 -m ipykernel install --sys-prefix \
    --name py310 --display-name "Python 3.10 (x86)"

# Jupyter Lab 設定 (root_dir・認証なし・自動起動向け)
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.root_dir = '/workspace'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> /root/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_lab_config.py

COPY . .
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
