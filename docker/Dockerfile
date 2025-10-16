FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Taipei \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_INSTALL_DIR="/usr/local/bin" \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && \ 
    apt-get install -y --no-install-recommends \
        software-properties-common \
        curl git ca-certificates gnupg build-essential wget unzip && \ 
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils && \ 
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \ 
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \ 
    curl -LsSf https://astral.sh/uv/install.sh | sh && \ 
    uv --version && \ 
    apt-mark hold python3.11 && \ 
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN uv pip install --no-cache-dir -r /workspace/requirements.txt && \
    uv pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --no-build-isolation --no-cache-dir flash-attn==2.7.4.post1

CMD ["/bin/bash"]
