FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Taipei
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    nano \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces/vrb

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}

RUN pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install \
    einops==0.6.1 \
    matplotlib==3.7.1 \
    numpy==1.24.2 \
    opencv-python==4.7.0.72 \
    Pillow==9.3.0 \
    scikit-learn==1.2.2

CMD ["/bin/bash"]