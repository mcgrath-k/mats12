M pytorch/pytorch:2.13.0-cuda12.6-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    tmux \
    htop \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Pin the things whose versions actually matter.
RUN pip install \
    "transformers==5.13.0" \
    "accelerate==1.14.0" \
    "huggingface-hub==1.23.0" \
    safetensors \
    ipykernel \
    jupyterlab

RUN pip install \
    "git+https://github.com/anthropics/jacobian-lens.git"

WORKDIR /workspace

