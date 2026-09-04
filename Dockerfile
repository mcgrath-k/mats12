FROM docker.io/tailscale/tailscale:stable AS tailscale

FROM runpod/pytorch:1.1.0-cu1281-torch291-ubuntu2404

# Model and kernel caches are deliberately disposable. Keep only code and
# experiment outputs under the persistent /workspace mount.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    XDG_CACHE_HOME=/root/.cache

# The RunPod base already starts SSH and Jupyter, which is what VS Code Remote SSH needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        htop \
        openssh-client \
        tmux \
        vim \
    && rm -rf /var/lib/apt/lists/*

COPY --from=tailscale /usr/local/bin/tailscale /usr/local/bin/tailscale
COPY --from=tailscale /usr/local/bin/tailscaled /usr/local/bin/tailscaled

COPY requirements.txt /tmp/requirements.txt
COPY requirements-deepseek-v4.txt /tmp/requirements-deepseek-v4.txt

# Qwen3.5 uses causal-conv1d plus FLA for its fast Gated DeltaNet path. The
# package publishes a wheel for this image's Python 3.12 / PyTorch 2.9 / CUDA 12.x.
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m pip install --no-cache-dir --no-build-isolation "causal-conv1d==1.7.0" \
    && python -m pip check \
    && python -c "from importlib.metadata import version; import causal_conv1d, fla, jlens, torch, transformer_lens, transformers, scikit-learn; print('lens environment OK:', torch.__version__, transformers.__version__, version('transformer-lens'))"

# Keep DeepSeek-V4's newer PyTorch stack isolated from the pinned lens stack.
RUN /usr/bin/uv venv --python /usr/bin/python3.12 /opt/deepseek-v4 \
    && /usr/bin/uv pip install --python /opt/deepseek-v4/bin/python \
        --torch-backend=cu128 \
        -r /tmp/requirements-deepseek-v4.txt setuptools wheel packaging ninja \
    && /usr/bin/uv pip install --python /opt/deepseek-v4/bin/python \
        --no-build-isolation fast_hadamard_transform \
    && /opt/deepseek-v4/bin/python -m ipykernel install \
        --prefix=/usr/local \
        --name deepseek-v4 \
        --display-name "Python (DeepSeek V4)"

COPY post_start.sh /post_start.sh
RUN chmod 0755 /post_start.sh

WORKDIR /workspace

# Keep the base image's /start.sh entrypoint/CMD: it provides SSH and Jupyter,
# then calls /post_start.sh to bring up the stable Tailscale connection.
