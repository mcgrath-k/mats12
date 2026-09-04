# RunPod mechanistic-interpretability image

This image is set up for VS Code Remote SSH, TransformerLens, Anthropic's
`jlens`, and Camila Blank's published J-lens/R-lens checkpoints. It targets
Qwen3.5-9B first and keeps the package stack suitable for later direct
Transformers work with DeepSeek-V4-Flash.

## Build

Push to `main` or run **Build RunPod image** from the GitHub Actions tab. The
workflow publishes:

```text
ghcr.io/<your-github-user>/mats12:latest
ghcr.io/<your-github-user>/mats12:v<build-number>
ghcr.io/<your-github-user>/mats12:<commit-sha>
```

When the build finishes, its GitHub Actions summary shows the exact versioned
address to paste into RunPod's **Container Image** field. Build numbers increase
automatically (`v1`, `v2`, `v3`, and so on). Make the GHCR package public, or
configure registry credentials in the RunPod template.

## RunPod template

- Container image: the GHCR image above
- Additional GPU filters: CUDA 12.8, 12.9, or 13.0 (the image requires 12.8+)
- Container disk: 40-50 GB is enough; model weights do not live on it
- Persistent network volume: mount at `/workspace`. It holds code, experiment
  output, and the Hugging Face cache, so size it for the models you use:
  about 25 GB for Qwen3.5-9B, 160 GB more for DeepSeek-V4-Flash
- TCP port: `22` initially, as a fallback until Tailscale is confirmed
- Optional HTTP port: `8888` for JupyterLab

Add a non-ephemeral RunPod secret `TS_AUTHKEY` to the template as an environment
variable. You can optionally set `TS_HOSTNAME`; it defaults to `mats12`.

The image starts Tailscale in userspace mode and keeps its device identity in
`/workspace/.tailscale`. On your Mac, install and sign in to Tailscale with the
same account. Then add this to `~/.ssh/config` for VS Code Remote SSH:

```ssh-config
Host mats12
    HostName mats12
    User root
```

After the first successful connection, remove or revoke `TS_AUTHKEY` and restart
the Pod. Its saved identity reconnects without the key. Keep RunPod TCP port 22
until this has worked once; afterward you can remove it from the template and
use only private Tailscale SSH. RunPod's normal SSH and Jupyter startup remains
available as a fallback.

Python packages live in the image. Put code, notebooks, and experiment output
under `/workspace`; only those files survive a Pod stop. The image sets
`HF_HOME=/workspace/.cache/huggingface`, so model weights are downloaded once
and reused across Pods. The Torch cache stays on the disposable container disk.

## First start

Open the Pod's **Connect** panel, follow its JupyterLab link, then open a
Jupyter terminal and get this repo:

```bash
cd /workspace
git clone https://github.com/mcgrath-k/mats12.git
```

On later starts, use `git -C /workspace/mats12 pull` instead. In Jupyter's file
browser, open `mats12/notebooks/qwen35_9b_lenses.ipynb` and run all cells. The
notebook downloads the BF16 model and Camila Blank's two checkpoints, wraps the
Qwen text decoder with the official `jlens` adapter, and prints a layer-by-layer
J-lens/R-lens smoke test.

Useful gotchas:

- RunPod clears the container disk on stop, while `/workspace` volume storage
  survives and remains billable. Commit/push code as a second backup. If
  everything important is in GitHub or copied elsewhere, terminate the Pod
  instead of stopping it to avoid retained Pod-volume charges.
- Camila's Qwen3.5-9B J/R lenses are matched checkpoints with about 1.04 GB per
  file. R-lens is a checkpoint, not a separate Python package. The reference
  `jlens` package is installed for fitting and lens operations.
- Published lenses use the model's raw Hugging Face activation basis. Load them
  against the Hugging Face model as the included notebook does.
- Qwen3.5 is a hybrid Gated DeltaNet/full-attention model. This image includes
  FLA and `causal-conv1d`; without them Transformers falls back to a slower,
  more memory-hungry implementation.
- DeepSeek-V4-Flash is roughly 284B total parameters and needs a multi-GPU
  setup. The Python stack can load it through Transformers, but renting one
  ordinary single-GPU Pod will not be enough for serious work with it.
