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
- Container disk: 40-50 GB for Qwen3.5-9B; resize to roughly 200-250 GB
  when you move to DeepSeek-V4-Flash
- Persistent volume: mount at `/workspace` and make it as small as your code
  and irreplaceable experiment output allow
- TCP port: `22` for VS Code Remote SSH
- Optional HTTP port: `8888` for JupyterLab

The RunPod base image starts SSH/Jupyter. Follow the SSH command shown in the
Pod's **Connect** panel from VS Code's Remote-SSH extension.

Python packages live in the image. Put code, notebooks, and experiment output
under `/workspace`; only those files survive a Pod stop. By default, Hugging
Face and Torch caches live on the disposable container disk and will be
downloaded again. If you decide model re-download time matters, set `HF_HOME`
to `/workspace/.cache/huggingface` in the RunPod template and allow roughly
25 GB more persistent space.

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
