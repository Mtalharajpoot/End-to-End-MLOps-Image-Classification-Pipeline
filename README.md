# VisionOps 🧠⚙️

[![CI](https://github.com/<your-username>/visionops/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/visionops/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/visionops/blob/main/notebooks/colab_demo.ipynb)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**An end-to-end ML engineering pipeline** — config-driven training, experiment tracking and
hyper-parameter sweeps, ONNX export with a numeric parity check, a lean Docker serving image, and CI.
Built as a hands-on implementation of the tooling covered in
[Full Stack Deep Learning 2022 — Lecture 2: Development Infrastructure & Tooling](https://fullstackdeeplearning.com/course/2022/).

![Architecture](assets/architecture.png)

## Why this repo exists

Most ML tutorials stop at `model.fit()`. In industry the model is the small part: the value is in a
**reproducible, tracked, tested and deployable** workflow. VisionOps is a compact reference for that
workflow, using image classification (CIFAR-10) as a simple, fast-to-train task.

## Features

- **Config-driven runs** — typed dataclass config + YAML; every field overridable from the CLI (`--lr 3e-4`).
- **PyTorch Lightning + `timm`** — swap backbones with `--backbone convnext_tiny`; mixed precision; early stopping; best-checkpoint selection.
- **Experiment tracking** — CSV logs by default, [Weights & Biases](https://wandb.ai) with `--logger wandb`; Bayesian sweeps with Hyperband early-stopping (`sweeps/sweep.yaml`).
- **Scale-out ready** — `--devices 4 --strategy ddp` (or `fsdp`, `deepspeed_stage_2`) with no code changes.
- **ONNX export** — dynamic batch axis + automatic PyTorch-vs-ONNX Runtime parity check.
- **Lean serving** — Streamlit demo running on ONNX Runtime in a Docker image **without PyTorch**.
- **Engineering hygiene** — `ruff` lint/format, pre-commit, pytest (unit tests + 1-epoch end-to-end smoke test), GitHub Actions CI.

## From lecture to code

| Lecture topic (FSDL 2022, Lecture 2) | Where it lives in this repo |
|---|---|
| Python, editors, linters, type hints (slides 9–12) | `ruff`, `.pre-commit-config.yaml`, type-annotated `src/` |
| Notebooks vs. modules — "import modules in notebooks" (16–18) | logic in `src/visionops/`, thin demo in `notebooks/colab_demo.ipynb` |
| Streamlit (19) | `app/streamlit_app.py` |
| Environment management (20) | `pyproject.toml` extras, `Dockerfile`, `Dockerfile.train` |
| PyTorch + PyTorch Lightning (26–28) | `model.py`, `train.py` |
| Model zoos: Hugging Face / timm (35–36) | `timm` backbones, pretrained weights |
| ONNX interoperability (34) | `export_onnx.py` |
| Distributed training: data / sharded parallelism (37–58) | `--devices`, `--strategy` flags |
| Compute: GPUs, 16-bit precision (65–80) | `precision: 16-mixed` (falls back to fp32 on CPU) |
| Resource management: Docker (93) | `Dockerfile`, `Dockerfile.train` |
| Experiment management: W&B + Sweeps (111–117) | `--logger wandb`, `sweeps/sweep.yaml` |

## Quick start

### Option A — Google Colab (free GPU, no setup)
Click the **Open in Colab** badge above, set the runtime to **GPU**, and run all cells.

### Option B — Local

```bash
git clone https://github.com/<your-username>/visionops.git && cd visionops
python -m venv .venv && source .venv/bin/activate
make install            # pip install -e ".[train,tracking,app,dev]"

make smoke              # 1-epoch offline run, ~1 min on CPU (no downloads)
make train              # real training on CIFAR-10 (GPU recommended)
```

Outputs land in `outputs/`: logs, the best checkpoint, `model.onnx`, `model_meta.json`, `results.json`.

### Override anything from the CLI

```bash
python -m visionops.train --backbone resnet50 --lr 3e-4 --max_epochs 10 --batch_size 256
python -m visionops.train --logger wandb --run_name baseline            # after `wandb login`
python -m visionops.train --devices 4 --strategy ddp                    # multi-GPU
```

### Hyper-parameter sweep (W&B)

```bash
wandb login
wandb sweep sweeps/sweep.yaml     # prints a sweep ID
wandb agent <entity>/<project>/<sweep-id>
```

### Serve the demo

```bash
cp outputs/model.onnx outputs/model_meta.json models/
make app                                   # local Streamlit
# or, containerised (no PyTorch inside the image):
make docker-build && make docker-run       # http://localhost:8501
```

### GPU training in Docker

```bash
docker build -f Dockerfile.train -t visionops-train .
docker run --gpus all -v $PWD/outputs:/workspace/outputs visionops-train --max_epochs 10
```

## Results

> Fill this in with **your own** run (numbers depend on seed, GPU and epochs). They are written to `outputs/results.json`.

| Backbone | Image size | Epochs | Precision | Test accuracy | Train time (T4) |
|---|---|---|---|---|---|
| resnet18 (pretrained) | 64 | 5 | 16-mixed | _your result_ | _your result_ |

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) for intended use and limitations.

## Project structure

```
visionops/
├── src/visionops/
│   ├── config.py        # typed config: YAML + CLI overrides
│   ├── data.py          # LightningDataModule (CIFAR-10 / offline fake data)
│   ├── model.py         # LightningModule around any timm backbone
│   ├── train.py         # training entry point (also importable: run(cfg))
│   ├── export_onnx.py   # ONNX export + parity check
│   └── inference.py     # torch-free ONNX Runtime predictor
├── app/streamlit_app.py # demo UI
├── configs/             # default.yaml, fast_dev.yaml (CI)
├── sweeps/sweep.yaml    # W&B sweep definition
├── tests/               # config, data, model, inference, end-to-end smoke test
├── notebooks/colab_demo.ipynb
├── Dockerfile           # lean serving image (ONNX Runtime + Streamlit)
├── Dockerfile.train     # GPU training image
├── Makefile · pyproject.toml · .pre-commit-config.yaml
└── .github/workflows/ci.yml
```

## Roadmap

- [ ] Add a `/predict` FastAPI endpoint next to the Streamlit demo
- [ ] Data versioning with DVC and a custom (non-CIFAR) dataset
- [ ] Model registry + automatic ONNX upload from CI (W&B Artifacts)
- [ ] Monitoring: log prediction confidence in the served app to detect drift

## License

MIT — see [LICENSE](LICENSE). Course content © Full Stack Deep Learning; this repo is an independent learning project.
