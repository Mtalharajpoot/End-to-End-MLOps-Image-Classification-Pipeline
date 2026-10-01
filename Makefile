.PHONY: install lint format test train smoke export app sweep docker-build docker-run

install:            ## editable install with everything
	pip install -e ".[train,tracking,app,dev]"

lint:
	ruff check . && ruff format --check .

format:
	ruff check --fix . && ruff format .

test:
	pytest -q

smoke:              ## 1-epoch offline run (no downloads) — what CI does
	python -m visionops.train --config configs/fast_dev.yaml --output_dir outputs/smoke

train:              ## real training on CIFAR-10
	python -m visionops.train --config configs/default.yaml

export:             ## usage: make export CKPT=outputs/.../best.ckpt
	python -m visionops.export_onnx --ckpt $(CKPT) --out models/model.onnx

app:
	streamlit run app/streamlit_app.py

sweep:              ## needs `wandb login`
	wandb sweep sweeps/sweep.yaml

docker-build:
	docker build -t visionops .

docker-run:
	docker run --rm -p 8501:8501 visionops
