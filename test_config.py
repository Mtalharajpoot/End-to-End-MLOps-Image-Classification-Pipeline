import pytest

from visionops.config import Config, load_config


def test_defaults():
    cfg = load_config()
    assert cfg == Config()
    assert cfg.backbone == "resnet18"


def test_yaml_then_cli_override(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("lr: 0.01\nmax_epochs: 3\npretrained: false\n")
    cfg = load_config(f, {"lr": "3e-4", "run_name": None})
    assert cfg.lr == pytest.approx(3e-4)  # CLI beats YAML, string is cast to float
    assert cfg.max_epochs == 3
    assert cfg.pretrained is False


def test_bool_and_int_coercion():
    cfg = load_config(overrides={"pretrained": "false", "batch_size": "64", "export_onnx": "true"})
    assert cfg.pretrained is False
    assert cfg.batch_size == 64
    assert cfg.export_onnx is True


def test_unknown_key_rejected():
    with pytest.raises(ValueError, match="Unknown config keys"):
        load_config(overrides={"learning_rate": "0.1"})


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")
