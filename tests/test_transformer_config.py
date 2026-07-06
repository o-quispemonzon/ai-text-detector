"""Tests ligeros del enfoque transformer (sin cargar torch: apto para CI)."""

from src.utils.config import load_config


def test_transformer_config_is_sane():
    cfg = load_config("configs/model_transformer.yaml")
    assert cfg["model_name"].startswith("microsoft/deberta")
    assert 128 <= cfg["max_length"] <= 512
    tr = cfg["training"]
    assert tr["batch_size"] * tr["grad_accum"] >= 16, "batch efectivo demasiado chico"
    assert 0 < float(tr["learning_rate"]) < 1e-3


def test_transformer_and_classic_share_seed():
    """Misma semilla en ambos enfoques: los splits submuestreados coinciden."""
    t = load_config("configs/model_transformer.yaml")
    c = load_config("configs/model_classic.yaml")
    assert t["seed"] == c["seed"]
