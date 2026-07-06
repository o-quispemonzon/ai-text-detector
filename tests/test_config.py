"""Tests de configuración: el YAML del proyecto es válido y completo."""

from src.utils.config import PROJECT_ROOT, load_config, resolve_path


def test_data_config_loads_and_has_required_keys():
    cfg = load_config("configs/data.yaml")
    assert {"datasets", "paths", "split"} <= cfg.keys()
    for ds in cfg["datasets"]:
        assert {"name", "slug"} <= ds.keys()
        assert "/" in ds["slug"], "el slug de Kaggle debe ser 'owner/dataset'"


def test_split_config_is_sane():
    split = load_config("configs/data.yaml")["split"]
    assert 0 < split["test_size"] < 0.5
    assert 0 < split["val_size"] < 0.5
    assert isinstance(split["seed"], int)


def test_resolve_path_is_relative_to_project_root():
    assert resolve_path("configs/data.yaml") == PROJECT_ROOT / "configs" / "data.yaml"
    assert resolve_path("configs/data.yaml").exists()
