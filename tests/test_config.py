"""La configuración del pipeline existe y trae las claves que usan los DAG."""
from src.config import load_config


def test_config_tiene_las_secciones_principales():
    cfg = load_config()
    assert isinstance(cfg, dict)
    assert "data" in cfg and "raw_dir" in cfg["data"]
