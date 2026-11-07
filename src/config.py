"""Lectura de la configuración y rutas del proyecto."""
import os

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_config():
    with open(os.path.join(PROJECT_ROOT, "config", "pipeline_config.yml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(rel):
    return os.path.join(PROJECT_ROOT, rel)


def raw_path(cfg, period):
    return path(f"{cfg['data']['raw_dir']}/p{period}_extrac.csv")


def inference_dir(cfg, period):
    d = path(f"{cfg['data']['inference_dir']}/p{period}")
    os.makedirs(d, exist_ok=True)
    return d


def monitoring_dir(cfg, period=None):
    d = path(cfg["data"]["monitoring_dir"] + (f"/p{period}" if period is not None else ""))
    os.makedirs(d, exist_ok=True)
    return d


def tracking_uri(cfg):
    """MLFLOW_TRACKING_URI (docker compose) o SQLite local dentro del proyecto."""
    env = os.environ.get("MLFLOW_TRACKING_URI")
    if env:
        return env
    rel = cfg["mlflow"]["tracking_uri"].replace("sqlite:///", "")
    return "sqlite:///" + path(rel)


def setup_mlflow(cfg, experiment):
    import mlflow
    uri = tracking_uri(cfg)
    if uri.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(uri.replace("sqlite:///", "")), exist_ok=True)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri
