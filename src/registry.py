import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import RocCurveDisplay, roc_auc_score


def setup_mlflow(tracking_uri, artifact_root, experiment_name):
    if tracking_uri.startswith("sqlite:///"):
        os.makedirs(os.path.dirname(tracking_uri.replace("sqlite:///", "")), exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow → {tracking_uri}  |  experimento: {experiment_name}")


def log_run(result, par_meta, X_test, y_test, extra_tags=None):
    tags = {
        "equipo":   "mlops-urp",
        "framework": "SDLF v2.0",
        "gpu_used":  str(par_meta.get("gpu_used", False)),
        "backend":   par_meta.get("backend", "loky"),
    }
    if extra_tags:
        tags.update(extra_tags)

    with mlflow.start_run(run_name=result["name"]) as run:
        mlflow.set_tags(tags)

        params_log = {k: str(v) for k, v in result["params"].items()}
        params_log["n_features"] = str(X_test.shape[1])
        mlflow.log_params(params_log)

        mlflow.log_metrics({
            "auc_cv":        result["auc_cv"],
            "auc_train":     result["auc_train"],
            "auc_test":      result["auc_test"],
            "decay_pct":     result["decay_pct"],
            "elapsed_s":     result["elapsed_s"],
            "speedup_cpu":   par_meta.get("speedup", 0),
            "t_parallel_s":  par_meta.get("t_parallel_s", 0),
            "t_sequential_s": par_meta.get("t_sequential_s", 0),
        })

        fig, ax = plt.subplots(figsize=(5, 4))
        try:
            RocCurveDisplay.from_estimator(
                result["model"], X_test, y_test, ax=ax, name=result["name"]
            )
        except Exception:
            pass
        ax.set_title(f"Curva ROC — {result['name']}")
        fig_path = f"/tmp/roc_{result['name']}.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        mlflow.log_artifact(fig_path, artifact_path="graficos")

        mlflow.sklearn.log_model(result["model"], name="modelo", serialization_format="cloudpickle")

        run_id = run.info.run_id
        print(f"  [{result['name']}]  auc_test={result['auc_test']}  run_id={run_id[:8]}")
        return run_id


def log_all_runs(results, par_meta, X_test, y_test, extra_tags=None):
    run_ids = []
    for r in results:
        rid = log_run(r, par_meta, X_test, y_test, extra_tags)
        run_ids.append((r["name"], rid))
    return run_ids


def get_best_run(experiment_name, metric="auc_test"):
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
    )
    if not runs:
        return None

    best = runs[0]
    print(f"Mejor run: {best.info.run_name}  {metric}={best.data.metrics.get(metric)}")
    return best


def register_champion(run_id, model_name, alias="champion"):
    """Registra el modelo del run en el Model Registry y le asigna el alias champion."""
    mv = mlflow.register_model(f"runs:/{run_id}/modelo", model_name)
    MlflowClient().set_registered_model_alias(model_name, alias, mv.version)
    print(f"Registrado {model_name} v{mv.version} con alias @{alias}")
    return mv.version


def load_champion(model_name, alias="champion"):
    """Carga el modelo con alias champion. Devuelve (modelo, versión, run_id)."""
    mv = MlflowClient().get_model_version_by_alias(model_name, alias)
    model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
    print(f"Champion cargado: {model_name} v{mv.version} (run {mv.run_id[:8]})")
    return model, mv.version, mv.run_id
