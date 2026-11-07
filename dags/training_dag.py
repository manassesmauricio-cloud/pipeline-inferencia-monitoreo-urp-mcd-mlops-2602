"""PIPELINE 0 (heredado del repo de entrenamiento): genera y registra el champion.

Solo se corre una vez para que la inferencia tenga un modelo con alias @champion
en el Model Registry. La lógica de HPO paralelo es la misma del repo de entrenamiento.
"""
import json
import os
import pickle
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, path, raw_path  # noqa: E402

_cfg = load_config()
TMP_DIR = "/tmp/airflow_mlops"

default_args = {
    "owner": _cfg["airflow"]["dag_owner"],
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": _cfg["airflow"]["retries"],
    "retry_delay": timedelta(minutes=_cfg["airflow"]["retry_delay_minutes"]),
}


def task_preprocess(**ctx):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.preprocessing import prepare_features

    d, t = _cfg["data"], _cfg["training"]
    periods = d["training_periods"]
    n_per = d["max_rows_train"] // len(periods)
    frames = []
    for p in periods:
        df = pd.read_csv(raw_path(_cfg, p), low_memory=False)
        frames.append(df.sample(n=min(n_per, len(df)), random_state=t["random_state"]))
        print(f"p{p}: {len(df):,} filas -> muestra {len(frames[-1]):,}")
    df = pd.concat(frames, ignore_index=True)

    y = df[d["target_col"]].astype(int)
    X = prepare_features(df, d["drop_cols"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=t["test_size"], random_state=t["random_state"], stratify=y)

    out = path(d["processed_dir"])
    os.makedirs(out, exist_ok=True)
    pd.concat([y_tr, X_tr], axis=1).to_csv(f"{out}/train.csv", index=False)
    pd.concat([y_te, X_te], axis=1).to_csv(f"{out}/test.csv", index=False)
    with open(f"{out}/feature_columns.json", "w") as f:
        json.dump(list(X.columns), f)
    print(f"Train {X_tr.shape} | Test {X_te.shape} | tasa target {y.mean():.4f}")


def task_train(**ctx):
    import pandas as pd
    from src.training import run_parallel_hpo

    t = _cfg["training"]
    out = path(_cfg["data"]["processed_dir"])
    tr, te = pd.read_csv(f"{out}/train.csv"), pd.read_csv(f"{out}/test.csv")
    target = tr.columns[0]
    results, champion, par_meta = run_parallel_hpo(
        tr.drop(columns=[target]), tr[target], te.drop(columns=[target]), te[target],
        n_iter=t["n_iter_hpo"], cv=t["cv_folds"], n_jobs=t["n_jobs"],
        random_state=t["random_state"], decay_threshold=t["decay_threshold"],
        use_gpu=t["use_gpu"],
    )
    if champion is None:  # ninguno pasa el umbral de decaimiento: se toma el mejor AUC test
        champion = max(results, key=lambda r: r["auc_test"])
        print("Ningún modelo pasa el umbral de decay; se usa el mejor AUC test")
    os.makedirs(TMP_DIR, exist_ok=True)
    with open(f"{TMP_DIR}/training_results.pkl", "wb") as f:
        pickle.dump({"results": results, "champion": champion["name"], "par_meta": par_meta}, f)


def task_register(**ctx):
    import mlflow
    import pandas as pd
    from src.config import setup_mlflow
    from src.registry import log_all_runs, register_champion

    m = _cfg["mlflow"]
    with open(f"{TMP_DIR}/training_results.pkl", "rb") as f:
        data = pickle.load(f)
    out = path(_cfg["data"]["processed_dir"])
    te = pd.read_csv(f"{out}/test.csv")
    target = te.columns[0]
    with open(f"{out}/feature_columns.json") as f:
        feature_columns = json.load(f)

    setup_mlflow(_cfg, m["exp_training"])
    run_ids = dict(log_all_runs(data["results"], data["par_meta"],
                                te.drop(columns=[target]), te[target]))
    champ_run = run_ids[data["champion"]]
    with mlflow.start_run(run_id=champ_run):
        mlflow.log_dict(feature_columns, "feature_columns.json")
        mlflow.set_tag("rol", "champion")
    version = register_champion(champ_run, m["model_name"], m["champion_alias"])
    print(f"Champion: {data['champion']} | run {champ_run[:8]} | versión {version}")


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    description="Entrena (HPO paralelo) y registra el champion en el Model Registry",
    schedule=None,
    catchup=False,
    tags=["urp", "entrenamiento"],
) as dag:
    t1 = PythonOperator(task_id="preprocesar_datos", python_callable=task_preprocess)
    t2 = PythonOperator(task_id="entrenar_modelos", python_callable=task_train)
    t3 = PythonOperator(task_id="registrar_champion", python_callable=task_register)
    t1 >> t2 >> t3
