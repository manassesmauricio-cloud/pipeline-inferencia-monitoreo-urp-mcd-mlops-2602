"""PIPELINE 1: inferencia.

validar_entrada -> cargar_champion -> preprocesar -> scoring -> posprocesar
-> registrar_mlflow -> disparar_monitoreo

Enfoque centralizado: el monitoreo es otro DAG. La última tarea solo lo dispara
(wait_for_completion=False), así la inferencia termina sin esperar al monitoreo.
Todas las tareas de inferencia corren en los núcleos reservados (monitoring.inference_cores).
"""
import json
import os
import sys
import time
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, raw_path, inference_dir  # noqa: E402
from src.cores import pin, inference_cores  # noqa: E402

_cfg = load_config()
M, D = _cfg["mlflow"], _cfg["data"]

default_args = {
    "owner": _cfg["airflow"]["dag_owner"],
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": _cfg["airflow"]["retries"],
    "retry_delay": timedelta(minutes=_cfg["airflow"]["retry_delay_minutes"]),
}


def _period(ctx):
    return int(ctx["params"]["period"])


def _start(name, ctx):
    cores = pin(inference_cores(_cfg))
    print(f"[{name}] p{_period(ctx)} | núcleos {cores}")
    return time.time()


def _load_champion():
    from src.config import setup_mlflow
    from src.registry import load_champion
    setup_mlflow(_cfg, M["exp_inference"])
    return load_champion(M["model_name"], M["champion_alias"])


# TAREAS ======================================================================

def task_validar(**ctx):
    t0 = _start("validar_entrada", ctx)
    p = raw_path(_cfg, _period(ctx))
    if not os.path.exists(p):
        raise FileNotFoundError(f"No existe {p}. Copie p{_period(ctx)}_extrac.csv del Drive en data/raw/")
    return round(time.time() - t0, 2)


def task_champion(**ctx):
    import mlflow
    t0 = _start("cargar_champion", ctx)
    _, version, run_id = _load_champion()
    cols = mlflow.artifacts.load_dict(f"runs:/{run_id}/feature_columns.json")
    model_name = mlflow.get_run(run_id).info.run_name
    meta = {"version": version, "run_id": run_id, "model": model_name, "n_features": len(cols)}
    out = inference_dir(_cfg, _period(ctx))
    with open(f"{out}/feature_columns.json", "w") as f:
        json.dump(cols, f)
    with open(f"{out}/champion.json", "w") as f:
        json.dump(meta, f)
    print(meta)
    return round(time.time() - t0, 2)


def task_prepro(**ctx):
    import pandas as pd
    from src.preprocessing import prepare_features
    t0 = _start("preprocesar", ctx)
    out = inference_dir(_cfg, _period(ctx))
    with open(f"{out}/feature_columns.json") as f:
        cols = json.load(f)
    df = pd.read_csv(raw_path(_cfg, _period(ctx)), low_memory=False)
    X = prepare_features(df, D["drop_cols"], feature_columns=cols)  # misma función del entrenamiento
    X.to_csv(f"{out}/preprocessed.csv", index=False)
    print(f"preprocessed.csv {X.shape}")
    return round(time.time() - t0, 2)


def task_scoring(**ctx):
    import pandas as pd
    from src.inference import score
    t0 = _start("scoring", ctx)
    out = inference_dir(_cfg, _period(ctx))
    model, _, _ = _load_champion()
    X = pd.read_csv(f"{out}/preprocessed.csv")
    pd.DataFrame({"score": score(model, X)}).to_csv(f"{out}/score.csv", index=False)
    print(f"score.csv {len(X):,} filas")
    return round(time.time() - t0, 2)


def task_pospro(**ctx):
    import pandas as pd
    from src.posprocessing import get_groups, save_replica
    t0 = _start("posprocesar", ctx)
    out = inference_dir(_cfg, _period(ctx))
    with open(f"{out}/champion.json") as f:
        champ = json.load(f)
    cols = ["partition", "key_value", "codunicocli", "monto", "prob_value_contact", "grp_campecs06m"]
    df = pd.read_csv(raw_path(_cfg, _period(ctx)), usecols=cols, low_memory=False)
    scores = pd.read_csv(f"{out}/score.csv")["score"].values
    df = get_groups(scores, df)
    path, rep = save_replica(df, champ["model"], _period(ctx), out)
    print(f"{os.path.basename(path)} {len(rep):,} filas")
    return round(time.time() - t0, 2)


def task_log(**ctx):
    import mlflow
    import pandas as pd
    from src.config import setup_mlflow
    p = _period(ctx)
    out = inference_dir(_cfg, p)
    with open(f"{out}/champion.json") as f:
        champ = json.load(f)
    ti = ctx["ti"]
    tiempos = {t: ti.xcom_pull(task_ids=t) for t in
               ["validar_entrada", "cargar_champion", "preprocesar", "scoring", "posprocesar"]}
    sc = pd.read_csv(f"{out}/score.csv")["score"]
    rep = pd.read_csv(f"{out}/scr_{champ['model']}_p{p}.txt", sep="|", usecols=["grupo_ejec"])

    setup_mlflow(_cfg, M["exp_inference"])
    with mlflow.start_run(run_name=f"inferencia_p{p}") as run:
        mlflow.set_tags({"periodo": f"p{p}", "model_version": champ["version"],
                         "model_run_id": champ["run_id"], "modelo": champ["model"]})
        mlflow.log_params({"period": p, "model_name": M["model_name"]})
        mlflow.log_metrics({"n_filas": len(sc), "score_medio": float(sc.mean()),
                            "score_p90": float(sc.quantile(0.9))})
        mlflow.log_metrics({f"t_{k}_s": v for k, v in tiempos.items() if v is not None})
        mlflow.log_dict(rep["grupo_ejec"].value_counts().sort_index().to_dict(),
                        "distribucion_grupo_ejec.json")
        print(f"run de inferencia {run.info.run_id} | tiempos {tiempos}")
        return run.info.run_id


# DAG =========================================================================

with DAG(
    dag_id="inference_pipeline",
    default_args=default_args,
    description="Inferencia batch con el champion + posprocesamiento TLV; dispara el monitoreo",
    schedule=None,
    catchup=False,
    max_active_runs=1,
    params={"period": Param(_cfg["inference"]["default_period"], type="integer")},
    tags=["urp", "inferencia"],
) as dag:
    validar = PythonOperator(task_id="validar_entrada", python_callable=task_validar)
    champion = PythonOperator(task_id="cargar_champion", python_callable=task_champion)
    prepro = PythonOperator(task_id="preprocesar", python_callable=task_prepro)
    scoring = PythonOperator(task_id="scoring", python_callable=task_scoring)
    pospro = PythonOperator(task_id="posprocesar", python_callable=task_pospro)
    registro = PythonOperator(task_id="registrar_mlflow", python_callable=task_log)
    monitoreo = TriggerDagRunOperator(
        task_id="disparar_monitoreo",
        trigger_dag_id="monitoring_pipeline",
        conf={"period": "{{ params.period }}",
              "inference_run_id": "{{ ti.xcom_pull(task_ids='registrar_mlflow') }}"},
        wait_for_completion=False,  # la inferencia no espera al monitoreo
    )
    validar >> champion >> prepro >> scoring >> pospro >> registro >> monitoreo
