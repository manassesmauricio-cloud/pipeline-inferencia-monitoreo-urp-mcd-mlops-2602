"""PIPELINE 2: monitoreo de data drift (enfoque centralizado).

Lo dispara inference_pipeline con conf = {"period": N, "inference_run_id": ...}.

preparar_referencia -> [drift_raw, drift_preprocessed, drift_score, drift_grupo_ejec]
-> consolidar_alertas

Las 4 etapas corren en paralelo (LocalExecutor), cada una fijada a un núcleo distinto
del que usa la inferencia. El cálculo de PSI y las métricas operativas vienen de
src/monitoring_utils.py (monitor_stage), sin modificaciones.
"""
import glob
import json
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import load_config, raw_path, inference_dir, monitoring_dir  # noqa: E402
from src.cores import pin, monitor_core  # noqa: E402

_cfg = load_config()
MON, D = _cfg["monitoring"], _cfg["data"]
REF = MON["reference_period"]
STAGES = MON["stages"]
PAYLOAD = {"params": {k: MON[k] for k in ["psi_quantils", "score_col", "grupo_col", "puntuacion_col"]}}

default_args = {
    "owner": _cfg["airflow"]["dag_owner"],
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "retries": 0,
    "retry_delay": timedelta(minutes=1),
}


def _period(ctx):
    return int(ctx["dag_run"].conf["period"])


def _replica(period):
    found = glob.glob(f"{inference_dir(_cfg, period)}/scr_*_p{period}.txt")
    return found[0] if found else None


def _actual_path(stage, period):
    d = inference_dir(_cfg, period)
    return {"raw": raw_path(_cfg, period),
            "preprocessed": f"{d}/preprocessed.csv",
            "score": f"{d}/score.csv",
            "grupo_ejec": _replica(period)}[stage]


def _load_reference(stage):
    """Referencia (D10): las salidas del mismo pipeline de inferencia sobre el periodo REF."""
    import pandas as pd
    path = _actual_path(stage, REF)
    if stage == "raw":  # sin targets ni identificadores: solo se comparan variables de entrada
        df = pd.read_csv(path, low_memory=False)
        return df.drop(columns=[c for c in D["drop_cols"] if c in df.columns])
    if stage == "grupo_ejec":
        return pd.read_csv(path, sep="|")
    return pd.read_csv(path)


# TAREAS ======================================================================

def task_preparar(**ctx):
    p = _period(ctx)
    if p == REF:
        print(f"p{p} es el periodo de referencia: no se monitorea contra sí mismo")
        return False
    faltan = [s for s in STAGES if not _actual_path(s, REF) or not os.path.exists(_actual_path(s, REF))]
    if faltan:
        raise FileNotFoundError(
            f"Falta la referencia p{REF} ({faltan}). Ejecute inference_pipeline con period={REF}")
    print(f"Monitoreo p{p} contra referencia p{REF}")
    return True


def task_drift(stage, idx, **ctx):
    from src.monitoring_utils import monitor_stage
    p = _period(ctx)
    cores = pin(monitor_core(_cfg, idx))
    print(f"[drift_{stage}] p{p} vs p{REF} | núcleos {cores} | pid {os.getpid()}")
    df_ref = _load_reference(stage)
    res = monitor_stage(stage, PAYLOAD, p, _actual_path(stage, p), df_ref)

    out = monitoring_dir(_cfg, p)
    rep = res["report"]
    rep.to_csv(f"{out}/{stage}_report.csv", index=(stage != "grupo_ejec"))
    top = []
    if stage in ("raw", "preprocessed") and len(rep):
        top = [{"feature": f, "psi": float(v)} for f, v in rep["psi"].head(5).items()]
    summary = {"stage": stage, "period": p, "reference": REF, "psi": res["psi_medio"],
               "cores": cores, "ops": res["ops"], "top5": top}
    with open(f"{out}/{stage}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary["psi"]


def task_consolidar(**ctx):
    import mlflow
    from src.config import setup_mlflow
    from src.monitoring_utils import estado_psi, compute_rolling_psi

    p = _period(ctx)
    out = monitoring_dir(_cfg, p)
    hist_path = f"{monitoring_dir(_cfg)}/history.json"
    history = json.load(open(hist_path)) if os.path.exists(hist_path) else {}

    etapas, orden = {}, {"OK": 0, "WARN": 1, "ALARM": 2}
    for s in STAGES:
        fp = f"{out}/{s}.json"
        if not os.path.exists(fp):
            etapas[s] = {"estado": "SIN_DATO"}
            continue
        r = json.load(open(fp))
        if r["psi"] is None:
            etapas[s] = {"estado": "SIN_DATO", "ops": r["ops"]}
            continue
        hs = {int(k): v for k, v in history.get(s, {}).items()}
        rolling = compute_rolling_psi(hs, p, MON["rolling_window"])
        estado, _ = estado_psi(r["psi"])
        etapas[s] = {"psi": round(r["psi"], 5), "psi_movil_previo": round(rolling, 5),
                     "estado": estado, "cores": r["cores"], "ops": r["ops"], "top5": r["top5"]}
        history.setdefault(s, {})[str(p)] = r["psi"]

    estados = [e["estado"] for e in etapas.values() if e["estado"] in orden]
    alerta = max(estados, key=orden.get) if estados else "SIN_DATO"
    reporte = {"period": p, "reference": REF, "alerta": alerta,
               "inference_run_id": ctx["dag_run"].conf.get("inference_run_id"),
               "generado": datetime.now().isoformat(timespec="seconds"), "etapas": etapas}
    with open(f"{out}/drift_report.json", "w") as f:
        json.dump(reporte, f, indent=2, default=float)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    setup_mlflow(_cfg, _cfg["mlflow"]["exp_monitoring"])
    with mlflow.start_run(run_name=f"monitoreo_p{p}"):
        mlflow.set_tags({"periodo": f"p{p}", "referencia": f"p{REF}", "alerta": alerta,
                         "inference_run_id": str(reporte["inference_run_id"])})
        for s, e in etapas.items():
            if "psi" in e:
                mlflow.log_metrics({f"psi_{s}": e["psi"], f"psi_movil_{s}": e["psi_movil_previo"]})
            if "ops" in e:
                mlflow.log_metrics({f"t_{s}_s": e["ops"]["duration_s"],
                                    f"ram_{s}_mb": e["ops"]["ram_proc_mb"]})
            if e.get("cores"):
                mlflow.log_metric(f"core_{s}", e["cores"][0])
        mlflow.log_artifacts(out, artifact_path="reportes")

    print(f"\nMONITOREO p{p} vs p{REF} -> ALERTA GLOBAL: {alerta}")
    for s, e in etapas.items():
        print(f"  {s:13s} PSI={e.get('psi', '-')!s:9s} {e['estado']:8s} núcleos={e.get('cores', '-')}")


# DAG =========================================================================

with DAG(
    dag_id="monitoring_pipeline",
    default_args=default_args,
    description="Data drift (PSI) en 4 etapas en paralelo; lo dispara la inferencia",
    schedule=None,
    catchup=False,
    max_active_runs=2,
    tags=["urp", "monitoreo"],
) as dag:
    prep = ShortCircuitOperator(task_id="preparar_referencia", python_callable=task_preparar)
    drifts = [PythonOperator(task_id=f"drift_{s}", python_callable=task_drift,
                             op_kwargs={"stage": s, "idx": i})
              for i, s in enumerate(STAGES)]
    consol = PythonOperator(task_id="consolidar_alertas", python_callable=task_consolidar,
                            trigger_rule="all_done")
    prep >> drifts >> consol
