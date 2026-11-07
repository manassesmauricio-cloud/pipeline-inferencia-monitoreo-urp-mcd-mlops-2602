# Pipeline de inferencia y monitoreo - URP-MCD MLOps 2602

**MCD-402 · 2026-2 · Universidad Ricardo Palma**

Continuación del [pipeline de entrenamiento](https://github.com/manassesmauricio-cloud/pipeline-entrenamiento-urp-mcd-mlops-2602).
Aquí el modelo ya existe: se usa para **inferir** sobre meses nuevos y se **monitorea** el data drift.

```
inference_pipeline (núcleo 0)                          monitoring_pipeline (núcleos 1..4)
validar -> champion -> preprocesar -> scoring  ------>  preparar_referencia
        -> posprocesar (TLV) -> registrar -> disparar      -> drift_raw          \
                                   (no espera)             -> drift_preprocessed  |-> consolidar_alertas
                                                           -> drift_score         |   (semáforo + MLflow)
                                                           -> drift_grupo_ejec   /
```

- **Enfoque centralizado:** el monitoreo es un DAG aparte. La inferencia solo lo dispara (`wait_for_completion=False`) y termina sin esperarlo.
- **Paralelismo:** las 4 etapas del monitoreo corren a la vez (LocalExecutor), cada una fijada a un núcleo distinto del que usa la inferencia (`src/cores.py`).
- **Cálculo del drift:** `src/monitoring_utils.py` (función `monitor_stage`: PSI + métricas operativas).
- **Referencia (D10):** el mes `reference_period` (p6, OOT), procesado por el **mismo** pipeline de inferencia.

## Estructura

```
config/pipeline_config.yml   <- todos los parámetros (periodos, referencia, núcleos, MLflow)
dags/training_dag.py         <- pipeline 0 (heredado): genera y registra el champion
dags/inference_dag.py        <- pipeline 1: inferencia
dags/monitoring_dag.py       <- pipeline 2: monitoreo (centralizado)
src/preprocessing.py         <- prepare_features(): la misma función para entrenar e inferir
src/registry.py              <- register_champion() / load_champion() (Model Registry, alias @champion)
src/posprocessing.py         <- posprocesamiento TLV y archivo réplica
src/monitoring_utils.py      <- PSI, semáforo, PSI móvil y métricas operativas
src/cores.py                 <- afinidad de núcleos
docker-compose.yml           <- postgres + mlflow + airflow
```

## Requisitos

- Docker Desktop con al menos 4 núcleos y 8 GB de RAM asignados.
- Archivos `p1_extrac.csv` ... `p7_extrac.csv` del Drive del curso en `data/raw/`.

## Ejecución

```powershell
git clone https://github.com/manassesmauricio-cloud/pipeline-inferencia-monitoreo-urp-mcd-mlops-2602.git
cd pipeline-inferencia-monitoreo-urp-mcd-mlops-2602
# copiar los CSV en data/raw/
docker compose up -d --build
```

- Airflow: http://localhost:8080 (admin / admin)
- MLflow: http://localhost:5000

Orden (desde la UI de Airflow con "Trigger DAG w/ config", o por consola):

```powershell
# 1) entrenar y registrar el champion (~5 min)
docker compose exec airflow airflow dags trigger training_pipeline
# 2) inferir el mes de referencia: genera la línea base (el monitoreo se salta)
docker compose exec airflow airflow dags trigger inference_pipeline -c '{\"period\": 6}'
# 3) inferir un mes nuevo: al terminar dispara el monitoreo contra p6
docker compose exec airflow airflow dags trigger inference_pipeline -c '{\"period\": 7}'
```

## Qué revisar

| Dónde | Qué se ve |
|---|---|
| Airflow, vista Gantt de `monitoring_pipeline` | las 4 tareas `drift_*` se solapan en el tiempo |
| Logs de cada tarea | `núcleos [0]` en la inferencia; `núcleos [1]`, `[2]`, `[3]`, `[4]` en el monitoreo |
| MLflow | experimentos de entrenamiento, inferencia y monitoreo; el run de monitoreo trae el tag `inference_run_id` |
| `data/monitoring/p7/drift_report.json` | PSI por etapa, semáforo (OK < 0.10 <= WARN < 0.25 <= ALARM), top 5 variables y métricas operativas |

## Notas

- Con SQLite, Airflow solo usa SequentialExecutor (una tarea a la vez). Por eso el compose usa Postgres.
- El servidor de MLflow tiene su propia imagen (`Dockerfile.mlflow`): Airflow 2 fija SQLAlchemy 1.4 y el servidor de MLflow 3 necesita SQLAlchemy 2.
- `pandas<3`: en pandas 3 las columnas de texto ya no son `object` y el preprocesamiento heredado dejaría de crear dummies.
- Puerto ocupado: cambiar solo el lado izquierdo del mapeo en `docker-compose.yml` (por ejemplo `8081:8080`).
