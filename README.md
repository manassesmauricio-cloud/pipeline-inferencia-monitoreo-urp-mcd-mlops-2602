# Pipeline de inferencia y monitoreo - URP-MCD MLOps 2602

**MCD-402 · 2026-2 · Universidad Ricardo Palma**

Continuación del [pipeline de entrenamiento](https://github.com/manassesmauricio-cloud/pipeline-entrenamiento-urp-mcd-mlops-2602).

**Repositorio vivo del curso.** El proyecto empezó en el pipeline de entrenamiento y continúa aquí:

| Clase | Qué se agregó |
|---|---|
| S1 a S8 | Pipeline de entrenamiento (AutoML, Airflow, MLflow), heredado como `dags/training_dag.py` |
| S11 | Inferencia y monitoreo centralizado con drift en paralelo |
| S13 | CI en GitHub Actions: pruebas, imagen y DAG (`.github/workflows/ci.yml`) |
| S14 | Datos desde el Drive, CI/CD en un runner autohospedado y publicación en GitHub Pages (`.github/workflows/cd-autohospedado.yml`) |

Repos auxiliares: [taller-docker](https://github.com/manassesmauricio-cloud/taller-docker) (S8) y [demo-github-actions](https://github.com/manassesmauricio-cloud/demo-github-actions-urp-mcd-mlops-2602) (S13, cómo activar Actions y Pages).
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
- Conexión a internet: los meses `pN_extrac.csv` se descargan solos de la carpeta del curso en Drive (`data.drive_folder`) la primera vez que un DAG los necesita, y quedan en `data/raw/`.

## Ejecución

```powershell
git clone https://github.com/manassesmauricio-cloud/pipeline-inferencia-monitoreo-urp-mcd-mlops-2602.git
cd pipeline-inferencia-monitoreo-urp-mcd-mlops-2602
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

## Integración continua (GitHub Actions)

El workflow `.github/workflows/ci.yml` corre en cada push o pull request a `main`:

| Job | Qué hace | Tiempo aprox. |
|---|---|---|
| `pruebas` | `pytest` sobre el PSI, los estados del semáforo, el posprocesamiento TLV y la configuración (`tests/`), con dependencias livianas (`requirements-ci.txt`) | 1 min |
| `imagen` | `docker build` de la imagen de Airflow y `tests/validar_dags.py`, que falla si algún DAG no carga | 6-8 min |

Para correr lo mismo en local:

```bash
pip install -r requirements-ci.txt && pytest -q tests
docker build -t pipeline-inferencia:ci .
docker run --rm -v "$PWD":/app -w /app pipeline-inferencia:ci python tests/validar_dags.py
```

El CI no usa el dataset real ni entrena modelos: valida el código y la imagen. El entrenamiento, la inferencia y el monitoreo siguen corriendo en Airflow.

## S14: CI/CD autohospedado y GitHub Pages

El CI de la S13 corre en un **servidor de GitHub**: no tiene los datos ni el Airflow del proyecto. El workflow `.github/workflows/cd-autohospedado.yml` corre en **una máquina propia** (runner autohospedado):

```
push o ejecución manual
  ci (mi máquina):        imagen -> pytest -> DAG que cargan -> descarga y valida los meses reales de Drive
  cd (mi máquina):        despliega imagen y DAG en Airflow -> CT (si se pide) -> gate del champion
                          -> inferencia p6 y p7 -> monitoreo -> página de la corrida
  publicar (GitHub):      publica la página en GitHub Pages
```

| Pieza | Archivo |
|---|---|
| Descarga desde Drive | `src/drive.py` (`asegurar_mes`) |
| CI con datos reales | `scripts/ci_local.sh`, `scripts/validar_datos_drive.py` |
| CD, CT y gate | `scripts/cd_local.sh`, `scripts/gate_champion.py` (`cd.auc_minimo` en la configuración) |
| Página interactiva | `scripts/reporte_pages.py`, `scripts/plantilla_pages.html` |
| Runner | `runner/registrar_runner.ps1`, `runner/quitar_runner.ps1` |

### Ensayo local (sin GitHub)

```bash
bash scripts/correr_como_runner.sh
```

### Con el runner autohospedado

1. **Activar GitHub Pages** (una vez): Settings -> Pages -> Source: **GitHub Actions**, o `gh api -X POST repos/<usuario>/<repo>/pages -f build_type=workflow`.
2. **Proteger el repo**: Settings -> Actions -> General -> *Fork pull request workflows from outside collaborators* -> **Require approval for all outside collaborators**.
3. **Registrar el runner** (PowerShell): `cd runner` y `.\registrar_runner.ps1 -Repo <usuario>/<repo>`; luego `cd $env:USERPROFILE\actions-runner-s14` y `.\run.cmd`.
4. **Disparar**: `gh workflow run cd-autohospedado.yml -f entrenar=auto` y `gh run watch`.
5. **Romper el gate a propósito**: subir `cd.auc_minimo` a `0.99`. El CD se detiene en el gate y la página no se actualiza.
6. **Al terminar**: `Ctrl+C` en el runner y `.\quitar_runner.ps1 -Repo <usuario>/<repo>`.

**Seguridad:** un runner autohospedado en un repo público permite que un pull request desde un fork ejecute código en tu máquina. En sus proyectos se usa **solo con repos privados**. En esta demo el runner se enciende solo durante la clase, con la aprobación de externos activada, y se da de baja al final.

**CT:** por defecto (`entrenar=auto`) solo se entrena si no hay champion. El reentrenamiento no se dispara en cada push: es una decisión. En el Proyecto Final lo dispara la política de gatillo.
