#!/usr/bin/env bash
# S14 - CD en el runner autohospedado:
#   1) despliega la imagen y los DAG nuevos en Airflow   2) CT: entrena si se pide o si no hay champion
#   3) gate de AUC mínimo   4) inferencia p6 y p7 (dispara el monitoreo)   5) reporte para GitHub Pages
# ENTRENAR = auto (solo si no hay champion) | si | no
set -euo pipefail
cd "$(dirname "$0")/.."
ENTRENAR="${ENTRENAR:-auto}"
RUN="${GITHUB_RUN_ID:-local}-$(date +%H%M%S)"
af() { docker compose exec -T airflow airflow "$@" 2>/dev/null | grep -v "^\["; }   # quita las líneas de log de Airflow
py() { docker compose exec -T -e GITHUB_SHA -e GITHUB_RUN_ID -e GITHUB_SERVER_URL -e GITHUB_REPOSITORY -e RUNNER_NAME airflow python "$@"; }
esperar() {   # dag run_id
  for i in $(seq 1 240); do
    st=$(af dags list-runs -d "$1" -o plain | awk -v r="$2" '$2==r {print $3; exit}')
    case "$st" in success) echo "   $1 $2: success"; return 0;; failed) echo "   $1 $2: failed"; return 1;; esac
    sleep 10
  done; echo "   $1 $2: timeout"; return 1
}
echo "== 1/5 Desplegar la imagen y los DAG nuevos"
docker compose up -d --build
for i in $(seq 1 60); do af dags list > /dev/null && break; sleep 5; done
echo "== 2/5 CT (entrenamiento continuo): ENTRENAR=$ENTRENAR"
if [ "$ENTRENAR" = "si" ] || { [ "$ENTRENAR" = "auto" ] && ! py scripts/gate_champion.py --existe > /dev/null; }; then
  af dags trigger training_pipeline -r "ct_$RUN" > /dev/null && esperar training_pipeline "ct_$RUN"
else
  echo "   se reutiliza el champion vigente"
fi
echo "== 3/5 Gate del despliegue"
py scripts/gate_champion.py
echo "== 4/5 Inferencia del mes de referencia y del mes nuevo (dispara el monitoreo)"
for p in 6 7; do
  af dags trigger inference_pipeline -r "inf_p${p}_$RUN" -c "{\"period\": $p}" > /dev/null && esperar inference_pipeline "inf_p${p}_$RUN"
done
MON=$(af dags list-runs -d monitoring_pipeline -o plain | awk '$1=="monitoring_pipeline" {print $2; exit}')
esperar monitoring_pipeline "$MON"
echo "== 5/5 Reporte para GitHub Pages"
py scripts/reporte_pages.py
echo "CD terminado. Airflow: http://localhost:8080 · MLflow: http://localhost:5000 · página: salidas/sitio/index.html"
