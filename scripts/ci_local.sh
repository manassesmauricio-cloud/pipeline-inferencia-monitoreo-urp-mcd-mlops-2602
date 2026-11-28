#!/usr/bin/env bash
# S14 - CI en el runner autohospedado: imagen, pruebas, DAG que cargan y datos reales descargados de Drive.
set -euo pipefail
cd "$(dirname "$0")/.."
IMG="urp-mcd-mlops-2602:latest"
corre() { MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W 2>/dev/null || pwd):/app" -w /app "$IMG" "$@"; }
echo "== 1/4 Construir la imagen del proyecto"
docker build -q -t "$IMG" . > /dev/null
echo "== 2/4 Pruebas (pytest)"
corre python -m pytest -q -p no:cacheprovider tests
echo "== 3/4 Los DAG cargan sin errores"
corre python tests/validar_dags.py
echo "== 4/4 Datos reales: descarga de Drive y validación"
corre python scripts/validar_datos_drive.py
echo "CI en verde"
