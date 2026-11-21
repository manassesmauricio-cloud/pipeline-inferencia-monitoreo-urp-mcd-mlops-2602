"""Valida que todos los DAG cargan sin errores de importación (se corre dentro de la imagen)."""
import sys

from airflow.models import DagBag

bag = DagBag(dag_folder="dags", include_examples=False)
for archivo, error in bag.import_errors.items():
    print(f"ERROR en {archivo}:\n{error}")
print(f"DAG cargados: {sorted(bag.dag_ids)}")
if bag.import_errors or not bag.dag_ids:
    sys.exit(1)
