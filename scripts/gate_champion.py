"""S14 (gate del CD): el champion vigente debe superar el AUC mínimo de config.yaml para que el CD siga."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow.tracking import MlflowClient  # noqa: E402

from src.config import load_config, path, setup_mlflow  # noqa: E402

cfg = load_config()
setup_mlflow(cfg, cfg["mlflow"]["exp_training"])
cliente, M = MlflowClient(), cfg["mlflow"]
try:
    mv = cliente.get_model_version_by_alias(M["model_name"], M["champion_alias"])
except Exception:
    print("No hay champion registrado: corra el CT (entrenamiento) primero.")
    sys.exit(2 if "--existe" in sys.argv else 1)
if "--existe" in sys.argv:
    sys.exit(0)
run = cliente.get_run(mv.run_id)
auc, minimo = run.data.metrics.get("auc_test"), cfg["cd"]["auc_minimo"]
pasa = auc is not None and auc >= minimo
gate = {"version": int(mv.version), "modelo": run.data.tags.get("model", run.info.run_name), "auc_test": round(auc or 0, 4),
        "auc_minimo": minimo, "pasa": pasa}
os.makedirs(path("salidas"), exist_ok=True)
json.dump(gate, open(path("salidas/gate.json"), "w"), indent=2)
print(gate)
if not pasa:
    print(f"Gate del CD: AUC {auc} menor que {minimo}. El despliegue se detiene y la página no se actualiza.")
    sys.exit(1)
