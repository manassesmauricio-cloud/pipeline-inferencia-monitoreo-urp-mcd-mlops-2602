"""S14: arma la página interactiva que el workflow publica en GitHub Pages (historial de corridas del CD).

Lee el gate (salidas/gate.json) y el reporte de drift del mes nuevo (data/monitoring/pN/drift_report.json),
agrega la corrida al historial (data/monitoring/historial_cd.json) y escribe salidas/sitio/index.html.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import config_hash_texto, load_config, monitoring_dir, path  # noqa: E402

cfg = load_config()
p = cfg["inference"]["default_period"]
gate = json.load(open(path("salidas/gate.json")))
drift = json.load(open(f"{monitoring_dir(cfg, p)}/drift_report.json"))
repo = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ.get('GITHUB_REPOSITORY', '')}".rstrip("/")
entrada = {
    "fecha": time.strftime("%d/%m/%Y %H:%M"), "commit": (os.environ.get("GITHUB_SHA") or "local")[:7],
    "run_id": os.environ.get("GITHUB_RUN_ID", "local"), "runner": os.environ.get("RUNNER_NAME", "local"), "repo": repo,
    "config_hash": config_hash_texto(), "gate": gate, "periodo": f"p{p}", "referencia": f"p{drift['reference']}",
    "alerta": drift["alerta"],
    "etapas": [{"etapa": k, "estado": v.get("estado"), "psi": v.get("psi"), "top5": v.get("top5", [])} for k, v in drift["etapas"].items()],
}
ruta_hist = f"{monitoring_dir(cfg)}/historial_cd.json"
hist = json.load(open(ruta_hist)) if os.path.exists(ruta_hist) else []
hist = [h for h in hist if (h["commit"], h["run_id"]) != (entrada["commit"], entrada["run_id"])] + [entrada]
json.dump(hist[-20:], open(ruta_hist, "w"), indent=2, default=str)
html = open(path("scripts/plantilla_pages.html"), encoding="utf-8").read().replace("__HISTORIAL__", json.dumps(hist[-20:], default=str, ensure_ascii=False))
os.makedirs(path("salidas/sitio"), exist_ok=True)
open(path("salidas/sitio/index.html"), "w", encoding="utf-8").write(html)
print(f"Página con {len(hist[-20:])} corridas en salidas/sitio/index.html (alerta del mes: {drift['alerta']})")
