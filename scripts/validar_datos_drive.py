"""S14 (CI con datos reales): descarga de Drive los meses que usa el pipeline y valida reglas mínimas de calidad."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, raw_path  # noqa: E402
from src.drive import asegurar_mes  # noqa: E402

FECHA_P1 = 20220301


def fecha_esperada(p):
    anio, m = divmod(2 + (p - 1), 12)
    return (2022 + anio) * 10000 + (m + 1) * 100 + 1


cfg = load_config()
d = cfg["data"]
meses = sorted(set(d["training_periods"]) | {cfg["monitoring"]["reference_period"], cfg["inference"]["default_period"]})
errores = []
for p in meses:
    asegurar_mes(cfg, p)
    df = pd.read_csv(raw_path(cfg, p), usecols=["key_value", "p_fecinformacion", d["target_col"]], low_memory=False)
    if df["key_value"].duplicated().any():
        errores.append(f"p{p}: clientes duplicados")
    fechas = set(df["p_fecinformacion"].dropna().astype(int))
    if fechas != {fecha_esperada(p)}:
        errores.append(f"p{p}: p_fecinformacion {sorted(fechas)[:2]} distinto de {fecha_esperada(p)}")
    if not set(df[d["target_col"]].dropna().unique()) <= {0, 1}:
        errores.append(f"p{p}: target no binario")
    print(f"p{p}: {len(df):,} filas {'con errores' if errores else 'OK'}")
if errores:
    print("Validación de datos fallida:\n- " + "\n- ".join(errores))
    sys.exit(1)
print(f"Datos validados: {', '.join(f'p{p}' for p in meses)}")
