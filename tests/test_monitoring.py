"""Pruebas del cálculo de drift (src/monitoring_utils.py)."""
import numpy as np
import pandas as pd

from src.monitoring_utils import calculate_drift, compute_rolling_psi, estado_psi, psi_report

rng = np.random.default_rng(42)


def test_psi_misma_distribucion_es_casi_cero():
    ref = rng.normal(0, 1, 20_000)
    act = rng.normal(0, 1, 20_000)
    psi, *_ = calculate_drift("PSI", ref, act, 10)
    assert psi < 0.01


def test_psi_detecta_un_corrimiento_fuerte():
    ref = rng.normal(0, 1, 20_000)
    act = rng.normal(1.5, 1, 20_000)
    psi, *_ = calculate_drift("PSI", ref, act, 10)
    assert psi > 0.25


def test_psi_es_simetrico_y_no_negativo():
    ref = rng.normal(0, 1, 5_000)
    act = rng.normal(0.5, 1.2, 5_000)
    psi, valores, *_ = calculate_drift("PSI", ref, act, 10)
    assert psi >= 0
    assert np.all(valores >= 0)


def test_estado_psi_respeta_los_umbrales():
    assert estado_psi(0.05)[0] == "OK"
    assert estado_psi(0.10)[0] == "WARN"
    assert estado_psi(0.30)[0] == "ALARM"


def test_psi_report_solo_usa_columnas_numericas_comunes():
    ref = pd.DataFrame({"a": rng.normal(0, 1, 500), "b": rng.normal(0, 1, 500), "texto": ["x"] * 500})
    act = pd.DataFrame({"a": rng.normal(2, 1, 500), "b": rng.normal(0, 1, 500), "texto": ["y"] * 500})
    rep = psi_report(act, ref, 10)
    assert list(rep.index) == ["a", "b"]
    assert rep.loc["a", "psi"] > rep.loc["b", "psi"]


def test_rolling_psi_promedia_los_periodos_anteriores():
    hist = {7: 0.1, 8: 0.2, 9: 0.3}
    assert compute_rolling_psi(hist, 10, 2) == 0.25
    assert compute_rolling_psi(hist, 7, 3) == 0.0
