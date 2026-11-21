"""Pruebas del posprocesamiento TLV (src/posprocessing.py)."""
import numpy as np
import pandas as pd

from src.posprocessing import get_groups

rng = np.random.default_rng(7)


def _datos(n=2_000):
    return pd.DataFrame({
        "grp_campecs06m": rng.choice(["G1", "G2", "G3", "G4", "G5"], n),
        "prob_value_contact": rng.uniform(0, 1, n),
        "monto": rng.uniform(1_000, 50_000, n),
    })


def test_grupos_de_ejecucion_van_de_1_a_10():
    df = get_groups(rng.uniform(0, 1, 2_000), _datos())
    assert set(df["grupo_ejec_tlv"].astype(int)) == set(range(1, 11))


def test_grupo_1_tiene_la_mayor_puntuacion():
    df = get_groups(rng.uniform(0, 1, 2_000), _datos())
    media = df.groupby("grupo_ejec_tlv", observed=True)["puntuacion_tlv"].mean()
    assert media.loc[1] > media.loc[10]


def test_proporciones_fijas_por_grupo():
    df = get_groups(rng.uniform(0, 1, 10_000), _datos(10_000))
    share_g1 = (df["grupo_ejec_tlv"].astype(int) == 1).mean()
    assert abs(share_g1 - 0.05) < 0.01  # el grupo 1 es el 5 % superior (0.95 -> 1.0)
