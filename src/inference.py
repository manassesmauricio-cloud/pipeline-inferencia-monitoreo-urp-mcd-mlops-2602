"""Scoring con el modelo champion."""
import numpy as np


def score(model, X):
    return np.asarray(model.predict_proba(X)[:, 1])
