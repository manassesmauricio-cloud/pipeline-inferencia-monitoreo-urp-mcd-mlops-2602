"""Afinidad de núcleos: la inferencia y cada etapa del monitoreo corren en núcleos distintos.

os.sched_setaffinity solo existe en Linux (Docker); en otros sistemas no hace nada.
"""
import os


def pin(cores):
    if cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cores))
    if hasattr(os, "sched_getaffinity"):
        return sorted(os.sched_getaffinity(0))
    return []


def inference_cores(cfg):
    n = os.cpu_count() or 1
    r = min(cfg["monitoring"]["inference_cores"], n)
    return list(range(r))


def monitor_core(cfg, idx):
    if not cfg["monitoring"].get("pin_cores", True):
        return []
    n = os.cpu_count() or 1
    r = cfg["monitoring"]["inference_cores"]
    libres = list(range(r, n))
    return [libres[idx % len(libres)]] if libres else []
