import json
import os
import pickle
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import randint, uniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


def detect_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def build_models(use_gpu="auto"):
    gpu_available = detect_gpu() if use_gpu == "auto" else (use_gpu is True)
    task_type = "GPU" if gpu_available else "CPU"

    print(f"GPU disponible: {gpu_available} → CatBoost task_type={task_type}")

    import xgboost as xgb
    import lightgbm as lgb
    import catboost as catb

    tree_device = "cuda" if gpu_available else "cpu"

    models = {
        "xgb": {
            "model": xgb.XGBClassifier(
                eval_metric="logloss", random_state=42,
                device=tree_device, n_jobs=1
            ),
            "param_grid": {
                "n_estimators":    randint(100, 500),
                "learning_rate":   uniform(0.01, 0.19),
                "max_depth":       randint(3, 10),
                "subsample":       uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.7, 0.3),
            },
        },
        "lgbm": {
            "model": lgb.LGBMClassifier(
                random_state=42, verbose=-1,
                device=tree_device, n_jobs=1
            ),
            "param_grid": {
                "n_estimators":  randint(100, 500),
                "learning_rate": uniform(0.01, 0.19),
                "num_leaves":    randint(20, 50),
                "max_depth":     randint(3, 10),
                "subsample":     uniform(0.7, 0.3),
                "colsample_bytree": uniform(0.7, 0.3),
            },
        },
        "catb": {
            "model": catb.CatBoostClassifier(
                verbose=0, random_state=42,
                task_type=task_type,
                thread_count=-1
            ),
            "param_grid": {
                "iterations":    randint(100, 500),
                "learning_rate": uniform(0.01, 0.19),
                "depth":         randint(3, 10),
                "subsample":     uniform(0.7, 0.3),
                "l2_leaf_reg":   uniform(1, 10),
            },
        },
    }
    return models, gpu_available


def _hpo_one_model(name, model_info, X_train, y_train, X_test, y_test,
                   n_iter, cv, random_state):
    search = RandomizedSearchCV(
        estimator=model_info["model"],
        param_distributions=model_info["param_grid"],
        n_iter=n_iter, cv=cv, scoring="roc_auc",
        n_jobs=1, random_state=random_state, verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best = search.best_estimator_
    auc_train = roc_auc_score(y_train, best.predict_proba(X_train)[:, 1])
    auc_test  = roc_auc_score(y_test,  best.predict_proba(X_test)[:, 1])
    decay = ((auc_train - auc_test) / auc_train * 100) if auc_train > 0 else float("inf")

    return {
        "name":       name,
        "model":      best,
        "params":     search.best_params_,
        "auc_cv":     round(search.best_score_, 4),
        "auc_train":  round(auc_train, 4),
        "auc_test":   round(auc_test,  4),
        "decay_pct":  round(decay, 2),
        "elapsed_s":  round(elapsed, 2),
    }


def run_parallel_hpo(X_train, y_train, X_test, y_test,
                     n_iter=15, cv=3, n_jobs=-1,
                     random_state=42, decay_threshold=10.0,
                     use_gpu="auto"):
    models, gpu_available = build_models(use_gpu)

    t_seq = time.time()
    results_seq = [
        _hpo_one_model(n, m, X_train, y_train, X_test, y_test,
                       n_iter, cv, random_state)
        for n, m in models.items()
    ]
    t_seq_total = time.time() - t_seq

    models_fresh, _ = build_models(use_gpu)

    t_par = time.time()
    results_par = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_hpo_one_model)(
            n, m, X_train, y_train, X_test, y_test, n_iter, cv, random_state
        )
        for n, m in models_fresh.items()
    )
    t_par_total = time.time() - t_par

    speedup = round(t_seq_total / t_par_total, 2) if t_par_total > 0 else 0

    parallelism_meta = {
        "backend":       "loky",
        "n_jobs":        n_jobs,
        "t_sequential_s": round(t_seq_total, 2),
        "t_parallel_s":   round(t_par_total, 2),
        "speedup":        speedup,
        "gpu_used":       gpu_available,
    }

    champion = None
    for r in results_par:
        print(f"  {r['name']:<8} auc_test={r['auc_test']}  decay={r['decay_pct']}%  "
              f"t={r['elapsed_s']}s")
        if r["decay_pct"] < decay_threshold:
            if champion is None or r["auc_test"] > champion["auc_test"]:
                champion = r

    print(f"\nSpeedup CPU paralelo: {speedup}×  |  GPU: {gpu_available}")

    return results_par, champion, parallelism_meta


def save_model(result, parallelism_meta, models_dir, run_timestamp=None):
    ts = run_timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(models_dir, ts)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, f"{result['name']}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(result["model"], f)

    def _clean(v):
        try:
            json.dumps(v)
            return v
        except (TypeError, OverflowError):
            return str(v)

    metadata = {
        "name":         result["name"],
        "performance": {
            "auc_cv":    result["auc_cv"],
            "auc_train": result["auc_train"],
            "auc_test":  result["auc_test"],
            "decay_pct": result["decay_pct"],
            "elapsed_s": result["elapsed_s"],
        },
        "hyperparameters": {k: _clean(v) for k, v in result["params"].items()},
        "parallelism":     parallelism_meta,
        "python_version":  platform.python_version(),
        "timestamp":       ts,
    }

    meta_path = os.path.join(out_dir, f"{result['name']}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Modelo → {model_path}")
    print(f"Metadata → {meta_path}")
    return model_path, meta_path


def main(train_path, test_path, models_dir, n_iter=15, cv=3,
         n_jobs=-1, random_state=42, decay_threshold=10.0, use_gpu="auto"):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    target = train_df.columns[0]
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_test,  y_test  = test_df.drop(columns=[target]),  test_df[target]

    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    results, champion, par_meta = run_parallel_hpo(
        X_train, y_train, X_test, y_test,
        n_iter=n_iter, cv=cv, n_jobs=n_jobs,
        random_state=random_state, decay_threshold=decay_threshold,
        use_gpu=use_gpu,
    )

    if champion:
        model_path, meta_path = save_model(champion, par_meta, models_dir)
        return champion, par_meta, model_path
    else:
        print("No se encontró campeón con decay < threshold.")
        return None, par_meta, None
