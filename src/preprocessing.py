import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_rawdata(dir_rawdata, period=None, type_work="training"):
    if type_work == "training":
        csv_files = glob.glob(f"{dir_rawdata}/*.csv")
        df_list = []
        for path in csv_files:
            try:
                df_list.append(pd.read_csv(path))
            except Exception as e:
                print(f"Error leyendo {path}: {e}")
        if not df_list:
            raise FileNotFoundError(f"No se encontraron CSV en {dir_rawdata}")
        return pd.concat(df_list, ignore_index=True)

    path_pattern = glob.glob(f"{dir_rawdata}/p{period}_extrac.csv")
    if not path_pattern:
        raise FileNotFoundError(f"No se encontró p{period}_extrac.csv en {dir_rawdata}")
    return pd.read_csv(path_pattern[0])


def process_vars(df):
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    float_cols = df.select_dtypes(include=["float16", "float32", "float64"]).columns
    df[float_cols] = df[float_cols].astype(float).round(4)

    df.replace(["", "null", "None"], np.nan, inplace=True)
    df.fillna(-9999999, inplace=True)

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df


def split_and_save(df, target_col, dir_processed, test_size=0.2, random_state=42):
    if target_col not in df.columns:
        raise ValueError(f"Columna target '{target_col}' no encontrada en el DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_df = pd.concat([y_train, X_train], axis=1)
    test_df  = pd.concat([y_test,  X_test],  axis=1)

    train_path = f"{dir_processed}/train.csv"
    test_path  = f"{dir_processed}/test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)

    print(f"Train: {train_df.shape}  →  {train_path}")
    print(f"Test:  {test_df.shape}   →  {test_path}")
    return train_path, test_path


def main(dir_rawdata, dir_processed, target_col="target",
         test_size=0.2, random_state=42, type_work="training", period=None):
    df = read_rawdata(dir_rawdata, period=period, type_work=type_work)
    df = process_vars(df)
    return split_and_save(df, target_col, dir_processed, test_size, random_state)


# FUNCIÓN COMPARTIDA ENTRENAMIENTO / INFERENCIA ===============================

def _clean_name(c):
    import re
    return re.sub(r"[^0-9a-zA-Z_]", "_", str(c))


def prepare_features(df, drop_cols, feature_columns=None):
    """Misma transformación para entrenar y para inferir.

    - quita identificadores, targets y variables con leakage
    - process_vars: tipos, nulos y dummies
    - limpia nombres de columnas (LightGBM y XGBoost rechazan caracteres especiales)
    - en inferencia, reindexa a las columnas del entrenamiento: una categoría nueva
      se descarta y una que no llegó se rellena con 0
    """
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    X = process_vars(X)
    X.columns = [_clean_name(c) for c in X.columns]
    X = X.loc[:, ~X.columns.duplicated()]
    if feature_columns is not None:
        X = X.reindex(columns=feature_columns, fill_value=0)
    return X
