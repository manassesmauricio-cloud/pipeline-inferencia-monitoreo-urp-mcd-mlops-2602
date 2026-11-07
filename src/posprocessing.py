"""Posprocesamiento TLV (misma lógica de scripts/posprocessing.py del curso)."""
import datetime

import numpy as np
import pandas as pd

# GENERAR GRUPOS DE EJECUCION =================================================

def get_groups(scores, df_post):
    dist_ge = [0, 0.035, 0.087, 0.237, 0.393, 0.529, 0.664, 0.787, 0.862, 0.95, 1.0]
    df_post['prob'] = scores
    df_post['prob_frescura'] = np.where(df_post['grp_campecs06m'] == 'G1', 0.066,
                               np.where(df_post['grp_campecs06m'] == 'G2', 0.028,
                               np.where(df_post['grp_campecs06m'] == 'G3', 0.022,
                               np.where(df_post['grp_campecs06m'] == 'G4', 0.008, 0.004))))

    df_post['prob_value_contact'] = df_post['prob_value_contact'].fillna(0.000001)
    df_post['puntuacion_tlv'] = df_post['prob'] * df_post['prob_value_contact'] * np.log(df_post['monto'] + 1) * df_post['prob_frescura']

    df_post['grupo_ejec_tlv'] = pd.qcut(df_post['puntuacion_tlv'], q=dist_ge,
                                        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    return df_post

# GUARDAR REPLICA =============================================================

def save_replica(df_post, model_name, period, dir_output):
    df_replica = pd.DataFrame()
    df_replica['codmes'] = df_post['partition']
    df_replica['tipdoc'] = '1'
    df_replica['coddoc'] = df_post['key_value']
    df_replica['puntuacion'] = df_post['puntuacion_tlv']
    df_replica['modelo'] = 'EC OMNICANAL'
    df_replica['fec_replica'] = datetime.date.today().strftime('%Y%m%d')
    df_replica['grupo_ejec'] = df_post['grupo_ejec_tlv'].astype(int)
    df_replica['score'] = df_post['prob']
    df_replica['orden'] = ''
    df_replica['variable1'] = df_post['codunicocli'].apply(lambda x: str(x).zfill(10))
    df_replica['variable2'] = df_post['monto']
    df_replica['variable3'] = ''

    # Eliminar duplicados
    df_replica = df_replica.sort_values('puntuacion', ascending=False)
    df_replica = df_replica.drop_duplicates('coddoc', keep='first')

    # Generar orden
    df_replica['orden'] = df_replica['puntuacion'].rank(method='first', ascending=False).astype(int)

    path = f'{dir_output}/scr_{model_name}_p{period}.txt'
    df_replica.to_csv(path, index=False, sep='|')
    return path, df_replica
