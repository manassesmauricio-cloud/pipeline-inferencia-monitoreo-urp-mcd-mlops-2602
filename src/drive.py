"""S14: los datos del curso se descargan de la carpeta de Google Drive (compartida por enlace), nunca de una copia local.

asegurar_mes(cfg, p) deja data/raw/pN_extrac.csv listo: si ya está, no lo vuelve a bajar.
"""
import os

from src.config import raw_path

_INDICE = {}


def _indice(carpeta):
    """Nombre -> id de los archivos de la carpeta de Drive, sin descargarlos."""
    if carpeta not in _INDICE:
        import gdown
        archivos = gdown.download_folder(id=carpeta, skip_download=True, quiet=True, use_cookies=False)
        _INDICE[carpeta] = {os.path.basename(a.path): a.id for a in archivos}
    return _INDICE[carpeta]


def asegurar_mes(cfg, period):
    destino = raw_path(cfg, period)
    if os.path.exists(destino):
        return destino
    import gdown
    nombre = os.path.basename(destino)
    indice = _indice(cfg["data"]["drive_folder"])
    if nombre not in indice:
        raise FileNotFoundError(f"{nombre} no está en la carpeta de Drive del curso")
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    print(f"Descargando {nombre} desde Drive")
    gdown.download(id=indice[nombre], output=destino + ".parcial", quiet=True, use_cookies=False)
    os.replace(destino + ".parcial", destino)
    return destino
