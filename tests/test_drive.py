"""S14: si el mes ya está descargado, asegurar_mes no vuelve a ir a Drive."""
from src import drive


def _no_debia_ir_a_drive(carpeta):
    raise AssertionError("no debía ir a Drive")


def test_no_descarga_si_el_mes_ya_existe(tmp_path, monkeypatch):
    archivo = tmp_path / "p3_extrac.csv"
    archivo.write_text("x\n1\n")
    monkeypatch.setattr(drive, "raw_path", lambda cfg, p: str(archivo))
    monkeypatch.setattr(drive, "_indice", _no_debia_ir_a_drive)
    assert drive.asegurar_mes({"data": {"drive_folder": "x"}}, 3) == str(archivo)
