import re
from typing import Iterable, List

# Coincide con cualquier carácter de control ASCII 1..31 (excluyendo 0 para evitar problemas con terminadores) 
CONTROL_CHARS_PATTERN = re.compile(r"[\x01-\x1F]")

def _replace_control(m: re.Match) -> str:
    """Convierte un carácter de control en una carpeta con dos dígitos.

    Ejemplos de mapeo:
      \x01 -> "\\01"
      \x02 -> "\\02"
      ...
      \x1F -> "\\31" (31 decimal)
    """
    code = ord(m.group(0))  # 1..31
    return f"\\{code:02d}"  # Barra + dos dígitos cero–rellenos

def normalize_control_path(text: str) -> str:
    """Normaliza un path reemplazando caracteres de control (\x01-\x1F) por subcarpetas numéricas de dos dígitos.

    Reglas:
      - Cada byte de control en el rango 1..31 se transforma en: '\\NN' (NN = código decimal con dos dígitos).
      - No modifica secuencias ya expandidas (p.ej. '\\01') porque no contiene el byte de control original.
      - Mantiene el resto del path intacto.

    Ejemplos:
        In : C:\\RUTA\\VIEW SCREENSHOTS\x01\x01_10-01.png
        Out: C:\\RUTA\\VIEW SCREENSHOTS\\01\\01_10-01.png

        In : C:\\RUTA\\VIEW SCREENSHOTS\x02\x02_10-01.png
        Out: C:\\RUTA\\VIEW SCREENSHOTS\\02\\02_10-01.png

    Args:
        text: Cadena original que puede contener caracteres de control crudos.

    Returns:
        str: Cadena con los caracteres de control reemplazados por carpetas numéricas.
    """
    if not text:
        return text
    return CONTROL_CHARS_PATTERN.sub(_replace_control, text)


def normalize_many(paths: Iterable[str]) -> List[str]:
    """Aplica normalize_control_path sobre múltiples rutas."""
    return [normalize_control_path(p) for p in paths]


__all__ = ["normalize_control_path", "normalize_many"]
