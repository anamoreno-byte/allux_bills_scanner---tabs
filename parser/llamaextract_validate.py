# parser/llamaextract_validate.py
from __future__ import annotations

from parser.parse_fields import normalize_tarifa


def clean_llamaextract_row(row: dict) -> dict:
    cleaned = dict(row)

    if cleaned.get("tarifa"):
        cleaned["tarifa"] = normalize_tarifa(cleaned["tarifa"])

    for key in [
        "no_servicio",
        "cuenta",
        "rmu",
        "rpu",
        "medidor",
    ]:
        if cleaned.get(key) is not None:
            cleaned[key] = str(cleaned[key]).strip()

    for key in [
        "carga_conectada_kw",
        "demanda_contratada_kw",
        "multiplicador",
        "lectura_actual_kwh",
        "lectura_anterior_kwh",
        "kwh_total",
        "importe_total",
        "cargo_fijo",
        "subtotal_energia",
        "subtotal",
        "iva",
        "fac_del_periodo",
        "total_linea",
    ]:
        cleaned[key] = safe_float(cleaned.get(key))

    hist = cleaned.get("historico_rows") or []
    cleaned["historico_count"] = len(hist)
    cleaned["tiene_consumo_historico"] = len(hist) > 0

    return cleaned


def safe_float(value):
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    s = str(value)
    s = s.replace("$", "").replace(",", "").replace(" ", "").strip()

    if not s:
        return None

    try:
        return float(s)
    except ValueError:
        return None