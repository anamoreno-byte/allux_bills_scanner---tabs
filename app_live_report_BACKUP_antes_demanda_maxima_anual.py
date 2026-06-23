from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re
from io import BytesIO

# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"

DEFAULT_PARSED_CSV = OUTPUT_DIR / "bills_parsed_v5_validated_by_tariff.csv"
DEFAULT_HISTORICO_CSV = OUTPUT_DIR / "bills_historico_v2_RUN_9933.csv"
DEFAULT_GENERAL_DATA = DATA_DIR / "260414_Datos Generales_19 CC.xlsx"

PDBT_KWH_RESCUE_CSV = OUTPUT_DIR / "results_PDBT_full.csv"

GDM_RESCUE_CSV = OUTPUT_DIR / "results_AnaTe18Jun_full_schema_v3.csv"

NEW_PARSER_ROWS_CSV = OUTPUT_DIR / "results_file_path_table_1_FULL_v1.csv"


st.set_page_config(
    page_title="Análisis del desempeño eléctrico",
    layout="wide"
)


# ============================================================
# Styling
# ============================================================
st.markdown(
    """
    <style>
    /* Quitar espacio superior default de Streamlit */
    .block-container {
        padding-top: 0.5rem !important;
    }

    /* NIVEL 1: Título principal de la app */
    .main-title {
        font-size: 3.0rem;
        font-weight: 800;
        color: #8A3F2A;
        margin-top: 0rem !important;
        padding-top: 2.2rem !important;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }

    /* NIVEL 2: Menú de tabs */
    .stTabs [data-baseweb="tab"] {
        color: #C96A1B !important;
        font-size: 3rem !important;
        font-weight: 800 !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #C96A1B !important;
        font-size: 1.35rem !important;
        font-weight: 800 !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
    }

    .stTabs [role="tab"] {
        margin-right: 0.75rem !important;
    }

    /* Tab activo */
    .stTabs [aria-selected="true"] p {
        color: #D97706 !important;
        font-size: 1.35rem !important;
        font-weight: 900 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #D97706 !important;
        font-weight: 900 !important;
    }

    /* NIVEL 3: Títulos principales dentro de cada tab */
    .section-title {
        color: #2E7D32;
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 1.8rem;
        margin-bottom: 1.2rem;
    }

    /* NIVEL 4: Subtítulos dentro de cada sección */
    .subsection-title {
        color: #2F6FB2;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 1.4rem;
        margin-bottom: 0.9rem;
    }

    .note-box {
        padding: 0.9rem 1rem;
        border-left: 5px solid #2E7D32;
        background-color: #F1F8E9;
        border-radius: 0.4rem;
        margin-bottom: 1rem;
        color: #333;
    }

    .warning-box {
        padding: 0.9rem 1rem;
        border-left: 5px solid #F57C00;
        background-color: #FFF8E1;
        border-radius: 0.4rem;
        margin-bottom: 1rem;
        color: #333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Utility functions
# ============================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    """
    Reads a CSV robustly.
    """

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    attempts = [
        {"encoding": "utf-8", "sep": ","},
        {"encoding": "utf-8-sig", "sep": ","},
        {"encoding": "latin1", "sep": ","},
        {"encoding": "utf-8", "sep": ";"},
        {"encoding": "latin1", "sep": ";"},
    ]

    last_error = None

    for kwargs in attempts:
        try:
            return pd.read_csv(
                path,
                dtype=str,
                engine="python",
                on_bad_lines="skip",
                **kwargs
            )
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not read CSV: {path}\nLast error: {last_error}")


def read_general_excel_all_sheets(path: Path) -> pd.DataFrame:
    """
    Reads all sheets from the general-data Excel file and stacks them into one table.
    Adds a source_sheet column to preserve mall/sheet origin.
    """
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    sheets = pd.read_excel(path, sheet_name=None, dtype=str)

    frames = []

    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Drop fully empty rows and columns.
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")

        if df.empty:
            continue

        df["source_sheet"] = sheet_name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def file_signature(path: Path):
    """
    Firma del archivo para que Streamlit sepa cuándo refrescar la caché.

    Si el archivo cambia de tamaño o fecha de modificación,
    Streamlit vuelve a leerlo.
    """
    path = Path(path)

    if not path.exists():
        return str(path), None, None

    stat = path.stat()

    return str(path), stat.st_mtime, stat.st_size


@st.cache_data(show_spinner=False)
def read_csv_cached(path_text, mtime, size):
    return read_csv_safe(Path(path_text))


@st.cache_data(show_spinner=False)
def read_excel_all_sheets_cached(path_text, mtime, size):
    return read_general_excel_all_sheets(Path(path_text))

def clean_number_series(s: pd.Series) -> pd.Series:
    """
    Converts messy numeric strings into numeric values.

    Handles:
    - commas as thousands separators
    - dollar signs
    - spaces
    - non-breaking spaces
    """
    if s is None:
        return pd.Series(dtype=float)

    return (
        s.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u00a0", "", regex=False)
        .replace({"": None, "nan": None, "None": None})
        .pipe(pd.to_numeric, errors="coerce")
    )


def clean_date_series(s: pd.Series):
    """
    Converts a date-like series into datetime.

    IMPORTANTE:
    Los recibos CFE vienen normalmente como DD/MM/AAAA,
    por eso usamos dayfirst=True.
    """
    if s is None:
        return pd.Series(dtype="datetime64[ns]")

    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def normalize_text_key(s: pd.Series) -> pd.Series:
    """
    Normalizes text keys for flexible matching.
    """
    if s is None:
        return pd.Series(dtype=str)

    return (
        s.astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("\u00a0", " ", regex=False)
        .replace({"NAN": None, "NONE": None, "": None})
    )

def normalize_cc_key(s):
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace("&", "Y", regex=False)
        .str.replace("Á", "A", regex=False)
        .str.replace("É", "E", regex=False)
        .str.replace("Í", "I", regex=False)
        .str.replace("Ó", "O", regex=False)
        .str.replace("Ú", "U", regex=False)
        .str.replace("Ñ", "N", regex=False)
        .str.replace("CITIBANAMEX", "BANAMEX", regex=False)
        .str.replace("CAFETERADE", "CAFETERA DE", regex=False)
        .str.replace(" SA DE CV", "", regex=False)
        .str.replace(" S DE RL DE CV", "", regex=False)
        .str.replace(" SAPI DE CV", "", regex=False)
        .str.replace(" INST DE BANCA MULT", "", regex=False)
        .str.replace(" INSTITUCION DE BANCA MULTIPLE", "", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

def normalize_meter_cc(s):
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.replace(".", "", regex=False)
    )

def normalize_service_cc(s):
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
        .replace({"NAN": "", "NONE": "", "<NA>": ""})
    )

def normalize_tarifa_value(value):
    """
    Normaliza tarifas leídas del parser o de DG.

    Corrige:
    - espacios antes/después
    - errores OCR comunes
    - variantes con NO pegado al final
    - tarifa doméstica 1C tratada como PDBT para este análisis
    """

    if pd.isna(value):
        return pd.NA

    t = str(value).upper().strip()

    t = (
        t.replace("\u00a0", "")
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
        .replace(",", "")
    )

    if t in ["", "NAN", "NONE", "<NA>", "SINTARIFA", "BLANKS", "(BLANKS)"]:
        return pd.NA

    # Tarifas domésticas o mal leídas que para este análisis se tratan como PDBT
    if t in ["1C", "01C", "IC"]:
        return "PDBT"

    # Correcciones OCR / variantes vistas en parser
    tarifa_alias = {
        "GDMIONO": "GDMTO",
        "GDM1ONO": "GDMTO",
        "GDMTO": "GDMTO",
        "GOMTO": "GDMTO",
        "GDMTH": "GDMTH",
        "GDBT": "GDBT",
        "PDBT": "PDBT",
        "PDSTNO": "PDBT",
        "POBTNO": "PDBT",
        "PDBTNO": "PDBT",
        "PPBTNO": "PDBT",
        "PDBTNO": "PDBT",
    }

    if t in tarifa_alias:
        return tarifa_alias[t]

    # Reglas flexibles por contenido
    if "GDMTH" in t:
        return "GDMTH"

    if "GDBT" in t:
        return "GDBT"

    if "GDMTO" in t or "GOMTO" in t or "GDMIO" in t:
        return "GDMTO"

    if "PDBT" in t or "PDST" in t or "POBT" in t or "PPBT" in t:
        return "PDBT"

    return t

def _first_existing_col(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalizar_file_path_enriq(s):
    ruta = (
        s.fillna("")
        .astype(str)
        .str.strip()
        .str.replace("\\", "/", regex=False)
        .str.replace(r"/+", "/", regex=True)
        .str.lower()
    )

    # En algunos archivos el parser trae:
    # .../7. Recibos CFE/RECIBOS CFE/GDMTH/...
    #
    # y los archivos de rescate traen:
    # .../7. Recibos CFE/GDMTH/...
    #
    # Normalizamos ese duplicado para que el match por file_path sí encuentre
    # el mismo recibo.
    ruta = ruta.str.replace(
        r"/(\d+\.\s*)?recibos cfe/recibos cfe/",
        r"/\1recibos cfe/",
        regex=True
    )

    return ruta


def _normalizar_tarifa_enriq(s):
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

def _normalizar_periodo_enriq(s):
    return (
        s.fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace("\u00a0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace("Á", "A", regex=False)
        .str.replace("É", "E", regex=False)
        .str.replace("Í", "I", regex=False)
        .str.replace("Ó", "O", regex=False)
        .str.replace("Ú", "U", regex=False)
        .str.replace(".", "", regex=False)
        .str.strip()
    )

def _to_num_enriq(s):
    return pd.to_numeric(
        s,
        errors="coerce"
    )


def _append_audit_status(df, mask, status_text):
    if "parser_enriquecido_status" not in df.columns:
        df["parser_enriquecido_status"] = ""

    mask = mask.fillna(False)

    current = (
        df.loc[mask, "parser_enriquecido_status"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df.loc[mask, "parser_enriquecido_status"] = np.where(
        current.eq(""),
        status_text,
        current + " | " + status_text
    )

    return df

def _preparar_rescate_csv(path):
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(
        path,
        low_memory=False
    )

    if df.empty:
        return df

    if "_status" in df.columns:
        df = df[
            df["_status"]
            .fillna("")
            .astype(str)
            .str.upper()
            .eq("COMPLETED")
        ].copy()

    file_path_col = _first_existing_col(
        df,
        [
            "file_path",
            "source_file_path"
        ]
    )

    if file_path_col is not None:
        df["_file_path_key_enriq"] = _normalizar_file_path_enriq(
            df[file_path_col]
        )
    else:
        df["_file_path_key_enriq"] = ""

    no_servicio_col = _first_existing_col(
        df,
        [
            "no_servicio",
            "No. servicio",
            "servicio"
        ]
    )

    medidor_col = _first_existing_col(
        df,
        [
            "medidor",
            "MEDIDOR",
            "No. De medidor",
            "No. de medidor"
        ]
    )

    periodo_col = _first_existing_col(
        df,
        [
            "periodo_fact",
            "periodo",
            "periodo_facturacion"
        ]
    )

    if no_servicio_col is not None:
        df["_key_no_servicio_enriq"] = normalize_service_cc(
            df[no_servicio_col]
        )
    else:
        df["_key_no_servicio_enriq"] = ""

    if medidor_col is not None:
        df["_key_medidor_enriq"] = normalize_meter_cc(
            df[medidor_col]
        )
    else:
        df["_key_medidor_enriq"] = ""

    if periodo_col is not None:
        df["_key_periodo_enriq"] = _normalizar_periodo_enriq(
            df[periodo_col]
        )
    else:
        df["_key_periodo_enriq"] = ""

    # Conservamos filas con al menos alguna llave útil.
    df = df[
        df["_file_path_key_enriq"].ne("")
        | (
            df["_key_no_servicio_enriq"].ne("")
            & df["_key_medidor_enriq"].ne("")
        )
    ].copy()

    # Solo deduplicamos por file_path cuando sí existe file_path.
    # No deduplicamos aquí por servicio/medidor porque puede haber varios periodos.
    if df["_file_path_key_enriq"].ne("").any():
        df = df.sort_values(
            by=[
                "_file_path_key_enriq",
                "_key_periodo_enriq"
            ]
        )

    return df

def _actualizar_por_llaves_rescate(
    p,
    rescue,
    rescue_value_col,
    target_cols,
    source_col,
    source_label,
    status_label,
    allowed_tarifas=None,
    usar_periodo=True
):
    """
    Actualiza p usando rescue por llaves alternativas.

    Para kwh_total y kwmax:
        usar_periodo=True → no_servicio + medidor + periodo

    Para demanda_contratada:
        usar_periodo=False → no_servicio + medidor
    """

    if rescue.empty or rescue_value_col not in rescue.columns:
        return p

    required_keys = [
        "_key_no_servicio_enriq",
        "_key_medidor_enriq"
    ]

    if usar_periodo:
        required_keys.append("_key_periodo_enriq")

    for col in required_keys:
        if col not in rescue.columns or col not in p.columns:
            return p

    lookup = rescue[
        required_keys + [rescue_value_col]
    ].copy()

    lookup[rescue_value_col] = _to_num_enriq(
        lookup[rescue_value_col]
    )

    lookup = lookup[
        lookup[rescue_value_col].gt(0)
    ].copy()

    for col in required_keys:
        lookup = lookup[
            lookup[col].fillna("").astype(str).str.strip().ne("")
        ].copy()

    if lookup.empty:
        return p

    # Si hay más de un valor para la misma llave:
    # - para kwmax/kWh por periodo normalmente debería ser único
    # - tomamos max como criterio conservador ante duplicados
    lookup = (
        lookup
        .groupby(required_keys, dropna=False)[rescue_value_col]
        .max()
        .reset_index()
    )

    temp_col = rescue_value_col + "_match_llaves"

    lookup = lookup.rename(
        columns={
            rescue_value_col: temp_col
        }
    )

    p = p.merge(
        lookup,
        on=required_keys,
        how="left"
    )

    mask_update = p[temp_col].gt(0)

    if allowed_tarifas is not None:
        mask_update = (
            mask_update
            & p["_tarifa_enriq"].isin(allowed_tarifas)
        )

    for target_col in target_cols:
        if target_col not in p.columns:
            p[target_col] = pd.NA

        p.loc[
            mask_update,
            target_col
        ] = p.loc[
            mask_update,
            temp_col
        ]

    if source_col not in p.columns:
        p[source_col] = "parser_original"

    p.loc[
        mask_update,
        source_col
    ] = source_label

    p = _append_audit_status(
        p,
        mask_update,
        status_label
    )

    p = p.drop(
        columns=[temp_col],
        errors="ignore"
    )

    return p

def enriquecer_parser_con_archivos_rescate(
    parsed,
    pdbt_kwh_path,
    gdm_rescue_path,
    new_rows_path
):
    """
    Enriquece el parser original con 3 fuentes externas:

    1) results_PDBT_full.csv:
       - Completa / sustituye kwh_total para PDBT existentes.

    2) results_AnaTe18Jun_full_schema_v3.csv:
       - Completa / sustituye demanda_contratada_kw y kwmax
         para GDMTH, GDMTO y GDBT existentes.

    3) results_file_path_table_1_FULL_v1.csv:
       - Agrega filas nuevas que no existen en el parser.
       - También puede sustituir kwh_total, kwmax y demanda_contratada_kw
         cuando coincida por file_path.

    No muestra diagnósticos en la app. Solo agrega columnas de auditoría.
    """

    if parsed.empty:
        return parsed

    p = parsed.copy()

    # ------------------------------------------------------------
    # Asegurar columnas base
    # ------------------------------------------------------------

    if "file_path" not in p.columns:
        p["file_path"] = ""

    p["_file_path_key_enriq"] = _normalizar_file_path_enriq(
        p["file_path"]
    )

    no_servicio_parser_col = _first_existing_col(
        p,
        [
            "no_servicio",
            "No. servicio",
            "servicio"
        ]
    )

    medidor_parser_col = _first_existing_col(
        p,
        [
            "medidor",
            "MEDIDOR",
            "No. De medidor",
            "No. de medidor"
        ]
    )

    periodo_parser_col = _first_existing_col(
        p,
        [
            "periodo_fact",
            "periodo",
            "periodo_facturacion"
        ]
    )

    if no_servicio_parser_col is not None:
        p["_key_no_servicio_enriq"] = normalize_service_cc(
            p[no_servicio_parser_col]
        )
    else:
        p["_key_no_servicio_enriq"] = ""

    if medidor_parser_col is not None:
        p["_key_medidor_enriq"] = normalize_meter_cc(
            p[medidor_parser_col]
        )
    else:
        p["_key_medidor_enriq"] = ""

    if periodo_parser_col is not None:
        p["_key_periodo_enriq"] = _normalizar_periodo_enriq(
            p[periodo_parser_col]
        )
    else:
        p["_key_periodo_enriq"] = ""

    # Auditoría
    audit_defaults = {
        "kwh_total_fuente": "parser_original",
        "kwmax_fuente": "parser_original",
        "demanda_contratada_fuente": "parser_original",
        "fila_agregada_desde": "",
        "parser_enriquecido_status": ""
    }

    for col, default_value in audit_defaults.items():
        if col not in p.columns:
            p[col] = default_value
        else:
            p[col] = p[col].fillna(default_value)

    # Columnas numéricas estándar
    if "kwh_total" not in p.columns:
        p["kwh_total"] = pd.NA

    if "kwh_total_num" not in p.columns:
        p["kwh_total_num"] = _to_num_enriq(p["kwh_total"])
    else:
        p["kwh_total_num"] = _to_num_enriq(p["kwh_total_num"])

    if "kwmax" not in p.columns:
        p["kwmax"] = pd.NA

    if "kwmax_num" not in p.columns:
        p["kwmax_num"] = _to_num_enriq(p["kwmax"])
    else:
        p["kwmax_num"] = _to_num_enriq(p["kwmax_num"])

    if "demanda_contratada_kw" not in p.columns:
        p["demanda_contratada_kw"] = pd.NA

    p["demanda_contratada_kw"] = _to_num_enriq(
        p["demanda_contratada_kw"]
    )

    # Tarifa del parser
    tarifa_col_parser = _first_existing_col(
        p,
        [
            "tarifa_norm",
            "tarifa",
            "Tarifa",
            "TARIFA_FINAL"
        ]
    )

    if tarifa_col_parser:
        p["_tarifa_enriq"] = _normalizar_tarifa_enriq(
            p[tarifa_col_parser]
        )
    else:
        p["_tarifa_enriq"] = ""

    # ============================================================
    # 1) results_PDBT_full.csv
    # ============================================================

    pdbt_rescue = _preparar_rescate_csv(
        pdbt_kwh_path
    )

    if not pdbt_rescue.empty and "kwh_total" in pdbt_rescue.columns:

        pdbt_rescue = pdbt_rescue[
            [
                "_file_path_key_enriq",
                "_key_no_servicio_enriq",
                "_key_medidor_enriq",
                "_key_periodo_enriq",
                "kwh_total"
            ]
        ].copy()

        pdbt_rescue["kwh_total_rescate_pdbt"] = _to_num_enriq(
            pdbt_rescue["kwh_total"]
        )

        pdbt_rescue = pdbt_rescue[
            pdbt_rescue["kwh_total_rescate_pdbt"].gt(0)
        ].copy()

        pdbt_rescue_path_lookup = pdbt_rescue[
            [
                "_file_path_key_enriq",
                "kwh_total_rescate_pdbt"
            ]
        ].drop_duplicates(
            subset=["_file_path_key_enriq"],
            keep="last"
        )

        p = p.merge(
            pdbt_rescue_path_lookup,
            on="_file_path_key_enriq",
            how="left"
        )

        mask_pdbt_update = (
            p["_tarifa_enriq"].eq("PDBT")
            & p["kwh_total_rescate_pdbt"].gt(0)
        )

        p.loc[
            mask_pdbt_update,
            "kwh_total"
        ] = p.loc[
            mask_pdbt_update,
            "kwh_total_rescate_pdbt"
        ]

        p.loc[
            mask_pdbt_update,
            "kwh_total_num"
        ] = p.loc[
            mask_pdbt_update,
            "kwh_total_rescate_pdbt"
        ]

        p.loc[
            mask_pdbt_update,
            "kwh_total_fuente"
        ] = "results_PDBT_full.csv"

        p = _append_audit_status(
            p,
            mask_pdbt_update,
            "kwh_total_rescatado_pdbt"
        )

        p = p.drop(
            columns=[
                "kwh_total_rescate_pdbt"
            ],
            errors="ignore"
        )

        # Fallback: PDBT por no_servicio + medidor + periodo_fact.
        # Esto evita depender solo de file_path cuando la ruta cambió.
        p = _actualizar_por_llaves_rescate(
            p=p,
            rescue=pdbt_rescue,
            rescue_value_col="kwh_total_rescate_pdbt",
            target_cols=[
                "kwh_total",
                "kwh_total_num"
            ],
            source_col="kwh_total_fuente",
            source_label="results_PDBT_full.csv",
            status_label="kwh_total_rescatado_pdbt_por_servicio_medidor_periodo",
            allowed_tarifas=[
                "PDBT"
            ],
            usar_periodo=True
        )

    # ============================================================
    # 2) results_AnaTe18Jun_full_schema_v3.csv
    # ============================================================

    gdm_rescue = _preparar_rescate_csv(
        gdm_rescue_path
    )

    if not gdm_rescue.empty:

        cols_gdm = [
            "_file_path_key_enriq",
            "_key_no_servicio_enriq",
            "_key_medidor_enriq",
            "_key_periodo_enriq"
        ]

        for col in [
            "demanda_contratada_kw",
            "kwmax",
            "kwh_total"
        ]:
            if col in gdm_rescue.columns:
                cols_gdm.append(col)

        gdm_rescue = gdm_rescue[cols_gdm].copy()

        if "demanda_contratada_kw" in gdm_rescue.columns:
            gdm_rescue["demanda_contratada_rescate_gdm"] = _to_num_enriq(
                gdm_rescue["demanda_contratada_kw"]
            )

        if "kwmax" in gdm_rescue.columns:
            gdm_rescue["kwmax_rescate_gdm"] = _to_num_enriq(
                gdm_rescue["kwmax"]
            )

        if "kwh_total" in gdm_rescue.columns:
            gdm_rescue["kwh_total_rescate_gdm"] = _to_num_enriq(
                gdm_rescue["kwh_total"]
            )

        keep_cols_gdm = [
            "_file_path_key_enriq",
            "_key_no_servicio_enriq",
            "_key_medidor_enriq",
            "_key_periodo_enriq"
        ]

        for col in [
            "demanda_contratada_rescate_gdm",
            "kwmax_rescate_gdm",
            "kwh_total_rescate_gdm"
        ]:
            if col in gdm_rescue.columns:
                keep_cols_gdm.append(col)

        gdm_rescue = (
            gdm_rescue[keep_cols_gdm]
            .drop_duplicates(
                subset=["_file_path_key_enriq"],
                keep="last"
            )
        )

        p = p.merge(
            gdm_rescue,
            on="_file_path_key_enriq",
            how="left"
        )

        mask_tarifa_medida = p["_tarifa_enriq"].isin(
            [
                "GDMTH",
                "GDMTO",
                "GDBT"
            ]
        )

        if "demanda_contratada_rescate_gdm" in p.columns:
            mask_demanda_contratada_gdm = (
                mask_tarifa_medida
                & p["demanda_contratada_rescate_gdm"].gt(0)
            )

            p.loc[
                mask_demanda_contratada_gdm,
                "demanda_contratada_kw"
            ] = p.loc[
                mask_demanda_contratada_gdm,
                "demanda_contratada_rescate_gdm"
            ]

            p.loc[
                mask_demanda_contratada_gdm,
                "demanda_contratada_fuente"
            ] = "results_AnaTe18Jun_full_schema_v3.csv"

            p = _append_audit_status(
                p,
                mask_demanda_contratada_gdm,
                "demanda_contratada_rescatada_gdm"
            )

        if "kwmax_rescate_gdm" in p.columns:
            mask_kwmax_gdm = (
                mask_tarifa_medida
                & p["kwmax_rescate_gdm"].gt(0)
            )

            p.loc[
                mask_kwmax_gdm,
                "kwmax"
            ] = p.loc[
                mask_kwmax_gdm,
                "kwmax_rescate_gdm"
            ]

            p.loc[
                mask_kwmax_gdm,
                "kwmax_num"
            ] = p.loc[
                mask_kwmax_gdm,
                "kwmax_rescate_gdm"
            ]

            p.loc[
                mask_kwmax_gdm,
                "kwmax_fuente"
            ] = "results_AnaTe18Jun_full_schema_v3.csv"

            p = _append_audit_status(
                p,
                mask_kwmax_gdm,
                "kwmax_rescatado_gdm"
            )

        if "kwh_total_rescate_gdm" in p.columns:
            mask_kwh_gdm = (
                mask_tarifa_medida
                & p["kwh_total_rescate_gdm"].gt(0)
            )

            p.loc[
                mask_kwh_gdm,
                "kwh_total"
            ] = p.loc[
                mask_kwh_gdm,
                "kwh_total_rescate_gdm"
            ]

            p.loc[
                mask_kwh_gdm,
                "kwh_total_num"
            ] = p.loc[
                mask_kwh_gdm,
                "kwh_total_rescate_gdm"
            ]

            p.loc[
                mask_kwh_gdm,
                "kwh_total_fuente"
            ] = "results_AnaTe18Jun_full_schema_v3.csv"

            p = _append_audit_status(
                p,
                mask_kwh_gdm,
                "kwh_total_rescatado_gdm"
            )

        p = p.drop(
            columns=[
                "demanda_contratada_rescate_gdm",
                "kwmax_rescate_gdm",
                "kwh_total_rescate_gdm"
            ],
            errors="ignore"
        )

        # Fallback: kwmax por no_servicio + medidor + periodo_fact.
        p = _actualizar_por_llaves_rescate(
            p=p,
            rescue=gdm_rescue,
            rescue_value_col="kwmax_rescate_gdm",
            target_cols=[
                "kwmax",
                "kwmax_num"
            ],
            source_col="kwmax_fuente",
            source_label="results_AnaTe18Jun_full_schema_v3.csv",
            status_label="kwmax_rescatado_gdm_por_servicio_medidor_periodo",
            allowed_tarifas=[
                "GDMTH",
                "GDMTO",
                "GDBT"
            ],
            usar_periodo=True
        )

        # Fallback: kWh por no_servicio + medidor + periodo_fact.
        p = _actualizar_por_llaves_rescate(
            p=p,
            rescue=gdm_rescue,
            rescue_value_col="kwh_total_rescate_gdm",
            target_cols=[
                "kwh_total",
                "kwh_total_num"
            ],
            source_col="kwh_total_fuente",
            source_label="results_AnaTe18Jun_full_schema_v3.csv",
            status_label="kwh_total_rescatado_gdm_por_servicio_medidor_periodo",
            allowed_tarifas=[
                "GDMTH",
                "GDMTO",
                "GDBT"
            ],
            usar_periodo=True
        )

        # Fallback: demanda contratada por no_servicio + medidor.
        # Aquí NO usamos periodo porque la demanda contratada es del servicio,
        # no del consumo de cada periodo.
        p = _actualizar_por_llaves_rescate(
            p=p,
            rescue=gdm_rescue,
            rescue_value_col="demanda_contratada_rescate_gdm",
            target_cols=[
                "demanda_contratada_kw"
            ],
            source_col="demanda_contratada_fuente",
            source_label="results_AnaTe18Jun_full_schema_v3.csv",
            status_label="demanda_contratada_rescatada_gdm_por_servicio_medidor",
            allowed_tarifas=[
                "GDMTH",
                "GDMTO",
                "GDBT"
            ],
            usar_periodo=False
        )

    # ============================================================
    # 3) results_file_path_table_1_FULL_v1.csv
    # ============================================================

    new_rows = _preparar_rescate_csv(
        new_rows_path
    )

    if not new_rows.empty:

        # ------------------------------------------------------------
        # 3A) Primero usarlo para actualizar filas existentes
        # ------------------------------------------------------------

        update_cols = [
            "_file_path_key_enriq",
            "_key_no_servicio_enriq",
            "_key_medidor_enriq",
            "_key_periodo_enriq"
        ]

        for col in [
            "kwh_total",
            "kwmax",
            "demanda_contratada_kw"
        ]:
            if col in new_rows.columns:
                update_cols.append(col)

        new_rows_update = new_rows[update_cols].copy()

        if "kwh_total" in new_rows_update.columns:
            new_rows_update["kwh_total_rescate_full"] = _to_num_enriq(
                new_rows_update["kwh_total"]
            )

        if "kwmax" in new_rows_update.columns:
            new_rows_update["kwmax_rescate_full"] = _to_num_enriq(
                new_rows_update["kwmax"]
            )

        if "demanda_contratada_kw" in new_rows_update.columns:
            new_rows_update["demanda_contratada_rescate_full"] = _to_num_enriq(
                new_rows_update["demanda_contratada_kw"]
            )

        keep_cols_full = [
            "_file_path_key_enriq"
        ]

        for col in [
            "kwh_total_rescate_full",
            "kwmax_rescate_full",
            "demanda_contratada_rescate_full"
        ]:
            if col in new_rows_update.columns:
                keep_cols_full.append(col)

        new_rows_update = (
            new_rows_update[keep_cols_full]
            .drop_duplicates(
                subset=["_file_path_key_enriq"],
                keep="last"
            )
        )

        p = p.merge(
            new_rows_update,
            on="_file_path_key_enriq",
            how="left"
        )

        if "kwh_total_rescate_full" in p.columns:
            mask_kwh_full = p["kwh_total_rescate_full"].gt(0)

            p.loc[
                mask_kwh_full,
                "kwh_total"
            ] = p.loc[
                mask_kwh_full,
                "kwh_total_rescate_full"
            ]

            p.loc[
                mask_kwh_full,
                "kwh_total_num"
            ] = p.loc[
                mask_kwh_full,
                "kwh_total_rescate_full"
            ]

            p.loc[
                mask_kwh_full,
                "kwh_total_fuente"
            ] = "results_file_path_table_1_FULL_v1.csv"

            p = _append_audit_status(
                p,
                mask_kwh_full,
                "kwh_total_rescatado_full"
            )

        if "kwmax_rescate_full" in p.columns:
            mask_kwmax_full = p["kwmax_rescate_full"].gt(0)

            p.loc[
                mask_kwmax_full,
                "kwmax"
            ] = p.loc[
                mask_kwmax_full,
                "kwmax_rescate_full"
            ]

            p.loc[
                mask_kwmax_full,
                "kwmax_num"
            ] = p.loc[
                mask_kwmax_full,
                "kwmax_rescate_full"
            ]

            p.loc[
                mask_kwmax_full,
                "kwmax_fuente"
            ] = "results_file_path_table_1_FULL_v1.csv"

            p = _append_audit_status(
                p,
                mask_kwmax_full,
                "kwmax_rescatado_full"
            )

        if "demanda_contratada_rescate_full" in p.columns:
            mask_demanda_full = p["demanda_contratada_rescate_full"].gt(0)

            p.loc[
                mask_demanda_full,
                "demanda_contratada_kw"
            ] = p.loc[
                mask_demanda_full,
                "demanda_contratada_rescate_full"
            ]

            p.loc[
                mask_demanda_full,
                "demanda_contratada_fuente"
            ] = "results_file_path_table_1_FULL_v1.csv"

            p = _append_audit_status(
                p,
                mask_demanda_full,
                "demanda_contratada_rescatada_full"
            )

        p = p.drop(
            columns=[
                "kwh_total_rescate_full",
                "kwmax_rescate_full",
                "demanda_contratada_rescate_full"
            ],
            errors="ignore"
        )

        # Fallback: kWh por no_servicio + medidor + periodo.
        p = _actualizar_por_llaves_rescate(
            p=p,
            rescue=new_rows_update,
            rescue_value_col="kwh_total_rescate_full",
            target_cols=[
                "kwh_total",
                "kwh_total_num"
            ],
            source_col="kwh_total_fuente",
            source_label="results_file_path_table_1_FULL_v1.csv",
            status_label="kwh_total_rescatado_full_por_servicio_medidor_periodo",
            allowed_tarifas=None,
            usar_periodo=True
        )

        # Fallback: kwmax por no_servicio + medidor + periodo.
        p = _actualizar_por_llaves_rescate(
            p=p,
            rescue=new_rows_update,
            rescue_value_col="kwmax_rescate_full",
            target_cols=[
                "kwmax",
                "kwmax_num"
            ],
            source_col="kwmax_fuente",
            source_label="results_file_path_table_1_FULL_v1.csv",
            status_label="kwmax_rescatado_full_por_servicio_medidor_periodo",
            allowed_tarifas=None,
            usar_periodo=True
        )

        # Fallback: demanda contratada por no_servicio + medidor.
        p = _actualizar_por_llaves_rescate(
            p=p,
            rescue=new_rows_update,
            rescue_value_col="demanda_contratada_rescate_full",
            target_cols=[
                "demanda_contratada_kw"
            ],
            source_col="demanda_contratada_fuente",
            source_label="results_file_path_table_1_FULL_v1.csv",
            status_label="demanda_contratada_rescatada_full_por_servicio_medidor",
            allowed_tarifas=None,
            usar_periodo=False
        )

        # ------------------------------------------------------------
        # 3B) Después agregar filas nuevas que no existen en el parser
        # ------------------------------------------------------------

        existing_keys = set(
            p["_file_path_key_enriq"]
            .dropna()
            .astype(str)
            .unique()
        )

        new_rows_to_append = new_rows[
            ~new_rows["_file_path_key_enriq"].isin(existing_keys)
        ].copy()

        if not new_rows_to_append.empty:

            # Asegurar mismas columnas en ambos lados
            for col in p.columns:
                if col not in new_rows_to_append.columns:
                    new_rows_to_append[col] = pd.NA

            for col in new_rows_to_append.columns:
                if col not in p.columns:
                    p[col] = pd.NA

            new_rows_to_append = new_rows_to_append[
                p.columns
            ].copy()

            new_rows_to_append["fila_agregada_desde"] = (
                "results_file_path_table_1_FULL_v1.csv"
            )

            new_rows_to_append["kwh_total_fuente"] = np.where(
                _to_num_enriq(new_rows_to_append["kwh_total"]).gt(0),
                "results_file_path_table_1_FULL_v1.csv",
                "sin_kwh"
            )

            new_rows_to_append["kwmax_fuente"] = np.where(
                _to_num_enriq(new_rows_to_append["kwmax"]).gt(0),
                "results_file_path_table_1_FULL_v1.csv",
                "sin_kwmax"
            )

            new_rows_to_append["demanda_contratada_fuente"] = np.where(
                _to_num_enriq(
                    new_rows_to_append["demanda_contratada_kw"]
                ).gt(0),
                "results_file_path_table_1_FULL_v1.csv",
                "sin_demanda_contratada"
            )

            new_rows_to_append["parser_enriquecido_status"] = (
                "fila_nueva_results_file_path_table_1_FULL_v1"
            )

            p = pd.concat(
                [
                    p,
                    new_rows_to_append
                ],
                ignore_index=True
            )

    # ============================================================
    # Derivados finales después de todos los rescates
    # ============================================================

    p["kwh_total_num"] = _to_num_enriq(
        p["kwh_total"]
    )

    p["kwmax_num"] = _to_num_enriq(
        p["kwmax"]
    )

    p["demanda_contratada_kw"] = _to_num_enriq(
        p["demanda_contratada_kw"]
    )

    tarifa_col_final = _first_existing_col(
        p,
        [
            "tarifa_norm",
            "tarifa",
            "Tarifa",
            "TARIFA_FINAL"
        ]
    )

    if tarifa_col_final:
        p["_tarifa_enriq"] = _normalizar_tarifa_enriq(
            p[tarifa_col_final]
        )
    else:
        p["_tarifa_enriq"] = ""

    mask_tarifa_medida_final = p["_tarifa_enriq"].isin(
        [
            "GDMTH",
            "GDMTO",
            "GDBT"
        ]
    )

    mask_kwmax_valido_final = p["kwmax_num"].gt(0)

    if "demanda_maxima_mensual_kw" not in p.columns:
        p["demanda_maxima_mensual_kw"] = pd.NA

    p.loc[
        mask_tarifa_medida_final & mask_kwmax_valido_final,
        "demanda_maxima_mensual_kw"
    ] = p.loc[
        mask_tarifa_medida_final & mask_kwmax_valido_final,
        "kwmax_num"
    ]

    # Compatibilidad con diagnósticos que todavía usen demanda_real_kw
    if "demanda_real_kw" not in p.columns:
        p["demanda_real_kw"] = pd.NA

    p.loc[
        mask_tarifa_medida_final & mask_kwmax_valido_final,
        "demanda_real_kw"
    ] = p.loc[
        mask_tarifa_medida_final & mask_kwmax_valido_final,
        "kwmax_num"
    ]

    if "criterio_demanda_mensual" not in p.columns:
        p["criterio_demanda_mensual"] = pd.NA

    p.loc[
        mask_tarifa_medida_final & mask_kwmax_valido_final,
        "criterio_demanda_mensual"
    ] = "kwmax parser enriquecido"

    p = p.drop(
        columns=[
            "_file_path_key_enriq",
            "_key_no_servicio_enriq",
            "_key_medidor_enriq",
            "_key_periodo_enriq",
            "_tarifa_enriq"
        ],
        errors="ignore"
    )

    return p

def normalize_tarifa_series(s):
    return s.apply(normalize_tarifa_value)

def normalizar_texto_simple(value):
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .upper()
        .replace("Á", "A")
        .replace("É", "E")
        .replace("Í", "I")
        .replace("Ó", "O")
        .replace("Ú", "U")
        .replace("Ñ", "N")
    )

def normalize_name_for_match(value):
    value = normalizar_texto_simple(value)

    value = value.upper().strip()

    # Limpieza de caracteres
    value = re.sub(r"[^A-Z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    # Remover sufijos legales comunes
    legal_terms = [
        "SA DE CV",
        "S A DE C V",
        "SAPI DE CV",
        "S A P I DE C V",
        "S DE RL DE CV",
        "S DE R L DE C V",
        "INSTITUCION DE BANCA MULTIPLE",
        "INST DE BANCA MULT",
        "GRUPO FINANCIERO",
        "SOCIEDAD ANONIMA",
        "DE CV",
        "SA",
        "CV",
        "SAPI",
        "S DE RL",
    ]

    for term in legal_terms:
        value = value.replace(term, " ")

    # Normalizaciones específicas útiles
    replacements = {
        "SMART FIT": "SMARTFIT",
        "CUIDADO CON EL PERRO": "CUIDADO CON EL PERRO",
        "DOLPHY": "DOLPHY",
        "DOLPH Y": "DOLPHY",
        "CINEMEX MORELIA": "CINEMEX",
    }

    for old, new in replacements.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value).strip()

    return value

def normalize_brand_name(value):
    value = normalizar_texto_simple(value)
    value = value.upper().strip()

    value = re.sub(r"[^A-Z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()

    legal_terms = [
        "SA DE CV",
        "S A DE C V",
        "SAPI DE CV",
        "S A P I DE C V",
        "S DE RL DE CV",
        "S DE R L DE C V",
        "DE CV",
        "SA",
        "CV",
        "SAPI",
        "S DE RL",
        "GRUPO FINANCIERO",
        "INSTITUCION DE BANCA MULTIPLE",
        "INST DE BANCA MULT",
    ]

    for term in legal_terms:
        value = value.replace(term, " ")

    alias_map = {
        "CUIDADO CON EL PERRO": "CCP",
        "SMART FIT": "SMARTFIT",
        "SMARTFIT": "SMARTFIT",
        "DOLPHY": "DOLPHY",
        "HELADOS Y DONAS DE MEXICO": "DOLPHY",
    }

    for old, new in alias_map.items():
        value = value.replace(old, new)

    value = re.sub(r"\s+", " ", value).strip()

    return value

def normalize_person_name_unordered(value):
    """
    Normaliza nombres de persona sin importar el orden.

    Ejemplo:
    AVRIL ANTON AGUILAR
    ANTON AGUILAR AVRIL

    Ambos quedan como:
    AGUILAR ANTON AVRIL
    """

    if pd.isna(value):
        return ""

    value = normalize_brand_name(value)

    words = [
        w for w in value.split()
        if len(w) >= 3
        and w not in [
            "SA", "CV", "SAPI", "GRUPO", "COMERCIAL",
            "OPERADORA", "SERVICIOS", "TIENDAS"
        ]
    ]

    words = sorted(set(words))

    return " ".join(words)


def same_person_name_unordered(a, b):
    a_key = normalize_person_name_unordered(a)
    b_key = normalize_person_name_unordered(b)

    if not a_key or not b_key:
        return False

    a_words = a_key.split()
    b_words = b_key.split()

    # Pedimos al menos 2 palabras para evitar falsos positivos.
    if len(a_words) < 2 or len(b_words) < 2:
        return False

    return a_key == b_key

def has_partial_match_cc(value, candidates):
    if not value:
        return False

    value_norm = normalize_brand_name(value)

    if not value_norm:
        return False

    value_compact = value_norm.replace(" ", "")

    value_words = set(value_norm.split())

    for candidate in candidates:
        if not candidate:
            continue

        candidate_norm = normalize_brand_name(candidate)
        candidate_compact = candidate_norm.replace(" ", "")
        candidate_words = set(candidate_norm.split())

        if not candidate_norm:
            continue

        # 1. Exacto
        if value_norm == candidate_norm:
            return True

        # 1.1 Exacto por palabras sin importar el orden
        # Ejemplo: AVRIL ANTON AGUILAR = ANTON AGUILAR AVRIL
        if same_person_name_unordered(value_norm, candidate_norm):
            return True

        # 2. Compacto exacto: SMART FIT vs SMARTFIT
        if value_compact == candidate_compact:
            return True

        # 3. Uno contenido en el otro: DOLPHY vs ALAIA GUANAJUATO DOLPHY
        if value_compact in candidate_compact or candidate_compact in value_compact:
            return True


        # 4. Coincidencia por palabras fuertes
        common_words = value_words & candidate_words

        stop_match_words = {
            "GRUPO", "MEXICO", "COMERCIAL", "COMERCIALIZADORA",
            "OPERADORA", "SERVICIOS", "TIENDAS", "ALMACENES",
            "PROMOCIONES", "INMOBILIARIAS", "RESTAURANTES",
            "FRANQUICIAS", "EMPRESAS", "COMPANY"
        }

        strong_common_words = {
            w for w in common_words
            if len(w) >= 5
            and w not in stop_match_words
        }

        # Solo acepta por palabras si hay al menos 2 palabras fuertes
        if len(strong_common_words) >= 2:
            return True

    return False

def limpiar_nombre_cc(nombre):
    if pd.isna(nombre):
        return ""

    nombre = str(nombre).strip()

    # Quita prefijos tipo: 01. V1_ / 01_V1_ / V1_
    nombre = re.sub(r"^\s*\d+\s*[\.\-_]\s*", "", nombre)
    nombre = re.sub(r"^\s*V\d+[_\-\s]*", "", nombre, flags=re.IGNORECASE)

    # Reemplaza guiones bajos por espacios
    nombre = nombre.replace("_", " ")

    # Limpia espacios dobles
    nombre = re.sub(r"\s+", " ", nombre).strip()

    return nombre

CC_NAME_MAP = {
    "AMBAR FASHION MALL TUXTLA": "AMBAR FASHION MALL TUXTLA",
    "FORUM TLAQUEPAQUE": "FORUM TLAQUEPAQUE",
    "LA ISLA VALLARTA": "LA ISLA VALLARTA",
    "MIDTOWN JALISCO": "MIDTOWN JALISCO",
    "MITIKAH": "MITIKAH",
    "OUTLET LERMA": "OUTLET LERMA",
    "PABELLON CUEMANCO": "PABELLON CUEMANCO",
    "ESPACIO AGUASCALIENTES": "ESPACIO AGUASCALIENTES",
    "PABELLON RIO DE LOS REMEDIOS": "PABELLON RIO DE LOS REMEDIOS",
    "PABELLON SALINA CRUZ": "PABELLON SALINA CRUZ",
    "SENDERO VILLAHERMOSA": "SENDERO VILLAHERMOSA",
    "PATIO TOLUCA": "PATIO TOLUCA",
    "UPTOWN MERIDA": "UPTOWN MERIDA",
    "ALAIA GUANAJUATO": "ALAIA GUANAJUATO",
    "ALAIA TAPACHULA": "ALAIA TAPACHULA",
    "PLAZA CENTRAL": "PLAZA CENTRAL",
    "SAMARA SATELITE": "SAMARA SATELITE",
    "UPTOWN JURIQUILLA": "UPTOWN JURIQUILLA",
    "PABELLON NAVOJOA": "PABELLON NAVOJOA",
}


def cc_key(nombre):
    nombre = limpiar_nombre_cc(nombre)
    nombre = normalizar_texto_simple(nombre)
    nombre = nombre.replace("_", " ")
    nombre = re.sub(r"\s+", " ", nombre).strip()

    for key in CC_NAME_MAP.keys():
        if key in nombre or nombre in key:
            return CC_NAME_MAP[key]

    return nombre

def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Returns the first column name that exists in a dataframe.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def preferred_col_index(columns, preferred_names):
    """
    Returns the index of the first preferred column found, otherwise 0.
    Useful for Streamlit selectbox default index.
    """
    cols = list(columns)
    for name in preferred_names:
        if name in cols:
            return cols.index(name)
    return 0


def prepare_parsed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes important columns from bills_parsed_v2.csv.
    """
    if df.empty:
        return df

    df = df.copy()

    numeric_cols = [
        "kwh_total",
        "importe_total",
        "subtotal",
        "iva",
        "cargo_fijo",
        "subtotal_energia",
        "fac_del_periodo",
        "total_linea",
        "lectura_actual_kwh",
        "lectura_anterior_kwh",
        "multiplicador",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col + "_num"] = clean_number_series(df[col])

    date_cols = [
        "limite_pago",
        "corte_a_partir",
        "periodo_inicio",
        "periodo_fin",
        "parsed_at",
    ]

    for col in date_cols:
        if col in df.columns:
            df[col + "_dt"] = clean_date_series(df[col])

    if "importe_total_num" in df.columns and "kwh_total_num" in df.columns:
        df["mxn_per_kwh"] = df["importe_total_num"] / df["kwh_total_num"]
        df.loc[df["kwh_total_num"] <= 0, "mxn_per_kwh"] = pd.NA

    return df


def prepare_historico_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes important columns from bills_historico_v2.csv.
    """
    if df.empty:
        return df

    df = df.copy()

    numeric_candidates = [
        "kwh",
        "consumo_kwh",
        "kwh_total",
        "consumo",
        "energia_kwh",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col + "_num"] = clean_number_series(df[col])

    date_candidates = [
        "fecha",
        "periodo",
        "mes",
        "periodo_inicio",
        "periodo_fin",
    ]

    for col in date_candidates:
        if col in df.columns:
            df[col + "_dt"] = clean_date_series(df[col])

    return df

def aplicar_override_iberdrola_liverpool(parsed: pd.DataFrame) -> pd.DataFrame:
    """
    Override manual:

    Todo recibo cuyo source_utility sea IBERDROLA debe asignarse
    al local Liverpool de Ambar Fashion Mall Tuxtla en DG.

    Regla:
    source_utility = IBERDROLA
    -> mall_folder = Ambar Fashion Mall Tuxtla
    -> cliente_nombre = OPERADORA DE ALMACENES LIVERPOOL
    -> recibos_subgroup = LIVERPOOL
    -> tenant = LIVERPOOL
    -> locatario = LIVERPOOL
    """

    if parsed.empty:
        return parsed

    p = parsed.copy()

    if "source_utility" not in p.columns:
        return p

    mask_iberdrola = (
        p["source_utility"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .eq("IBERDROLA")
    )

    if not mask_iberdrola.any():
        return p

    # Asegurar columnas necesarias para el match global
    for col in [
        "mall_folder",
        "cliente_nombre",
        "recibos_subgroup",
        "tenant",
        "locatario",
        "override_match_dg"
    ]:
        if col not in p.columns:
            p[col] = pd.NA

    # Forzar centro comercial
    p.loc[
        mask_iberdrola,
        "mall_folder"
    ] = "Ambar Fashion Mall Tuxtla"

    # Forzar cliente de DG
    p.loc[
        mask_iberdrola,
        "cliente_nombre"
    ] = "OPERADORA DE ALMACENES LIVERPOOL"

    # Forzar nombre comercial de DG
    p.loc[
        mask_iberdrola,
        "recibos_subgroup"
    ] = "LIVERPOOL"

    # Respaldos por si otras partes de la app usan estas columnas
    p.loc[
        mask_iberdrola,
        "tenant"
    ] = "LIVERPOOL"

    p.loc[
        mask_iberdrola,
        "locatario"
    ] = "LIVERPOOL"

    # Auditoría visible para saber que fue forzado manualmente
    p.loc[
        mask_iberdrola,
        "override_match_dg"
    ] = "source_utility IBERDROLA → Liverpool / Ambar Tuxtla"

    return p


def compact_carpeta_key(value):
    return re.sub(
        r"[^A-Z0-9]",
        "",
        normalizar_carpeta_key(value)
    )


def prepare_carpetas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte Carpetas.xlsx de formato ancho a formato largo.

    El archivo viene así:
    - cada columna = centro comercial
    - cada celda debajo = carpeta / recibo original

    Devuelve:
    - Centro Comercial
    - Carpeta registro externo
    - carpeta_key
    - carpeta_compact
    """

    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    rows = []

    for col_cc in d.columns:

        if col_cc in ["source_sheet"]:
            continue

        centro = limpiar_nombre_cc(col_cc)
        centro_key = cc_key(centro)

        # Excluir Ambar y Tapachula, porque dijiste que no aplican
        # para este registro externo.
        if centro_key in [
            "AMBAR FASHION MALL TUXTLA",
            "ALAIA TAPACHULA"
        ]:
            continue

        for value in d[col_cc].dropna().astype(str).tolist():

            carpeta = value.strip()

            if carpeta == "":
                continue

            carpeta_key = normalizar_carpeta_key(carpeta)
            carpeta_compact = compact_carpeta_key(carpeta)

            if carpeta_compact == "":
                continue

            rows.append({
                "Centro Comercial": centro_key,
                "Carpeta registro externo": carpeta,
                "carpeta_key": carpeta_key,
                "carpeta_compact": carpeta_compact
            })

    return pd.DataFrame(rows)

def prepare_general_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes general-data table.
    """
    if df.empty:
        return df

    df = df.copy()

    # Normalize common area columns if they exist.
    possible_area_cols = [
        "MTS2",
        "M2",
        "m2",
        "MTS 2",
        "MTS²",
        "AREA_M2",
        "AREA M2",
        "SUPERFICIE",
        "SUPERFICIE M2",
    ]

    for col in possible_area_cols:
        if col in df.columns:
            df[col + "_num"] = clean_number_series(df[col])

    # Normalize common power columns if they exist.
    possible_power_cols = [
        "Carga Conectada (kW)",
        "Demanda Contratada (kW)",
        "CONSUMO REAL",
    ]

    for col in possible_power_cols:
        if col in df.columns:
            df[col + "_num"] = clean_number_series(df[col])

    return df


def format_number(x, decimals=2):
    if pd.isna(x):
        return "—"
    return f"{x:,.{decimals}f}"


def format_money(x):
    if pd.isna(x):
        return "—"
    return f"${x:,.2f}"


def format_money_compact(x):
    """
    Compact money format to avoid clipped Streamlit metric cards.
    """
    if pd.isna(x):
        return "—"

    x = float(x)

    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:,.2f} M"

    if abs(x) >= 1_000:
        return f"${x / 1_000:,.2f} K"

    return f"${x:,.2f}"


def format_mxn_per_kwh(x):
    if pd.isna(x):
        return "—"
    return f"${x:,.2f}"

NOTA_DEMANDA_PROMEDIO_ANUAL = (
    "Nota: Demanda promedio anual (kW) = promedio de las demandas máximas "
    "de los recibos más recientes disponibles para cada servicio/local. "
    "Si el servicio tiene recibos bimestrales, se usan hasta 6 recibos; "
    "si tiene recibos mensuales, se usan hasta 12 recibos. "
    "Cuando no existe información de un año completo, el promedio se calcula "
    "con los recibos disponibles. "
    "Para GDMTH, GDMTO y GDBT se toma el kwmax del parser; "
    "para PDBT se usa la demanda estimada con perfiles NREL cuando existe información suficiente."
)

def mostrar_pie_composicion_fijo(
    etiquetas,
    valores
):
    """
    Gráfico de pastel con tamaño fijo y canvas compacto.
    Mantiene el mismo diámetro visual para tarifa y giro comercial,
    sin generar espacios blancos verticales innecesarios.
    """

    # Canvas más bajo: elimina los espacios blancos de arriba y abajo.
    fig = plt.figure(figsize=(5.2, 3.9), dpi=110)

    # El eje ocupa prácticamente toda la altura del canvas.
    # Mantiene espacio horizontal para etiquetas laterales.
    ax = fig.add_axes([0.12, 0.02, 0.76, 0.96])

    ax.pie(
        valores,
        labels=etiquetas,
        autopct="%1.1f%%",
        startangle=90,
        radius=1.0,
        labeldistance=1.08,
        pctdistance=0.67,
        textprops={
            "fontsize": 7
        }
    )

    # Sin título vertical de eje.
    ax.set_ylabel("")

    # Ventana fija para conservar el mismo diámetro del círculo
    # tanto para tarifas como para giros.
    ax.set_xlim(-1.34, 1.34)
    ax.set_ylim(-1.23, 1.23)
    ax.set_aspect("equal")
    ax.axis("off")

    buffer = BytesIO()

    # No usar bbox_inches="tight": alteraría el tamaño entre gráficas.
    fig.savefig(
        buffer,
        format="png",
        dpi=110,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(fig)

    return buffer.getvalue()

def mostrar_pie_fijo(
    valores,
    etiqueta_superior,
    etiqueta_inferior
):
    """
    Renderiza un gráfico de pastel con canvas fijo.
    Así Streamlit no recorta distinto cada gráfica según las etiquetas.
    """

    fig = plt.figure(figsize=(6, 6), dpi=120)

    # El eje ocupa exactamente la misma zona del canvas en ambas gráficas.
    ax = fig.add_axes([0.10, 0.10, 0.80, 0.80])

    ax.pie(
        valores,
        labels=None,
        autopct="%1.1f%%",
        startangle=90,
        radius=1.0,
        pctdistance=0.67,
        textprops={
            "fontsize": 8
        }
    )

    # Mismo espacio visible para todas las gráficas.
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal")
    ax.axis("off")

    # Etiquetas en posiciones idénticas dentro del mismo canvas.
    ax.text(
        1.10,
        1.08,
        etiqueta_superior,
        ha="center",
        va="center",
        fontsize=7.5
    )

    ax.text(
        -1.10,
        -1.12,
        etiqueta_inferior,
        ha="center",
        va="center",
        fontsize=7.5
    )

    buffer = BytesIO()

    # Importante: NO usar bbox_inches="tight".
    # El canvas queda fijo en ambas gráficas.
    fig.savefig(
        buffer,
        format="png",
        dpi=120,
        facecolor="white",
        bbox_inches=None,
        pad_inches=0
    )

    plt.close(fig)

    return buffer.getvalue()

@st.cache_data(show_spinner=False)
def construir_perfil_mensual_nrel(profile_path):

    profile_path = Path(profile_path)

    if not profile_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(profile_path)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    energy_col = "out.electricity.total.energy_consumption.kwh"

    df["mes"] = df["timestamp"].dt.month
    df["hora"] = df["timestamp"].dt.hour

    perfil_mensual = (
        df.groupby(["mes", "hora"])[energy_col]
        .mean()
        .reset_index(name="peso")
    )

    # Normaliza cada mes como perfil de un día promedio:
    # la suma de las 24 horas del mes = 1 día típico
    perfil_mensual["peso_normalizado"] = (
        perfil_mensual.groupby("mes")["peso"]
        .transform(lambda x: x / x.sum())
    )

    perfil_mensual["peso_max_horario_mes"] = (
        perfil_mensual.groupby("mes")["peso_normalizado"]
        .transform("max")
    )

    return perfil_mensual

    return perfil_mensual

def obtener_tipo_perfil_nrel_pdbt(subgiro_comercial, tipo_local):
    """
    Asigna un perfil NREL aproximado según el giro del local.

    Ajustaremos este mapeo conforme revisemos los giros reales de Datos Generales.
    """

    subgiro = normalizar_texto_simple(subgiro_comercial)
    tipo = normalizar_texto_simple(tipo_local)

    if "ALIMENTOS" in subgiro or "BEBIDAS" in subgiro or "RESTAURANTE" in subgiro:
        if "FOOD COURT" in tipo or "COMIDA RAPIDA" in subgiro or "FAST FOOD" in subgiro:
            return "quickservicerestaurant", "Comida rápida (Quick Service Restaurant)"

        return "fullservicerestaurant", "Restaurante (Full Service Restaurant)"

    if (
        "TIENDA DEPARTAMENTAL" in subgiro
        or "TIENDAS DEPARTAMENTALES" in subgiro
        or "ANCLA" in tipo
    ):
        return "retailstandalone", "Tienda departamental / ancla"

    if "SUPERMERCADO" in subgiro or "AUTOSERVICIO" in subgiro:
        return "retailstandalone", "Supermercado / autoservicio"

    if "CINE" in subgiro:
        return "retailstripmall", "Cine aproximado como local comercial"

    if "GIMNASIO" in subgiro:
        return "retailstripmall", "Gimnasio aproximado como local comercial"

    return "retailstripmall", "Local comercial general"


def obtener_zona_nrel_por_cc_pdbt(centro_comercial, climate_mapping_df):
    """
    Busca la zona NREL del centro comercial en cc_master_data.csv.

    Versión flexible:
    - No depende de un solo nombre de columna.
    - Busca columnas que parezcan centro comercial.
    - Busca columnas que parezcan zona NREL / zona climática.
    - Compara usando cc_key().
    - Permite matches parciales entre nombres como:
        ALAIA GUANAJUATO
        14. V1_ALAIA_GUANAJUATO
    """

    if climate_mapping_df is None or climate_mapping_df.empty:
        return None

    centro_key = cc_key(centro_comercial)

    if centro_key in ["", "NAN", "NONE"]:
        return None

    mapping = climate_mapping_df.copy()
    mapping.columns = mapping.columns.astype(str).str.strip()

    # --------------------------------------------------
    # 1) Detectar columnas candidatas de centro comercial
    # --------------------------------------------------

    columnas = list(mapping.columns)

    cc_cols = []

    nombres_cc_preferidos = [
        "centro_comercial",
        "Centro Comercial",
        "CENTRO COMERCIAL",
        "Centro comercial",
        "NOMBRE DEL CC",
        "Nombre del CC",
        "NOMBRE_CC",
        "nombre_cc",
        "CC",
        "cc",
        "PLAZA",
        "Plaza",
        "plaza",
        "mall_folder",
        "Mall",
        "MALL",
        "source_sheet"
    ]

    for col in nombres_cc_preferidos:
        if col in mapping.columns and col not in cc_cols:
            cc_cols.append(col)

    for col in columnas:
        col_key = normalizar_texto_simple(col)

        if (
            "CENTRO" in col_key
            or "COMERCIAL" in col_key
            or col_key in ["CC", "PLAZA", "MALL"]
            or "PLAZA" in col_key
            or "MALL" in col_key
        ):
            if col not in cc_cols:
                cc_cols.append(col)

    # --------------------------------------------------
    # 2) Detectar columnas candidatas de zona NREL
    # --------------------------------------------------

    zona_cols = []

    nombres_zona_preferidos = [
        "zona_nrel",
        "Zona NREL",
        "ZONA_NREL",
        "zona NREL",
        "NREL",
        "nrel",
        "nrel_zone",
        "NREL_ZONE",
        "climate_zone",
        "Climate Zone",
        "CLIMATE_ZONE",
        "zona_climatica",
        "Zona climática",
        "ZONA CLIMATICA",
        "zona_clima",
        "Zona clima",
        "ZONA_CLIMA",
        "zona_ashrae",
        "ASHRAE",
        "ashrae"
    ]

    for col in nombres_zona_preferidos:
        if col in mapping.columns and col not in zona_cols:
            zona_cols.append(col)

    for col in columnas:
        col_key = normalizar_texto_simple(col)

        if (
            "NREL" in col_key
            or "CLIMATE" in col_key
            or "CLIMA" in col_key
            or "CLIMATICA" in col_key
            or "ASHRAE" in col_key
            or col_key in ["ZONA", "ZONE"]
        ):
            if col not in zona_cols:
                zona_cols.append(col)

    if not cc_cols or not zona_cols:
        return None

    # --------------------------------------------------
    # 3) Buscar match exacto o parcial
    # --------------------------------------------------

    for cc_col in cc_cols:
        mapping["_cc_key_temp"] = mapping[cc_col].apply(cc_key)

        # Match exacto
        exact_match = mapping[
            mapping["_cc_key_temp"].eq(centro_key)
        ]

        if not exact_match.empty:
            for zona_col in zona_cols:
                zona = exact_match.iloc[0].get(zona_col)

                if pd.notna(zona) and str(zona).strip() != "":
                    return str(zona).strip()

        # Match parcial en ambos sentidos
        for _, row in mapping.iterrows():
            cc_map_key = row.get("_cc_key_temp", "")

            if pd.isna(cc_map_key):
                continue

            cc_map_key = str(cc_map_key).strip()

            if cc_map_key == "":
                continue

            if cc_map_key in centro_key or centro_key in cc_map_key:
                for zona_col in zona_cols:
                    zona = row.get(zona_col)

                    if pd.notna(zona) and str(zona).strip() != "":
                        return str(zona).strip()

    return None


def resolver_profile_path_nrel(profiles_dir: Path, zona_nrel, perfil_code):
    """
    Resuelve el archivo CSV del perfil NREL.

    Los archivos tienen nombres fijos como:
    - up0-mixed-dry-fullservicerestaurant.csv
    - up0-mixed-dry-quickservicerestaurant.csv
    - up0-hot-dry-retailstandalone.csv
    """

    if profiles_dir is None or not profiles_dir.exists():
        return None

    if zona_nrel is None or pd.isna(zona_nrel):
        return None

    if perfil_code is None or pd.isna(perfil_code):
        return None

    zona = str(zona_nrel).strip().lower()
    perfil = str(perfil_code).strip().lower()

    if zona == "" or perfil == "":
        return None

    zona_slug = (
        zona
        .replace("_", "-")
        .replace(" ", "-")
    )

    perfil_slug = (
        perfil
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )

    expected_name = f"up0-{zona_slug}-{perfil_slug}.csv"

    expected_path = profiles_dir / expected_name

    if expected_path.exists():
        return expected_path

    # Fallback flexible por si hay diferencias menores de mayúsculas,
    # espacios, guiones o underscores.
    def key_file(value):
        return re.sub(
            r"[^A-Z0-9]",
            "",
            normalizar_texto_simple(value)
        )

    expected_key = key_file(expected_name)
    zona_key = key_file(zona_slug)
    perfil_key = key_file(perfil_slug)

    for path in profiles_dir.rglob("*.csv"):
        path_key = key_file(path.name)

        if path_key == expected_key:
            return path

        if zona_key in path_key and perfil_key in path_key:
            return path

    return None


def estimar_kwmax_pdbt_desde_nrel(
    kwh_total,
    periodo_inicio,
    periodo_fin,
    profile_path
):
    """
    Estima la demanda máxima mensual para PDBT.

    Metodología:
    1. El recibo PDBT trae kWh mensuales, pero no trae demanda medida.
    2. El perfil NREL se transforma en un perfil horario típico mensual.
    3. Para cada mes, se calcula el peso de la hora pico del día típico.
    4. kWh mensual / días del periodo = kWh por día.
    5. kWh por día * peso horario máximo = kWh en la hora pico.
    6. Como la ventana es horaria, kWh/h equivale aproximadamente a kW.
    """

    if pd.isna(kwh_total) or kwh_total <= 0:
        return pd.NA

    if profile_path is None:
        return pd.NA

    if pd.isna(periodo_fin):
        return pd.NA

    periodo_fin = pd.to_datetime(periodo_fin, errors="coerce")
    periodo_inicio = pd.to_datetime(periodo_inicio, errors="coerce")

    if pd.isna(periodo_fin):
        return pd.NA

    if pd.isna(periodo_inicio):
        dias_periodo = periodo_fin.days_in_month
    else:
        dias_periodo = (periodo_fin - periodo_inicio).days

        if dias_periodo <= 0:
            dias_periodo = periodo_fin.days_in_month

    mes = int(periodo_fin.month)

    try:
        perfil_mensual = construir_perfil_mensual_nrel(str(profile_path))
    except Exception:
        return pd.NA

    perfil_mes = perfil_mensual[
        perfil_mensual["mes"] == mes
    ].copy()

    if perfil_mes.empty:
        return pd.NA

    peso_max_horario_mes = perfil_mes["peso_max_horario_mes"].max()

    if pd.isna(peso_max_horario_mes) or peso_max_horario_mes <= 0:
        return pd.NA

    kwh_dia_promedio = kwh_total / dias_periodo

    kwmax_estimado = kwh_dia_promedio * peso_max_horario_mes

    return kwmax_estimado


@st.cache_data(show_spinner=False)
def aplicar_estimacion_pdbt_nrel_a_parsed(
    parsed: pd.DataFrame,
    muestra_con_recibo: pd.DataFrame,
    data_dir: Path
) -> pd.DataFrame:
    """
    Estima demanda máxima mensual para recibos PDBT usando perfiles NREL.

    Esta función modifica parsed agregando:
    - demanda_maxima_mensual_kw para PDBT
    - criterio_demanda_mensual = estimado PDBT con NREL
    - zona_nrel
    - perfil_nrel_code
    - perfil_nrel_nombre
    - profile_path_nrel
    """

    if parsed.empty or muestra_con_recibo.empty:
        return parsed

    p = parsed.copy()
    m = muestra_con_recibo.copy()

    cc_master_path = data_dir / "profiles" / "cc_master_data.csv"
    profiles_dir = data_dir / "profiles"

    if not cc_master_path.exists():
        p["pdbt_nrel_status"] = "No existe cc_master_data.csv"
        return p

    climate_mapping_df = pd.read_csv(cc_master_path)
    climate_mapping_df.columns = climate_mapping_df.columns.str.strip()

    # --------------------------------------------------
    # Llaves en parser
    # --------------------------------------------------

    if "mall_folder" in p.columns:
        p["_cc_key"] = p["mall_folder"].apply(cc_key)
    else:
        p["_cc_key"] = ""

    p["_key_medidor"] = (
        normalize_meter_cc(p["medidor"])
        if "medidor" in p.columns
        else ""
    )

    p["_key_no_servicio"] = (
        normalize_service_cc(p["no_servicio"])
        if "no_servicio" in p.columns
        else ""
    )

    p["_key_cliente"] = (
        normalize_cc_key(p["cliente_nombre"])
        if "cliente_nombre" in p.columns
        else ""
    )

    p["_key_nombre"] = (
        normalize_cc_key(p["recibos_subgroup"])
        if "recibos_subgroup" in p.columns
        else ""
    )

    # --------------------------------------------------
    # Llaves en muestra validada
    # --------------------------------------------------

    if "_centro_comercial_limpio" in m.columns:
        m["_cc_key"] = m["_centro_comercial_limpio"].apply(cc_key)
    elif "NOMBRE DEL CC" in m.columns:
        m["_cc_key"] = m["NOMBRE DEL CC"].apply(cc_key)
    elif "source_sheet" in m.columns:
        m["_cc_key"] = m["source_sheet"].apply(cc_key)
    else:
        m["_cc_key"] = ""

    general_meter_col = first_existing_column(
        m,
        [
            "No. De medidor",
            "No. de medidor",
            "No de medidor",
            "MEDIDOR",
            "Medidor"
        ]
    )

    if "_key_medidor" not in m.columns:
        m["_key_medidor"] = (
            normalize_meter_cc(m[general_meter_col])
            if general_meter_col
            else ""
        )

    if "_key_cliente" not in m.columns:
        m["_key_cliente"] = (
            normalize_cc_key(m["CLIENTE"])
            if "CLIENTE" in m.columns
            else ""
        )

    if "_key_nombre_comercial" not in m.columns:
        m["_key_nombre_comercial"] = (
            normalize_cc_key(m["NOMBRE COMERCIAL"])
            if "NOMBRE COMERCIAL" in m.columns
            else ""
        )

    giro_col = first_existing_column(
        m,
        [
            "SUBGIRO_COMERCIAL",
            "SUBGIRO COMERCIAL",
            "GIRO_COMERCIAL",
            "GIRO COMERCIAL",
            "GIRO",
            "Giro"
        ]
    )

    tipo_local_col = first_existing_column(
        m,
        [
            "TIPO LOCAL",
            "Tipo Local",
            "TIPO_LOCAL",
            "tipo_local"
        ]
    )

    centro_col = first_existing_column(
        m,
        [
            "_centro_comercial_limpio",
            "NOMBRE DEL CC",
            "CENTRO COMERCIAL",
            "CC",
            "PLAZA",
            "source_sheet"
        ]
    )

    # --------------------------------------------------
    # Columnas para fallback por No. de local / dirección
    # --------------------------------------------------

    local_col_muestra = first_existing_column(
        m,
        [
            "No de Local",
            "No de local",
            "No. de Local",
            "No. de local",
            "No Local",
            "No. Local",
            "LOCAL",
            "Local"
        ]
    )

    parser_text_cols = [
        col for col in [
            "direccion_completa",
            "direccion_raw",
            "file_path",
            "file_name",
            "recibos_subgroup"
        ]
        if col in p.columns
    ]

    def generar_variantes_no_local_pdbt(no_local):
        raw = str(no_local).upper().strip()

        if raw in ["", "NAN", "NONE"]:
            return []

        raw = (
            raw
            .replace("LOCAL", "")
            .replace("LOC.", "")
            .replace("LOC", "")
            .replace("NO.", "")
            .replace("NO", "")
            .replace("#", "")
            .strip()
        )

        raw = re.sub(r"\s+", " ", raw)

        variantes = set()

        prefijo_match = re.match(r"^([A-Z0-9]+)[\-\s]*", raw)
        prefijo_base = prefijo_match.group(1) if prefijo_match else ""

        partes = re.split(r",|/|\bY\b|\bE\b|;", raw)

        for parte in partes:
            parte = parte.strip()

            if not parte:
                continue

            compact = re.sub(r"[^A-Z0-9]", "", parte)

            if len(compact) >= 3:
                variantes.add(compact)

            if prefijo_base and re.fullmatch(r"[0-9]+[A-Z]?", compact):
                variantes.add(f"{prefijo_base}{compact}")

            m_local = re.match(r"^([A-Z0-9]+)[\-\s]*([0-9]+[A-Z]?)$", parte)

            if m_local:
                variantes.add(f"{m_local.group(1)}{m_local.group(2)}")

        full_compact = re.sub(r"[^A-Z0-9]", "", raw)

        if len(full_compact) >= 3:
            variantes.add(full_compact)

        variantes = {
            v for v in variantes
            if len(v) >= 3
            and v not in ["LOCAL", "NAN", "NONE"]
        }

        return list(variantes)

    def normalizar_texto_para_local_pdbt(value):
        return re.sub(
            r"[^A-Z0-9]",
            "",
            normalizar_texto_simple(value)
        )

    # Columnas nuevas en parser
    for col in [
        "zona_nrel",
        "perfil_nrel_code",
        "perfil_nrel_nombre",
        "profile_path_nrel",
        "pdbt_nrel_status",
        "centro_comercial_nrel_input",
        "subgiro_nrel_input",
        "tipo_local_nrel_input"
    ]:
        if col not in p.columns:
            p[col] = pd.NA


    def split_match_keys(value, tipo="servicio"):
        """
        Convierte valores guardados por el match global como:
        '671221152141 | 673201100237'
        o
        '381AB0 | 644RRT'
        en una lista de llaves normalizadas.
        """

        if pd.isna(value):
            return []

        raw_values = str(value).split("|")

        keys = []

        for raw in raw_values:
            raw = raw.strip()

            if raw.upper() in ["", "NAN", "NONE", "<NA>"]:
                continue

            if tipo == "servicio":
                key = normalize_service_cc(pd.Series([raw])).iloc[0]

            elif tipo == "medidor":
                key = normalize_meter_cc(pd.Series([raw])).iloc[0]

            else:
                key = str(raw).strip().upper()

            if key and key.upper() not in ["", "NAN", "NONE", "<NA>"]:
                keys.append(key)

        return sorted(set(keys))

    # --------------------------------------------------
    # Asignar perfil NREL a registros PDBT del parser
    # usando la muestra validada.
    # --------------------------------------------------

    muestra_iter = m.copy()

    for _, row in muestra_iter.iterrows():

        subgiro = row.get(giro_col, "") if giro_col else ""
        tipo_local = row.get(tipo_local_col, "") if tipo_local_col else ""
        # --------------------------------------------------
        # Centro comercial usado para buscar zona NREL
        # --------------------------------------------------
        # Probamos varias columnas porque a veces el nombre limpio,
        # source_sheet y NOMBRE DEL CC no son idénticos al cc_master_data.csv.

        centros_posibles = []

        if centro_col:
            centros_posibles.append(row.get(centro_col, ""))

        for col_cc_alt in [
            "_centro_comercial_limpio",
            "NOMBRE DEL CC",
            "CENTRO COMERCIAL",
            "CC",
            "PLAZA",
            "source_sheet"
        ]:
            if col_cc_alt in m.columns:
                centros_posibles.append(row.get(col_cc_alt, ""))

        centros_posibles = [
            c for c in centros_posibles
            if pd.notna(c) and str(c).strip() != ""
        ]

        centro_comercial = centros_posibles[0] if centros_posibles else ""

        zona_nrel = None

        for centro_candidato in centros_posibles:
            zona_nrel = obtener_zona_nrel_por_cc_pdbt(
                centro_candidato,
                climate_mapping_df
            )

            if zona_nrel is not None and str(zona_nrel).strip() != "":
                centro_comercial = centro_candidato
                break

        perfil_code, perfil_nombre = obtener_tipo_perfil_nrel_pdbt(
            subgiro,
            tipo_local
        )

        profile_path = resolver_profile_path_nrel(
            profiles_dir=profiles_dir,
            zona_nrel=zona_nrel,
            perfil_code=perfil_code
        )

        mask_base = (
            p["tarifa_norm"].astype(str).str.upper().eq("PDBT")
            & p["_cc_key"].eq(row.get("_cc_key", ""))
        )

        # --------------------------------------------------
        # 0) Match preferente usando datos del match global
        # --------------------------------------------------
        # Si el match global ya guardó no_servicio o medidor del parser,
        # usamos eso primero. Esto es clave para casos como:
        # '671221152141 | 673201100237'
        # y para matches flexibles por cliente/nombre.

        servicios_match_global = split_match_keys(
            row.get("parser_no_servicio_match", ""),
            tipo="servicio"
        )

        medidores_match_global = split_match_keys(
            row.get("parser_medidor_match", ""),
            tipo="medidor"
        )

        if servicios_match_global:
            mask_match = (
                mask_base
                & p["_key_no_servicio"].astype(str).isin(servicios_match_global)
            )

        elif medidores_match_global:
            mask_match = (
                mask_base
                & p["_key_medidor"].astype(str).isin(medidores_match_global)
            )

        else:
            mask_match = pd.Series(False, index=p.index)

        # --------------------------------------------------
        # 1) Fallback por medidor original de DG
        # --------------------------------------------------

        if not mask_match.any():
            mask_match = (
                mask_base
                & p["_key_medidor"].astype(str).ne("")
                & p["_key_medidor"].eq(str(row.get("_key_medidor", "")))
            )

        # 2) Fallback por cliente
        if not mask_match.any():
            mask_match = (
                mask_base
                & p["_key_cliente"].astype(str).ne("")
                & p["_key_cliente"].eq(str(row.get("_key_cliente", "")))
            )

        # 3) Fallback por nombre comercial
        if not mask_match.any():
            mask_match = (
                mask_base
                & p["_key_nombre"].astype(str).ne("")
                & p["_key_nombre"].eq(str(row.get("_key_nombre_comercial", "")))
            )

        # 4) Fallback por No. de local / dirección
        if not mask_match.any() and local_col_muestra and parser_text_cols:
            variantes_local = generar_variantes_no_local_pdbt(
                row.get(local_col_muestra, "")
            )

            if variantes_local:
                candidatos_idx = p[mask_base].index

                if len(candidatos_idx) > 0:
                    textos_parser = pd.Series(
                        "",
                        index=candidatos_idx,
                        dtype="object"
                    )

                    for col_texto in parser_text_cols:
                        textos_parser = (
                            textos_parser.astype(str)
                            + " "
                            + p.loc[
                                candidatos_idx,
                                col_texto
                            ].fillna("").astype(str)
                        )

                    textos_parser_key = textos_parser.apply(
                        normalizar_texto_para_local_pdbt
                    )

                    mask_local_idx = textos_parser_key[
                        textos_parser_key.apply(
                            lambda texto: any(
                                variante in texto
                                for variante in variantes_local
                            )
                        )
                    ].index

                    if len(mask_local_idx) > 0:
                        mask_match = p.index.isin(mask_local_idx)

        if mask_match.any():
            p.loc[mask_match, "centro_comercial_nrel_input"] = centro_comercial
            p.loc[mask_match, "subgiro_nrel_input"] = subgiro
            p.loc[mask_match, "tipo_local_nrel_input"] = tipo_local

            p.loc[mask_match, "zona_nrel"] = zona_nrel
            p.loc[mask_match, "perfil_nrel_code"] = perfil_code
            p.loc[mask_match, "perfil_nrel_nombre"] = perfil_nombre
            p.loc[mask_match, "profile_path_nrel"] = (
                str(profile_path) if profile_path is not None else pd.NA
            )

            if zona_nrel is None:
                p.loc[mask_match, "pdbt_nrel_status"] = "Sin zona NREL para centro comercial"

            elif profile_path is None:
                p.loc[mask_match, "pdbt_nrel_status"] = (
                    "Sin archivo de perfil NREL: "
                    + str(zona_nrel)
                    + " / "
                    + str(perfil_code)
                )

            else:
                p.loc[mask_match, "pdbt_nrel_status"] = "Perfil NREL asignado"


    # --------------------------------------------------
    # Estimar demanda mensual PDBT
    # --------------------------------------------------
    # Solo intentamos estimar registros que tienen:
    # - perfil NREL asignado
    # - kWh del recibo
    # - periodo inicio
    # - periodo fin
    #
    # Los registros incompletos se saltan rápido para evitar
    # que la app se congele procesando filas que no pueden calcularse.

    mask_pdbt = p["tarifa_norm"].astype(str).str.upper().eq("PDBT")

    for idx in p[mask_pdbt].index:

        profile_path_value = p.at[idx, "profile_path_nrel"]

        if pd.isna(profile_path_value) or str(profile_path_value).strip() == "":
            if (
                pd.isna(p.at[idx, "pdbt_nrel_status"])
                or str(p.at[idx, "pdbt_nrel_status"]).strip() == ""
            ):
                p.at[idx, "pdbt_nrel_status"] = "No estimado: sin perfil NREL asignado"
            continue

        kwh_value = (
            p.at[idx, "kwh_total_num"]
            if "kwh_total_num" in p.columns
            else pd.NA
        )

        periodo_inicio_value = (
            p.at[idx, "periodo_inicio_dt"]
            if "periodo_inicio_dt" in p.columns
            else pd.NaT
        )

        periodo_fin_value = (
            p.at[idx, "periodo_fin_dt"]
            if "periodo_fin_dt" in p.columns
            else pd.NaT
        )

        faltantes_estimacion = []

        if pd.isna(kwh_value):
            faltantes_estimacion.append("kwh_total_num")

        if pd.isna(periodo_inicio_value):
            faltantes_estimacion.append("periodo_inicio_dt")

        if pd.isna(periodo_fin_value):
            faltantes_estimacion.append("periodo_fin_dt")

        if faltantes_estimacion:
            p.at[idx, "pdbt_nrel_status"] = (
                "No estimado: falta "
                + ", ".join(faltantes_estimacion)
            )
            continue

        kw_estimado = estimar_kwmax_pdbt_desde_nrel(
            kwh_total=kwh_value,
            periodo_inicio=periodo_inicio_value,
            periodo_fin=periodo_fin_value,
            profile_path=Path(profile_path_value)
        )

        if pd.notna(kw_estimado):
            p.at[idx, "demanda_maxima_mensual_kw"] = kw_estimado
            p.at[idx, "criterio_demanda_mensual"] = "estimado PDBT con perfil NREL"
            p.at[idx, "pdbt_nrel_status"] = "Demanda estimada con NREL"

        else:
            p.at[idx, "pdbt_nrel_status"] = (
                "No estimado: no se pudo calcular kW con perfil NREL"
            )
    return p

@st.cache_data(show_spinner=False)
def calcular_demanda_promedio_ultimos_12_meses(parsed: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la Demanda promedio anual (kW) por servicio/local.

    Definición estándar de la app:
    Demanda promedio anual (kW) =
    promedio de las demandas máximas de los recibos más recientes disponibles.

    Regla:
    - Si el recibo más reciente del servicio es bimestral, se usan hasta 6 recibos.
    - Si el recibo más reciente del servicio es mensual, se usan hasta 12 recibos.
    - Si hay menos recibos disponibles, se promedia con los recibos existentes.

    La columna base para el promedio es demanda_maxima_mensual_kw:
    - Para GDMTH/GDMTO/GDBT viene del kwmax del parser.
    - Para PDBT viene de la estimación con perfiles NREL.
    """

    if parsed.empty:
        return parsed

    p = parsed.copy()

    # --------------------------------------------------
    # Asegurar fechas y numéricos
    # --------------------------------------------------

    if "periodo_inicio_dt" not in p.columns and "periodo_inicio" in p.columns:
        p["periodo_inicio_dt"] = pd.to_datetime(
            p["periodo_inicio"],
            errors="coerce",
            dayfirst=True
        )

    if "periodo_fin_dt" not in p.columns and "periodo_fin" in p.columns:
        p["periodo_fin_dt"] = pd.to_datetime(
            p["periodo_fin"],
            errors="coerce",
            dayfirst=True
        )

    if "_fecha_orden_demanda" not in p.columns:
        if "periodo_fin_dt" in p.columns:
            p["_fecha_orden_demanda"] = p["periodo_fin_dt"]
        elif "periodo_inicio_dt" in p.columns:
            p["_fecha_orden_demanda"] = p["periodo_inicio_dt"]
        else:
            p["_fecha_orden_demanda"] = pd.NaT

    if "demanda_maxima_mensual_kw" in p.columns:
        p["demanda_maxima_mensual_kw"] = pd.to_numeric(
            p["demanda_maxima_mensual_kw"],
            errors="coerce"
        )
    else:
        p["demanda_maxima_mensual_kw"] = pd.NA

    if "kwh_total_num" not in p.columns and "kwh_total" in p.columns:
        p["kwh_total_num"] = clean_number_series(p["kwh_total"])

    # --------------------------------------------------
    # Unidad de cálculo anual
    # --------------------------------------------------
    # La demanda anual debe agruparse por no_servicio, no por medidor.
    # Un mismo no_servicio puede tener varios medidores durante el año.

    group_cols = [
        col for col in [
            "mall_folder",
            "no_servicio",
            "tarifa_norm"
        ]
        if col in p.columns
    ]

    if not group_cols:
        return p

    base = p.dropna(
        subset=[
            "demanda_maxima_mensual_kw",
            "_fecha_orden_demanda"
        ]
    ).copy()

    if base.empty:
        p["demanda_promedio_anual_kw"] = pd.NA
        p["demanda_maxima_anual_kw"] = pd.NA
        p["meses_con_demanda"] = pd.NA
        p["periodo_12m_inicio"] = pd.NaT
        p["periodo_12m_fin"] = pd.NaT
        p["kwh_12m"] = pd.NA
        p["periodicidad_recibo_demanda"] = pd.NA
        p["recibos_usados_demanda"] = pd.NA
        p["recibos_esperados_demanda"] = pd.NA
        p["cobertura_recibos_demanda_pct"] = pd.NA
        return p

    # --------------------------------------------------
    # Función auxiliar: duración del recibo
    # --------------------------------------------------

    def estimar_meses_recibo(inicio, fin):
        """
        Estima si un recibo/fila es mensual o bimestral usando
        periodo_inicio y periodo_fin.

        Regla:
        - duración <= 45 días: mensual
        - duración > 45 días: bimestral

        Si no hay fechas suficientes, asumimos mensual.
        """

        inicio = pd.to_datetime(inicio, errors="coerce")
        fin = pd.to_datetime(fin, errors="coerce")

        if pd.isna(inicio) or pd.isna(fin):
            return 1

        if fin < inicio:
            inicio, fin = fin, inicio

        dias = (fin - inicio).days

        if dias > 45:
            return 2

        return 1

    # --------------------------------------------------
    # Calcular demanda promedio anual por grupo
    # --------------------------------------------------

    rows = []

    for group_key, g in base.groupby(group_cols, dropna=False):

        g_ordenado = g.copy()

        g_ordenado["_periodo_sort"] = pd.to_datetime(
            g_ordenado["_fecha_orden_demanda"],
            errors="coerce"
        )

        g_ordenado = g_ordenado[
            g_ordenado["_periodo_sort"].notna()
            & g_ordenado["demanda_maxima_mensual_kw"].notna()
        ].copy()

        if g_ordenado.empty:
            continue

        g_ordenado = g_ordenado.sort_values(
            "_periodo_sort",
            ascending=False
        )

        recibo_mas_reciente = g_ordenado.iloc[0]

        recibo_inicio_reciente = (
            recibo_mas_reciente.get("periodo_inicio_dt", pd.NaT)
            if "periodo_inicio_dt" in g_ordenado.columns
            else recibo_mas_reciente.get("_fecha_orden_demanda", pd.NaT)
        )

        recibo_fin_reciente = (
            recibo_mas_reciente.get("periodo_fin_dt", pd.NaT)
            if "periodo_fin_dt" in g_ordenado.columns
            else recibo_mas_reciente.get("_fecha_orden_demanda", pd.NaT)
        )

        meses_recibo_reciente = estimar_meses_recibo(
            recibo_inicio_reciente,
            recibo_fin_reciente
        )

        if meses_recibo_reciente >= 2:
            n_recibos_ventana = 6
            periodicidad = "Bimestral"
        else:
            n_recibos_ventana = 12
            periodicidad = "Mensual"

        # Si hay menos recibos que la ventana ideal, head() toma solo los disponibles.
        g_ventana = g_ordenado.head(n_recibos_ventana).copy()

        if g_ventana.empty:
            continue

        periodo_inicio_real = (
            g_ventana["periodo_inicio_dt"].min()
            if "periodo_inicio_dt" in g_ventana.columns
            else g_ventana["_periodo_sort"].min()
        )

        periodo_fin_real = (
            g_ventana["periodo_fin_dt"].max()
            if "periodo_fin_dt" in g_ventana.columns
            else g_ventana["_periodo_sort"].max()
        )

        row = {}

        if isinstance(group_key, tuple):
            for col, value in zip(group_cols, group_key):
                row[col] = value
        else:
            row[group_cols[0]] = group_key

        demanda_ventana = pd.to_numeric(
            g_ventana["demanda_maxima_mensual_kw"],
            errors="coerce"
        ).dropna()

        if demanda_ventana.empty:
            continue

        row["demanda_promedio_anual_kw"] = demanda_ventana.mean()
        row["demanda_maxima_anual_kw"] = demanda_ventana.max()

        row["recibos_usados_demanda"] = len(demanda_ventana)
        row["recibos_esperados_demanda"] = n_recibos_ventana
        row["cobertura_recibos_demanda_pct"] = (
            len(demanda_ventana)
            / n_recibos_ventana
            * 100
            if n_recibos_ventana > 0
            else pd.NA
        )

        # Meses estimados cubiertos por los recibos usados.
        # Mensual: 1 mes por recibo.
        # Bimestral: 2 meses por recibo.
        row["meses_con_demanda"] = int(
            len(demanda_ventana) * meses_recibo_reciente
        )

        row["periodo_12m_inicio"] = periodo_inicio_real
        row["periodo_12m_fin"] = periodo_fin_real
        row["periodicidad_recibo_demanda"] = periodicidad

        if "kwh_total_num" in g_ventana.columns:
            row["kwh_12m"] = pd.to_numeric(
                g_ventana["kwh_total_num"],
                errors="coerce"
            ).sum()
        else:
            row["kwh_12m"] = pd.NA

        rows.append(row)

    demanda_promedio_12m_df = pd.DataFrame(rows)

    if demanda_promedio_12m_df.empty:
        p["demanda_promedio_anual_kw"] = pd.NA
        p["demanda_maxima_anual_kw"] = pd.NA
        p["meses_con_demanda"] = pd.NA
        p["periodo_12m_inicio"] = pd.NaT
        p["periodo_12m_fin"] = pd.NaT
        p["kwh_12m"] = pd.NA
        p["periodicidad_recibo_demanda"] = pd.NA
        p["recibos_usados_demanda"] = pd.NA
        p["recibos_esperados_demanda"] = pd.NA
        p["cobertura_recibos_demanda_pct"] = pd.NA
        return p

    cols_to_drop = [
        col for col in [
            "demanda_promedio_anual_kw",
            "demanda_maxima_anual_kw",
            "meses_con_demanda",
            "periodo_12m_inicio",
            "periodo_12m_fin",
            "kwh_12m",
            "periodicidad_recibo_demanda",
            "recibos_usados_demanda",
            "recibos_esperados_demanda",
            "cobertura_recibos_demanda_pct"
        ]
        if col in p.columns
    ]

    p = p.drop(columns=cols_to_drop, errors="ignore")

    p = p.merge(
        demanda_promedio_12m_df,
        on=group_cols,
        how="left"
    )

    return p

# ============================================================
# Header
# ============================================================

st.markdown('<div class="main-title">Benchmark de demanda eléctrica de centros comerciales de Allux </div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Benchmark de demanda eléctrica de centros comerciales de Allux</div>',
    unsafe_allow_html=True
)


# ============================================================
# Sidebar: data paths
# ============================================================

with st.sidebar:
    st.header("Datos")

    parsed_path_text = st.text_input(
        "CSV de recibos parseados",
        value=str(DEFAULT_PARSED_CSV)
    )

    historico_path_text = st.text_input(
        "CSV de histórico",
        value=str(DEFAULT_HISTORICO_CSV)
    )

    general_data_path_text = st.text_input(
        "Excel de datos generales",
        value=str(DEFAULT_GENERAL_DATA)
    )

    st.caption(
        "Los dos primeros archivos son generados por `app_scanner.py` / `scanner.py`. "
        "El tercero contiene datos generales como m², giro, local y medidor."
    )


# ============================================================
# Load data
# ============================================================

parsed_path = Path(parsed_path_text).expanduser()
historico_path = Path(historico_path_text).expanduser()
general_data_path = Path(general_data_path_text).expanduser()

parsed_sig = file_signature(parsed_path)
historico_sig = file_signature(historico_path)
general_sig = file_signature(general_data_path)

parsed_raw = read_csv_cached(*parsed_sig)
historico_raw = read_csv_cached(*historico_sig)
general_raw = read_excel_all_sheets_cached(*general_sig)

parsed = prepare_parsed_data(parsed_raw)

# ------------------------------------------------------------
# Overrides manuales de parser antes del match global
# ------------------------------------------------------------
# Deben aplicarse antes de crear tarifa_norm y antes de
# construir_muestra_con_recibo_global(...).

parsed = aplicar_override_iberdrola_liverpool(parsed)

# ------------------------------------------------------------
# Enriquecimiento del parser con archivos externos de rescate
# ------------------------------------------------------------
# Este paso debe ocurrir antes de crear tarifa_norm, kwmax_num,
# kwh_total_num, demanda_maxima_mensual_kw y antes del match global.

parsed = enriquecer_parser_con_archivos_rescate(
    parsed=parsed,
    pdbt_kwh_path=PDBT_KWH_RESCUE_CSV,
    gdm_rescue_path=GDM_RESCUE_CSV,
    new_rows_path=NEW_PARSER_ROWS_CSV
)

# ============================================================
# Demanda real estándar por tarifa
# ============================================================

parsed["tarifa_norm"] = (
    normalize_tarifa_series(parsed["tarifa"])
    if "tarifa" in parsed.columns
    else pd.NA
)

parsed["kwmax_num"] = clean_number_series(
    parsed["kwmax"]
) if "kwmax" in parsed.columns else pd.NA

parsed["kwh_total_num"] = clean_number_series(
    parsed["kwh_total"]
) if "kwh_total" in parsed.columns else pd.NA

parsed["demanda_real_kw"] = pd.NA
parsed["criterio_demanda_real"] = pd.NA

mask_demanda_medida = parsed["tarifa_norm"].isin(
    ["GDMTH", "GDMTO", "GDBT"]
)

parsed.loc[
    mask_demanda_medida,
    "demanda_real_kw"
] = parsed.loc[
    mask_demanda_medida,
    "kwmax_num"
]

parsed.loc[
    mask_demanda_medida,
    "criterio_demanda_real"
] = "kwmax medido en recibo"

mask_pdbt = parsed["tarifa_norm"].eq("PDBT")

parsed.loc[
    mask_pdbt,
    "criterio_demanda_real"
] = "pendiente estimación NREL"


historico = prepare_historico_data(historico_raw)
general_data = prepare_general_data(general_raw)


if parsed.empty:
    st.error(
        "No encontré datos en `bills_parsed_v2.csv`. "
        "Primero ejecuta el scanner o revisa la ruta del CSV."
    )
    st.stop()


# ============================================================
# Sidebar filters and merge settings
# ============================================================

with st.sidebar:
    st.header("Filtros")

    mall_col = first_existing_column(parsed, ["mall_folder", "mall", "centro_comercial"])
    tenant_col = first_existing_column(parsed, ["recibos_subgroup", "cliente_nombre", "tenant", "locatario"])
    service_col = first_existing_column(parsed, ["no_servicio", "servicio"])
    tariff_col = first_existing_column(parsed, ["tarifa"])

    filtered = parsed.copy()

    if mall_col:
        malls = sorted([x for x in filtered[mall_col].dropna().unique()])

        use_all_malls = st.checkbox(
            "Incluir todos los centros comerciales",
            value=True
        )

        if use_all_malls:
            selected_malls = malls
            st.caption(f"Centros comerciales incluidos: {len(selected_malls)}")
        else:
            selected_malls = st.multiselect(
                "Centro comercial",
                options=malls,
                default=[]
            )

        if selected_malls:
            filtered = filtered[filtered[mall_col].isin(selected_malls)]
        else:
            filtered = filtered.iloc[0:0]

    if tenant_col:
        tenants = sorted([x for x in filtered[tenant_col].dropna().unique()])
        selected_tenants = st.multiselect(
            "Locatario / subgrupo",
            options=tenants,
            default=[]
        )
        if selected_tenants:
            filtered = filtered[filtered[tenant_col].isin(selected_tenants)]

    if tariff_col:
        tariffs = sorted([x for x in filtered[tariff_col].dropna().unique()])
        selected_tariffs = st.multiselect(
            "Tarifa",
            options=tariffs,
            default=[]
        )
        if selected_tariffs:
            filtered = filtered[filtered[tariff_col].isin(selected_tariffs)]

    if "kwh_total_num" in filtered.columns:
        kwh_nonnull = filtered["kwh_total_num"].dropna()

        if not kwh_nonnull.empty:
            st.markdown("#### Rango de kWh")

            use_compact_kwh_slider = st.checkbox(
                "Usar escala compacta para kWh",
                value=True,
                help=(
                    "Usa el percentil 99 como máximo visual para evitar que valores "
                    "extremos hagan poco útil el control deslizante."
                )
            )

            min_kwh = float(kwh_nonnull.min())
            absolute_max_kwh = float(kwh_nonnull.max())

            if use_compact_kwh_slider and len(kwh_nonnull) > 10:
                visual_max_kwh = float(kwh_nonnull.quantile(0.99))
                visual_max_kwh = max(visual_max_kwh, min_kwh)
            else:
                visual_max_kwh = absolute_max_kwh

            selected_kwh_range = st.slider(
                "Rango de kWh por recibo",
                min_value=float(min_kwh),
                max_value=float(visual_max_kwh),
                value=(float(min_kwh), float(visual_max_kwh))
            )

            filtered = filtered[
                (filtered["kwh_total_num"].isna()) |
                (
                    (filtered["kwh_total_num"] >= selected_kwh_range[0]) &
                    (filtered["kwh_total_num"] <= selected_kwh_range[1])
                )
            ]

            if use_compact_kwh_slider and visual_max_kwh < absolute_max_kwh:
                st.caption(
                    f"Máximo real en los datos: {absolute_max_kwh:,.0f} kWh. "
                    f"Máximo visual usado: {visual_max_kwh:,.0f} kWh."
                )

    st.header("Cruce con datos generales")

    use_general_merge = False
    receipt_key_col = None
    general_key_col = None
    area_col = None

    if general_data.empty:
        st.warning("No se cargó el archivo de datos generales.")
    else:
        use_general_merge = st.checkbox(
            "Activar cruce con datos generales",
            value=True
        )

        if use_general_merge:
            receipt_options = list(filtered.columns)
            general_options = list(general_data.columns)

            receipt_key_candidates = [
                "recibos_subgroup",
                "cliente_nombre",
                "medidor",
                "no_servicio",
                "cuenta",
            ]

            general_key_candidates = [
                "NOMBRE COMERCIAL",
                "CLIENTE",
                "No. De medidor",
                "No. de medidor",
                "TARIFA",
            ]

            area_candidates = [
                "MTS2",
                "M2",
                "m2",
                "MTS 2",
                "MTS²",
                "AREA_M2",
                "AREA M2",
                "SUPERFICIE",
                "SUPERFICIE M2",
            ]

            receipt_key_col = st.selectbox(
                "Llave en recibos",
                options=receipt_options,
                index=preferred_col_index(receipt_options, receipt_key_candidates)
            )

            general_key_col = st.selectbox(
                "Llave en datos generales",
                options=general_options,
                index=preferred_col_index(general_options, general_key_candidates)
            )

            area_col = st.selectbox(
                "Columna de superficie",
                options=general_options,
                index=preferred_col_index(general_options, area_candidates)
            )
# ============================================================
# Enriched data: recibos + datos generales
# ============================================================

if not general_data.empty and "cliente_nombre" in filtered.columns and "CLIENTE" in general_data.columns:
    filtered_enriched = filtered.copy()
    general_data_enriched = general_data.copy()

    filtered_enriched["_cliente_key"] = normalize_text_key(filtered_enriched["cliente_nombre"])
    general_data_enriched["_cliente_key"] = normalize_text_key(general_data_enriched["CLIENTE"])

    filtered_enriched = filtered_enriched.merge(
        general_data_enriched,
        on="_cliente_key",
        how="left",
        suffixes=("", "_general")
    )
else:
    filtered_enriched = filtered.copy()

# ============================================================
# Demanda máxima mensual por recibo
# ============================================================

parsed["tarifa_norm"] = (
    normalize_tarifa_series(parsed["tarifa"])
    if "tarifa" in parsed.columns
    else pd.NA
) if "tarifa" in parsed.columns else pd.NA

parsed["kwmax_num"] = (
    clean_number_series(parsed["kwmax"])
    if "kwmax" in parsed.columns
    else pd.NA
)

parsed["kwh_total_num"] = (
    clean_number_series(parsed["kwh_total"])
    if "kwh_total" in parsed.columns
    else pd.NA
)

parsed["periodo_inicio_dt"] = (
    clean_date_series(parsed["periodo_inicio"])
    if "periodo_inicio" in parsed.columns
    else pd.NaT
)

parsed["periodo_fin_dt"] = (
    clean_date_series(parsed["periodo_fin"])
    if "periodo_fin" in parsed.columns
    else pd.NaT
)

# Fecha de orden para seleccionar los últimos 12 meses por servicio/local.
parsed["_fecha_orden_demanda"] = parsed["periodo_fin_dt"]

parsed.loc[
    parsed["_fecha_orden_demanda"].isna(),
    "_fecha_orden_demanda"
] = parsed.loc[
    parsed["_fecha_orden_demanda"].isna(),
    "periodo_inicio_dt"
]

parsed["demanda_maxima_mensual_kw"] = pd.NA
parsed["criterio_demanda_mensual"] = pd.NA

# Tarifas con demanda medida en recibo.
mask_demanda_medida = parsed["tarifa_norm"].isin([
    "GDMTH",
    "GDMTO",
    "GDBT"
])

parsed.loc[
    mask_demanda_medida,
    "demanda_maxima_mensual_kw"
] = parsed.loc[
    mask_demanda_medida,
    "kwmax_num"
]

parsed.loc[
    mask_demanda_medida,
    "criterio_demanda_mensual"
] = "kwmax medido en recibo"

# ------------------------------------------------------------
# Override demanda medida para recibos Iberdrola / Liverpool
# ------------------------------------------------------------
# Iberdrola no necesariamente entra como GDMTH/GDMTO/GDBT en tarifa_norm,
# pero si el parser trae kwmax, ese kwmax debe tratarse como demanda medida.

if "source_utility" in parsed.columns:

    mask_iberdrola_kwmax = (
        parsed["source_utility"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
        .eq("IBERDROLA")
        & parsed["kwmax_num"].notna()
        & parsed["kwmax_num"].gt(0)
    )

    parsed.loc[
        mask_iberdrola_kwmax,
        "demanda_maxima_mensual_kw"
    ] = parsed.loc[
        mask_iberdrola_kwmax,
        "kwmax_num"
    ]

    parsed.loc[
        mask_iberdrola_kwmax,
        "demanda_real_kw"
    ] = parsed.loc[
        mask_iberdrola_kwmax,
        "kwmax_num"
    ]

    parsed.loc[
        mask_iberdrola_kwmax,
        "criterio_demanda_mensual"
    ] = "kwmax medido en recibo Iberdrola"

    parsed.loc[
        mask_iberdrola_kwmax,
        "criterio_demanda_real"
    ] = "kwmax medido en recibo Iberdrola"

    parsed.loc[
        mask_iberdrola_kwmax,
        "tarifa_norm"
    ] = parsed.loc[
        mask_iberdrola_kwmax,
        "tarifa_norm"
    ].fillna("IBERDROLA")

# PDBT no tiene demanda medida.
# Se estimará más adelante con perfiles NREL, después de construir muestra_con_recibo.
mask_pdbt = parsed["tarifa_norm"].eq("PDBT")

parsed.loc[
    mask_pdbt,
    "criterio_demanda_mensual"
] = "pendiente estimación NREL"

# ============================================================
# Construcción global de muestra validada con recibo
# ============================================================

@st.cache_data(show_spinner=False)
def construir_muestra_con_recibo_global(parsed, general_data, mall_col):
    """
    Construye la muestra validada de locales ocupados con recibo
    antes de renderizar los tabs.

    Esta función replica la lógica base de Calidad de Datos:
    - mismo centro comercial
    - match por medidor
    - match por cliente
    - match por nombre comercial
    - match por no. de local en textos del parser
    """

    muestra_con_recibo_rows = []
    coverage_rows = []
    match_summary_rows = []
    sin_match_rows = []
    parser_sin_match_rows = []
    debug_sin_tarifa_rows = []

    if parsed.empty or general_data.empty or mall_col is None:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    general_mall_col = first_existing_column(
        general_data,
        ["NOMBRE DEL CC", "CENTRO COMERCIAL", "CC", "PLAZA", "source_sheet"]
    )

    if general_mall_col is None:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame()
        )

    for mall_name in sorted(parsed[mall_col].dropna().unique()):

        parser_cc = parsed[
            parsed[mall_col] == mall_name
        ].copy()

        # --------------------------------------------------
        # Llaves normalizadas del parser para guardar el match original
        # --------------------------------------------------
        # Importante: antes solo se creaban sets de medidor/cliente/nombre,
        # pero no columnas dentro de parser_cc. Por eso después no se podía
        # copiar tarifa/medidor/no_servicio desde el parser original.

        parser_cc["_key_medidor"] = (
            normalize_meter_cc(parser_cc["medidor"])
            if "medidor" in parser_cc.columns
            else ""
        )

        parser_cc["_key_cliente"] = (
            normalize_cc_key(parser_cc["cliente_nombre"])
            if "cliente_nombre" in parser_cc.columns
            else ""
        )

        parser_cc["_key_nombre"] = (
            normalize_cc_key(parser_cc["recibos_subgroup"])
            if "recibos_subgroup" in parser_cc.columns
            else ""
        )

        mall_name_norm = str(mall_name).strip().upper()

        mall_name_norm = str(mall_name).strip().upper()

        general_cc = general_data[
            general_data[general_mall_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .apply(lambda x: x in mall_name_norm or mall_name_norm in x)
        ].copy()

        if general_cc.empty:
            continue

        cliente_status = (
            general_cc["CLIENTE"]
            if "CLIENTE" in general_cc.columns
            else pd.Series([None] * len(general_cc))
        )

        nombre_status = (
            general_cc["NOMBRE COMERCIAL"]
            if "NOMBRE COMERCIAL" in general_cc.columns
            else pd.Series([None] * len(general_cc))
        )

        disponible_o_vacio = (
            cliente_status.isna()
            | nombre_status.isna()
            | (cliente_status.astype(str).str.strip() == "")
            | (nombre_status.astype(str).str.strip() == "")
            | (cliente_status.astype(str).str.strip().str.lower() == "disponible")
            | (nombre_status.astype(str).str.strip().str.lower() == "disponible")
        )

        general_ocupados_cc = general_cc[~disponible_o_vacio].copy()

        general_meter_col = first_existing_column(
            general_ocupados_cc,
            [
                "No. De medidor",
                "No. de medidor",
                "No de medidor",
                "MEDIDOR",
                "Medidor"
            ]
        )

        general_ocupados_cc["_key_medidor"] = (
            normalize_meter_cc(general_ocupados_cc[general_meter_col])
            if general_meter_col
            else ""
        )

        general_ocupados_cc["_key_cliente"] = (
            normalize_cc_key(general_ocupados_cc["CLIENTE"])
            if "CLIENTE" in general_ocupados_cc.columns
            else ""
        )

        general_ocupados_cc["_key_nombre_comercial"] = (
            normalize_cc_key(general_ocupados_cc["NOMBRE COMERCIAL"])
            if "NOMBRE COMERCIAL" in general_ocupados_cc.columns
            else ""
        )

        parser_medidores = (
            set(parser_cc["_key_medidor"].dropna().astype(str).unique())
            if "_key_medidor" in parser_cc.columns
            else set()
        )

        parser_clientes = (
            set(parser_cc["_key_cliente"].dropna().astype(str).unique())
            if "_key_cliente" in parser_cc.columns
            else set()
        )

        parser_nombres = (
            set(parser_cc["_key_nombre"].dropna().astype(str).unique())
            if "_key_nombre" in parser_cc.columns
            else set()
        )

        parser_medidores.discard("")
        parser_clientes.discard("")
        parser_nombres.discard("")

        general_ocupados_cc["_match_medidor"] = (
            general_ocupados_cc["_key_medidor"].isin(parser_medidores)
            & (general_ocupados_cc["_key_medidor"] != "")
        )

        general_ocupados_cc["_match_cliente"] = (
            general_ocupados_cc["_key_cliente"]
            .apply(lambda x: has_partial_match_cc(x, parser_clientes))
        )

        general_ocupados_cc["_match_nombre_comercial"] = (
            general_ocupados_cc["_key_nombre_comercial"]
            .apply(lambda x: has_partial_match_cc(x, parser_nombres))
        )

        # --------------------------------------------------
        # Match por número de local en dirección/textos del parser
        # --------------------------------------------------

        general_local_col = first_existing_column(
            general_ocupados_cc,
            [
                "No de Local",
                "No de local",
                "No. de Local",
                "No. de local",
                "No Local",
                "No. Local",
                "LOCAL",
                "Local"
            ]
        )

        parser_textos_local = set()

        for col in [
            "direccion_completa",
            "direccion_raw",
            "file_path",
            "file_name",
            "recibos_subgroup"
        ]:
            if col in parser_cc.columns:
                parser_textos_local.update(
                    parser_cc[col]
                    .dropna()
                    .astype(str)
                    .unique()
                )

        def generar_variantes_no_local(no_local):
            raw = str(no_local).upper().strip()

            if raw in ["", "NAN", "NONE"]:
                return []

            raw = (
                raw
                .replace("LOCAL", "")
                .replace("LOC.", "")
                .replace("LOC", "")
                .replace("NO.", "")
                .replace("NO", "")
                .replace("#", "")
                .strip()
            )

            raw = re.sub(r"\s+", " ", raw)

            variantes = set()

            prefijo_match = re.match(r"^([A-Z0-9]+)[\-\s]*", raw)
            prefijo_base = prefijo_match.group(1) if prefijo_match else ""

            partes = re.split(r",|/|\bY\b|\bE\b|;", raw)

            for parte in partes:
                parte = parte.strip()

                if not parte:
                    continue

                compact = re.sub(r"[^A-Z0-9]", "", parte)

                if len(compact) >= 3:
                    variantes.add(compact)

                if prefijo_base and re.fullmatch(r"[0-9]+[A-Z]?", compact):
                    variantes.add(f"{prefijo_base}{compact}")

                m = re.match(r"^([A-Z0-9]+)[\-\s]*([0-9]+[A-Z]?)$", parte)
                if m:
                    variantes.add(f"{m.group(1)}{m.group(2)}")

            full_compact = re.sub(r"[^A-Z0-9]", "", raw)

            if len(full_compact) >= 3:
                variantes.add(full_compact)

            variantes = {
                v for v in variantes
                if len(v) >= 3
                and v not in ["LOCAL", "NAN", "NONE"]
            }

            return list(variantes)

        def normalizar_texto_para_local(value):
            return re.sub(
                r"[^A-Z0-9]",
                "",
                normalizar_texto_simple(value)
            )

        def match_no_local_en_texto(no_local, textos_parser):
            variantes = generar_variantes_no_local(no_local)

            if not variantes:
                return False

            for texto in textos_parser:
                texto_key = normalizar_texto_para_local(texto)

                for variante in variantes:
                    if variante in texto_key:
                        return True

            return False

        general_ocupados_cc["_match_no_local_direccion"] = (
            general_ocupados_cc[general_local_col]
            .apply(lambda x: match_no_local_en_texto(x, parser_textos_local))
            if general_local_col
            else False
        )

        general_ocupados_cc["_tiene_recibo"] = (
            general_ocupados_cc["_match_cliente"]
            | general_ocupados_cc["_match_nombre_comercial"]
            | general_ocupados_cc["_match_no_local_direccion"]
            | general_ocupados_cc["_match_medidor"]
        )

        def criterio_match(row):
            criterios = []

            if row["_match_cliente"]:
                criterios.append("Cliente")

            if row["_match_nombre_comercial"]:
                criterios.append("Nombre comercial")

            if row["_match_no_local_direccion"]:
                criterios.append("No. local / dirección")

            if row["_match_medidor"]:
                criterios.append("Medidor")

            return ", ".join(criterios)

        general_ocupados_cc["criterio_match"] = (
            general_ocupados_cc.apply(
                criterio_match,
                axis=1
            )
        )

        # --------------------------------------------------
        # Guardar información del parser que originó el match
        # --------------------------------------------------
        # Esto evita que después, en Densidad de demanda,
        # tengamos que volver a adivinar contra el parser.

        parser_cols_match = [
            col for col in [
                "no_servicio",
                "medidor",
                "tarifa_norm",
                "demanda_contratada_kw",
                "cliente_nombre",
                "recibos_subgroup",
                "direccion_completa",
                "direccion_raw",
                "file_path",
                "file_name"
            ]
            if col in parser_cc.columns
        ]

        for col in [
            "parser_no_servicio_match",
            "parser_medidor_match",
            "parser_tarifa_match",
            "demanda_contratada_kw",
            "parser_cliente_match",
            "parser_recibos_subgroup_match",
            "parser_direccion_match",
            "parser_file_match",
            "parser_criterio_match"
        ]:
            if col not in general_ocupados_cc.columns:
                general_ocupados_cc[col] = pd.NA

        def copiar_parser_match_a_general(mask_general, mask_parser, criterio):
            if not mask_general.any() or not mask_parser.any():
                return

            parser_match = parser_cc.loc[mask_parser, parser_cols_match].copy()

            if parser_match.empty:
                return
            
            # --------------------------------------------------
            # Expandir match a todo el no_servicio
            # --------------------------------------------------
            # Si encontramos una fila del parser por medidor, cliente,
            # nombre comercial o local, primero identificamos su no_servicio
            # y después traemos TODAS las filas de ese mismo no_servicio.
            #
            # Esto permite capturar cambios durante el año:
            # - varios medidores para el mismo no_servicio
            # - varios cliente_nombre para el mismo no_servicio
            # - varios recibos_subgroup para el mismo no_servicio

            if "no_servicio" in parser_match.columns and "no_servicio" in parser_cc.columns:

                servicios_match = (
                    parser_match["no_servicio"]
                    .dropna()
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .str.strip()
                )

                servicios_match = sorted({
                    s for s in servicios_match
                    if s.upper() not in ["", "NAN", "NONE", "<NA>"]
                })

                if len(servicios_match) == 1:
                    servicio_match = servicios_match[0]

                    parser_cc["_key_no_servicio_temp"] = normalize_service_cc(
                        parser_cc["no_servicio"]
                    )

                    parser_match = parser_cc.loc[
                        parser_cc["_key_no_servicio_temp"].eq(
                            normalize_service_cc(pd.Series([servicio_match])).iloc[0]
                        ),
                        parser_cols_match
                    ].copy()

            # --------------------------------------------------
            # Evitar asignar varios no_servicio a un solo local
            # --------------------------------------------------
            # Un cliente puede aparecer con varios no_servicio en el parser.
            # Eso NO significa que todos correspondan al mismo local de DG.
            #
            # Si el criterio es amplio, intentamos desambiguar con:
            # 1. NOMBRE COMERCIAL contra file_path / dirección / file_name
            # 2. No. local contra file_path / dirección / file_name
            #
            # Si después de eso siguen quedando varios no_servicio,
            # NO copiamos nada para evitar asignar un servicio incorrecto.

            if criterio in [
                "Cliente flexible",
                "Nombre comercial flexible",
                "Cliente original",
                "Nombre comercial original"
            ] and "no_servicio" in parser_match.columns:

                def servicios_unicos_de_parser_match(df_match):
                    servicios = (
                        df_match["no_servicio"]
                        .dropna()
                        .astype(str)
                        .str.replace(r"\.0$", "", regex=True)
                        .str.strip()
                    )

                    return sorted({
                        s for s in servicios
                        if s.upper() not in ["", "NAN", "NONE", "<NA>"]
                    })

                servicios_unicos_match = servicios_unicos_de_parser_match(
                    parser_match
                )

                if len(servicios_unicos_match) > 1:

                    # Solo intentamos desambiguar si mask_general apunta a un local.
                    idxs_general = general_ocupados_cc.loc[mask_general].index

                    if len(idxs_general) == 1:

                        idx_general = idxs_general[0]
                        row_general_match = general_ocupados_cc.loc[idx_general]

                        parser_text_cols_desempate = [
                            col for col in [
                                "direccion_completa",
                                "direccion_raw",
                                "file_path",
                                "file_name",
                                "recibos_subgroup"
                            ]
                            if col in parser_match.columns
                        ]

                        parser_match_desempatado = parser_match.copy()

                        # ----------------------------------------------
                        # 1) Desempate por nombre comercial
                        # ----------------------------------------------

                        nombre_comercial_general = (
                            row_general_match.get("NOMBRE COMERCIAL", "")
                            if "NOMBRE COMERCIAL" in general_ocupados_cc.columns
                            else ""
                        )

                        if (
                            pd.notna(nombre_comercial_general)
                            and str(nombre_comercial_general).strip() != ""
                            and parser_text_cols_desempate
                        ):

                            textos_parser_match = pd.Series(
                                "",
                                index=parser_match_desempatado.index,
                                dtype="object"
                            )

                            for col_texto in parser_text_cols_desempate:
                                textos_parser_match = (
                                    textos_parser_match.astype(str)
                                    + " "
                                    + parser_match_desempatado[col_texto]
                                    .fillna("")
                                    .astype(str)
                                )

                            mask_nombre_en_texto = textos_parser_match.apply(
                                lambda texto: has_partial_match_cc(
                                    nombre_comercial_general,
                                    [texto]
                                )
                            )

                            if mask_nombre_en_texto.any():
                                parser_match_desempatado = (
                                    parser_match_desempatado.loc[
                                        mask_nombre_en_texto
                                    ].copy()
                                )

                        servicios_despues_nombre = servicios_unicos_de_parser_match(
                            parser_match_desempatado
                        )

                        if len(servicios_despues_nombre) == 1:
                            parser_match = parser_match_desempatado.copy()

                        else:
                            # ------------------------------------------
                            # 2) Desempate por No. local / dirección
                            # ------------------------------------------

                            if (
                                general_local_col
                                and parser_text_cols_desempate
                                and general_local_col in general_ocupados_cc.columns
                            ):

                                variantes_local_desempate = generar_variantes_no_local(
                                    row_general_match.get(general_local_col, "")
                                )

                                if variantes_local_desempate:

                                    textos_parser_match = pd.Series(
                                        "",
                                        index=parser_match.index,
                                        dtype="object"
                                    )

                                    for col_texto in parser_text_cols_desempate:
                                        textos_parser_match = (
                                            textos_parser_match.astype(str)
                                            + " "
                                            + parser_match[col_texto]
                                            .fillna("")
                                            .astype(str)
                                        )

                                    textos_parser_key = textos_parser_match.apply(
                                        normalizar_texto_para_local
                                    )

                                    mask_local_en_texto = textos_parser_key.apply(
                                        lambda texto: any(
                                            variante in texto
                                            for variante in variantes_local_desempate
                                        )
                                    )

                                    if mask_local_en_texto.any():
                                        parser_match_desempatado_local = (
                                            parser_match.loc[
                                                mask_local_en_texto
                                            ].copy()
                                        )

                                        servicios_despues_local = (
                                            servicios_unicos_de_parser_match(
                                                parser_match_desempatado_local
                                            )
                                        )

                                        if len(servicios_despues_local) == 1:
                                            parser_match = (
                                                parser_match_desempatado_local.copy()
                                            )

                    servicios_unicos_match = servicios_unicos_de_parser_match(
                        parser_match
                    )

                    if len(servicios_unicos_match) > 1:
                        return

            def valores_unicos_join(serie, upper=False):
                vals = []

                for v in serie.dropna().astype(str).tolist():
                    v = v.strip()

                    if upper:
                        v = v.upper()

                    if v in ["", "nan", "None", "NONE", "NAN", "<NA>"]:
                        continue

                    if re.match(r"^\d+\.0$", v):
                        v = v.replace(".0", "")

                    vals.append(v)

                vals = sorted(set(vals))

                return " | ".join(vals) if vals else pd.NA

            idxs = general_ocupados_cc.loc[mask_general].index

            if "no_servicio" in parser_match.columns:
                general_ocupados_cc.loc[
                    idxs,
                    "parser_no_servicio_match"
                ] = valores_unicos_join(
                    parser_match["no_servicio"]
                )

            if "medidor" in parser_match.columns:
                general_ocupados_cc.loc[
                    idxs,
                    "parser_medidor_match"
                ] = valores_unicos_join(
                    parser_match["medidor"],
                    upper=True
                )

            if "tarifa_norm" in parser_match.columns:
                general_ocupados_cc.loc[
                    idxs,
                    "parser_tarifa_match"
                ] = valores_unicos_join(
                    parser_match["tarifa_norm"],
                    upper=True
                )

            if "demanda_contratada_kw" in parser_match.columns:

                demanda_parser_vals = pd.to_numeric(
                    parser_match["demanda_contratada_kw"],
                    errors="coerce"
                ).dropna()

                if not demanda_parser_vals.empty:
                    general_ocupados_cc.loc[
                        idxs,
                        "demanda_contratada_kw"
                    ] = demanda_parser_vals.max()

            if "cliente_nombre" in parser_match.columns:
                general_ocupados_cc.loc[
                    idxs,
                    "parser_cliente_match"
                ] = valores_unicos_join(
                    parser_match["cliente_nombre"]
                )

            if "recibos_subgroup" in parser_match.columns:
                general_ocupados_cc.loc[
                    idxs,
                    "parser_recibos_subgroup_match"
                ] = valores_unicos_join(
                    parser_match["recibos_subgroup"]
                )

            direccion_val = pd.NA

            if "direccion_completa" in parser_match.columns:
                direccion_val = valores_unicos_join(
                    parser_match["direccion_completa"]
                )

            elif "direccion_raw" in parser_match.columns:
                direccion_val = valores_unicos_join(
                    parser_match["direccion_raw"]
                )

            general_ocupados_cc.loc[
                idxs,
                "parser_direccion_match"
            ] = direccion_val

            file_val = pd.NA

            if "file_name" in parser_match.columns:
                file_val = valores_unicos_join(
                    parser_match["file_name"]
                )

            elif "file_path" in parser_match.columns:
                file_val = valores_unicos_join(
                    parser_match["file_path"]
                )

            general_ocupados_cc.loc[
                idxs,
                "parser_file_match"
            ] = file_val

            general_ocupados_cc.loc[
                idxs,
                "parser_criterio_match"
            ] = criterio

        # Match por medidor
        if "_key_medidor" in general_ocupados_cc.columns and "_key_medidor" in parser_cc.columns:
            for key in general_ocupados_cc["_key_medidor"].dropna().unique():
                key = str(key).strip()

                if key == "" or key.upper() in ["NAN", "NONE", "<NA>"]:
                    continue

                mask_general = general_ocupados_cc["_key_medidor"].astype(str).eq(key)
                mask_parser = parser_cc["_key_medidor"].astype(str).eq(key)

                copiar_parser_match_a_general(
                    mask_general=mask_general,
                    mask_parser=mask_parser,
                    criterio="Medidor original"
                )

        # Match por cliente
        if "_key_cliente" in general_ocupados_cc.columns and "_key_cliente" in parser_cc.columns:
            for key in general_ocupados_cc["_key_cliente"].dropna().unique():
                key = str(key).strip()

                if key == "" or key.upper() in ["NAN", "NONE", "<NA>"]:
                    continue

                mask_general = general_ocupados_cc["_key_cliente"].astype(str).eq(key)
                mask_parser = parser_cc["_key_cliente"].astype(str).eq(key)

                mask_general = mask_general & general_ocupados_cc["parser_tarifa_match"].isna()

                copiar_parser_match_a_general(
                    mask_general=mask_general,
                    mask_parser=mask_parser,
                    criterio="Cliente original"
                )

        # Match por nombre comercial
        if "_key_nombre_comercial" in general_ocupados_cc.columns and "_key_nombre" in parser_cc.columns:
            for key in general_ocupados_cc["_key_nombre_comercial"].dropna().unique():
                key = str(key).strip()

                if key == "" or key.upper() in ["NAN", "NONE", "<NA>"]:
                    continue

                mask_general = general_ocupados_cc["_key_nombre_comercial"].astype(str).eq(key)
                mask_parser = parser_cc["_key_nombre"].astype(str).eq(key)

                mask_general = mask_general & general_ocupados_cc["parser_tarifa_match"].isna()

                copiar_parser_match_a_general(
                    mask_general=mask_general,
                    mask_parser=mask_parser,
                    criterio="Nombre comercial original"
                )

        # --------------------------------------------------
        # Fallback flexible por cliente / nombre para copiar datos del parser
        # --------------------------------------------------
        # _match_cliente y _match_nombre_comercial pueden venir de coincidencias
        # flexibles. Los bloques anteriores solo copian datos cuando la llave es
        # exactamente igual. Por eso algunos locales quedan con _tiene_recibo=True
        # pero sin parser_tarifa_match.
        #
        # Aquí buscamos nuevamente el parser con la misma lógica flexible y copiamos
        # no_servicio, medidor, tarifa_norm, file_path, etc.

        # 1) Copiar datos del parser por cliente flexible
        if "_key_cliente" in general_ocupados_cc.columns and "_key_cliente" in parser_cc.columns:

            candidatos_cliente_flexible = general_ocupados_cc[
                general_ocupados_cc["_match_cliente"]
                & (
                    general_ocupados_cc["parser_tarifa_match"].isna()
                    | general_ocupados_cc["parser_tarifa_match"]
                    .astype(str)
                    .str.upper()
                    .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
                )
            ].copy()

            for idx_general, row_general in candidatos_cliente_flexible.iterrows():

                key_general = str(row_general.get("_key_cliente", "")).strip()

                if key_general == "" or key_general.upper() in ["NAN", "NONE", "<NA>"]:
                    continue

                mask_parser_cliente_flexible = parser_cc["_key_cliente"].apply(
                    lambda x: has_partial_match_cc(
                        key_general,
                        [x]
                    )
                )

                if not mask_parser_cliente_flexible.any():
                    continue

                mask_general_cliente_flexible = pd.Series(
                    False,
                    index=general_ocupados_cc.index
                )

                mask_general_cliente_flexible.loc[idx_general] = True

                copiar_parser_match_a_general(
                    mask_general=mask_general_cliente_flexible,
                    mask_parser=mask_parser_cliente_flexible,
                    criterio="Cliente flexible"
                )

        # 2) Copiar datos del parser por nombre comercial flexible
        if "_key_nombre_comercial" in general_ocupados_cc.columns and "_key_nombre" in parser_cc.columns:

            candidatos_nombre_flexible = general_ocupados_cc[
                general_ocupados_cc["_match_nombre_comercial"]
                & (
                    general_ocupados_cc["parser_tarifa_match"].isna()
                    | general_ocupados_cc["parser_tarifa_match"]
                    .astype(str)
                    .str.upper()
                    .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
                )
            ].copy()

            for idx_general, row_general in candidatos_nombre_flexible.iterrows():

                key_general = str(row_general.get("_key_nombre_comercial", "")).strip()

                if key_general == "" or key_general.upper() in ["NAN", "NONE", "<NA>"]:
                    continue

                mask_parser_nombre_flexible = parser_cc["_key_nombre"].apply(
                    lambda x: has_partial_match_cc(
                        key_general,
                        [x]
                    )
                )

                if not mask_parser_nombre_flexible.any():
                    continue

                mask_general_nombre_flexible = pd.Series(
                    False,
                    index=general_ocupados_cc.index
                )

                mask_general_nombre_flexible.loc[idx_general] = True

                copiar_parser_match_a_general(
                    mask_general=mask_general_nombre_flexible,
                    mask_parser=mask_parser_nombre_flexible,
                    criterio="Nombre comercial flexible"
                )

        # --------------------------------------------------
        # Match por No. local / dirección para copiar datos del parser
        # --------------------------------------------------
        # Ya usamos No. local / dirección para marcar _tiene_recibo.
        # Aquí copiamos también los datos reales del parser:
        # no_servicio, medidor, tarifa_norm, file_path, etc.
        #
        # Esto evita que un local entre como "con recibo" pero quede
        # con TARIFA_FINAL = SIN TARIFA.

        parser_text_cols_local = [
            col for col in [
                "direccion_completa",
                "direccion_raw",
                "file_path",
                "file_name",
                "recibos_subgroup"
            ]
            if col in parser_cc.columns
        ]

        if (
            general_local_col
            and parser_text_cols_local
            and "_match_no_local_direccion" in general_ocupados_cc.columns
            and "parser_tarifa_match" in general_ocupados_cc.columns
        ):

            candidatos_local = general_ocupados_cc[
                general_ocupados_cc["_match_no_local_direccion"]
                & (
                    general_ocupados_cc["parser_tarifa_match"].isna()
                    | general_ocupados_cc["parser_tarifa_match"]
                    .astype(str)
                    .str.upper()
                    .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
                )
            ].copy()

            for idx_general, row_general in candidatos_local.iterrows():

                variantes_local = generar_variantes_no_local(
                    row_general.get(general_local_col, "")
                )

                if not variantes_local:
                    continue

                textos_parser = pd.Series(
                    "",
                    index=parser_cc.index,
                    dtype="object"
                )

                for col_texto in parser_text_cols_local:
                    textos_parser = (
                        textos_parser.astype(str)
                        + " "
                        + parser_cc[col_texto].fillna("").astype(str)
                    )

                textos_parser_key = textos_parser.apply(
                    normalizar_texto_para_local
                )

                mask_parser_local = textos_parser_key.apply(
                    lambda texto: any(
                        variante in texto
                        for variante in variantes_local
                    )
                )

                if not mask_parser_local.any():
                    continue

                mask_general_local = pd.Series(
                    False,
                    index=general_ocupados_cc.index
                )

                mask_general_local.loc[idx_general] = True

                copiar_parser_match_a_general(
                    mask_general=mask_general_local,
                    mask_parser=mask_parser_local,
                    criterio="No. local / dirección original"
                )

        # --------------------------------------------------
        # Tarifa final global del local
        # --------------------------------------------------
        # La tarifa final debe venir primero del parser, porque DG
        # no tiene tarifa para todos los CCs ni para todos los clientes.
        #
        # Esta columna queda guardada en muestra_con_recibo_global y debe
        # ser la fuente única para composición, densidad y demanda.

        general_ocupados_cc["TARIFA_FINAL"] = pd.NA

        posibles_cols_tarifa_global = [
            "parser_tarifa_match",   # tarifa copiada del parser que hizo match
            "tarifa_norm",           # por si la muestra ya trae tarifa normalizada
            "tarifa",                # tarifa cruda del parser, si existiera
            "TARIFA_ANALISIS",       # respaldo DG
            "TARIFA",                # respaldo DG
            "Tarifa"                 # respaldo visual
        ]

        for col_tarifa in posibles_cols_tarifa_global:

            if col_tarifa not in general_ocupados_cc.columns:
                continue

            valores_tarifa = normalize_tarifa_series(
                general_ocupados_cc[col_tarifa]
            )

            valores_tarifa = valores_tarifa.replace({
                "": pd.NA,
                "NAN": pd.NA,
                "NONE": pd.NA,
                "<NA>": pd.NA,
                "SIN TARIFA": pd.NA,
                "SIN TARIFA / SIN DEMANDA ASOCIADA": pd.NA
            })

            faltantes_tarifa = (
                general_ocupados_cc["TARIFA_FINAL"].isna()
                | general_ocupados_cc["TARIFA_FINAL"]
                .astype(str)
                .str.upper()
                .isin([
                    "",
                    "NAN",
                    "NONE",
                    "<NA>",
                    "SIN TARIFA",
                    "SIN TARIFA / SIN DEMANDA ASOCIADA"
                ])
            )

            general_ocupados_cc.loc[
                faltantes_tarifa,
                "TARIFA_FINAL"
            ] = valores_tarifa.loc[faltantes_tarifa]

        general_ocupados_cc["TARIFA_FINAL"] = (
            general_ocupados_cc["TARIFA_FINAL"]
            .fillna("SIN TARIFA")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        # --------------------------------------------------
        # Evitar que un mismo no_servicio cuente para varios locales
        # --------------------------------------------------
        # Regla:
        # Un no_servicio del parser representa un servicio/local eléctrico.
        # Por lo tanto, dentro de un mismo centro comercial, un no_servicio
        # solo puede quedar asignado a UN local ocupado de DG.
        #
        # Si el mismo no_servicio quedó copiado a varios locales por un match
        # amplio de CLIENTE o NOMBRE COMERCIAL, conservamos el match más fuerte
        # y limpiamos los demás.
        #
        # Esto corrige casos como:
        # - ITX RETAIL MEXICO SA DE CV
        #   donde Bershka / Stradivarius sí tienen recibo,
        #   pero Pull & Bear no debe tomar esos mismos no_servicio.
        #
        # - Coppel / Burger King / etc.
        #   donde un cliente puede repetirse en varios locales.

        parser_match_cols_limpieza = [
            "parser_no_servicio_match",
            "parser_medidor_match",
            "parser_tarifa_match",
            "parser_cliente_match",
            "parser_recibos_subgroup_match",
            "parser_direccion_match",
            "parser_file_match",
            "parser_criterio_match"
        ]

        def split_pipe_normalizado(value, tipo="texto"):
            if pd.isna(value):
                return []

            vals = []

            for raw in str(value).split("|"):
                raw = raw.strip()

                if raw.upper() in ["", "NAN", "NONE", "<NA>", "SIN TARIFA"]:
                    continue

                if tipo == "servicio":
                    key = normalize_service_cc(pd.Series([raw])).iloc[0]

                elif tipo == "medidor":
                    key = normalize_meter_cc(pd.Series([raw])).iloc[0]

                elif tipo == "cc":
                    key = normalize_cc_key(pd.Series([raw])).iloc[0]

                else:
                    key = str(raw).strip().upper()

                if key and str(key).upper() not in ["", "NAN", "NONE", "<NA>"]:
                    vals.append(key)

            return sorted(set(vals))

        def texto_contiene_no_local(texto, no_local):
            if pd.isna(texto) or pd.isna(no_local):
                return False

            texto_key = re.sub(
                r"[^A-Z0-9]",
                "",
                normalizar_texto_simple(texto)
            )

            variantes = generar_variantes_no_local(no_local)

            if not variantes:
                return False

            return any(v in texto_key for v in variantes)

        def score_match_local(row):
            """
            Puntaje para decidir qué local se queda con un no_servicio duplicado.

            Prioridad:
            1. Medidor exacto DG-parser
            2. No. local aparece en file_path/dirección
            3. Nombre comercial exacto DG-parser
            4. Match por nombre comercial
            5. Match por cliente
            """

            score = 0

            criterio = str(row.get("parser_criterio_match", "")).upper()
            criterio_original = str(row.get("criterio_match", "")).upper()

            # 1) Medidor exacto
            medidores_parser = split_pipe_normalizado(
                row.get("parser_medidor_match", ""),
                tipo="medidor"
            )

            medidor_dg = str(row.get("_key_medidor", "")).strip()

            if medidor_dg and medidor_dg in medidores_parser:
                score += 1000

            if "MEDIDOR" in criterio or "MEDIDOR" in criterio_original:
                score += 700

            # 2) No. local en dirección / file_path
            no_local_val = (
                row.get(general_local_col, "")
                if general_local_col
                else ""
            )

            textos_parser_match = [
                row.get("parser_file_match", ""),
                row.get("parser_direccion_match", "")
            ]

            if any(
                texto_contiene_no_local(texto, no_local_val)
                for texto in textos_parser_match
            ):
                score += 500

            if "LOCAL" in criterio or "DIRECCION" in criterio or "DIRECCIÓN" in criterio:
                score += 400

            # 2.5) Nombre comercial aparece en textos del parser
            # Esto corrige casos donde el no_servicio quedó asignado
            # por CLIENTE corporativo a otro local, pero el file_path
            # claramente contiene la marca correcta.
            nombre_comercial_raw = (
                row.get("NOMBRE COMERCIAL", "")
                if "NOMBRE COMERCIAL" in row.index
                else ""
            )

            textos_parser_nombre = [
                row.get("parser_file_match", ""),
                row.get("parser_direccion_match", ""),
                row.get("parser_recibos_subgroup_match", "")
            ]

            if (
                pd.notna(nombre_comercial_raw)
                and str(nombre_comercial_raw).strip() != ""
                and any(
                    has_partial_match_cc(nombre_comercial_raw, [texto_parser])
                    for texto_parser in textos_parser_nombre
                    if pd.notna(texto_parser)
                )
            ):
                score += 650

            # 3) Nombre comercial exacto
            nombre_dg_key = (
                normalize_cc_key(
                    pd.Series([row.get("NOMBRE COMERCIAL", "")])
                ).iloc[0]
                if "NOMBRE COMERCIAL" in row.index
                else ""
            )

            nombres_parser = split_pipe_normalizado(
                row.get("parser_recibos_subgroup_match", ""),
                tipo="cc"
            )

            if nombre_dg_key and nombre_dg_key in nombres_parser:
                score += 300

            if "NOMBRE COMERCIAL" in criterio or "NOMBRE COMERCIAL" in criterio_original:
                score += 200

            # 4) Cliente exacto
            cliente_dg_key = (
                normalize_cc_key(
                    pd.Series([row.get("CLIENTE", "")])
                ).iloc[0]
                if "CLIENTE" in row.index
                else ""
            )

            clientes_parser = split_pipe_normalizado(
                row.get("parser_cliente_match", ""),
                tipo="cc"
            )

            if cliente_dg_key and cliente_dg_key in clientes_parser:
                score += 50

            if "CLIENTE" in criterio or "CLIENTE" in criterio_original:
                score += 25

            return score

        def limpiar_match_parser_en_indices(indices, motivo):
            for col_limpieza in parser_match_cols_limpieza:
                if col_limpieza in general_ocupados_cc.columns:
                    general_ocupados_cc.loc[
                        indices,
                        col_limpieza
                    ] = pd.NA

            general_ocupados_cc.loc[
                indices,
                "criterio_match"
            ] = (
                general_ocupados_cc.loc[
                    indices,
                    "criterio_match"
                ].astype(str)
                + " | SIN RECIBO CONFIRMADO: "
                + motivo
            )

        # Mapa no_servicio -> filas de DG que lo tienen asignado
        servicio_a_indices = {}

        if "parser_no_servicio_match" in general_ocupados_cc.columns:
            for idx_servicio, value_servicio in general_ocupados_cc[
                "parser_no_servicio_match"
            ].dropna().items():

                servicios_idx = split_pipe_normalizado(
                    value_servicio,
                    tipo="servicio"
                )

                for servicio_idx in servicios_idx:
                    servicio_a_indices.setdefault(
                        servicio_idx,
                        []
                    ).append(idx_servicio)

        # Resolver servicios duplicados
        for servicio_dup, indices_dup in servicio_a_indices.items():

            indices_dup = sorted(set(indices_dup))

            if len(indices_dup) <= 1:
                continue

            candidatos_dup = general_ocupados_cc.loc[
                indices_dup
            ].copy()

            candidatos_dup["_score_match_servicio"] = (
                candidatos_dup.apply(
                    score_match_local,
                    axis=1
                )
            )

            candidatos_dup = candidatos_dup.sort_values(
                [
                    "_score_match_servicio"
                ],
                ascending=False
            )

            idx_ganador = candidatos_dup.index[0]

            indices_perdedores = [
                idx for idx in indices_dup
                if idx != idx_ganador
            ]

            if indices_perdedores:
                limpiar_match_parser_en_indices(
                    indices_perdedores,
                    f"no_servicio {servicio_dup} ya asignado a otro local"
                )

        # Si después de limpiar un local quedó con varios no_servicio
        # y no fue match por medidor, también lo limpiamos.
        # Un match amplio que trae varios servicios no es confiable.
        if "parser_no_servicio_match" in general_ocupados_cc.columns:
            for idx_multi, row_multi in general_ocupados_cc.iterrows():

                servicios_multi = split_pipe_normalizado(
                    row_multi.get("parser_no_servicio_match", ""),
                    tipo="servicio"
                )

                if len(servicios_multi) <= 1:
                    continue

                criterio_multi = str(
                    row_multi.get("parser_criterio_match", "")
                ).upper()

                criterio_original_multi = str(
                    row_multi.get("criterio_match", "")
                ).upper()

                es_match_fuerte = (
                    "MEDIDOR" in criterio_multi
                    or "MEDIDOR" in criterio_original_multi
                    or "LOCAL" in criterio_multi
                    or "DIRECCION" in criterio_multi
                    or "DIRECCIÓN" in criterio_multi
                )

                if not es_match_fuerte:
                    limpiar_match_parser_en_indices(
                        [idx_multi],
                        "match amplio con varios no_servicio posibles"
                    )

        # --------------------------------------------------
        # Segunda pasada: rescatar matches seguros no asignados
        # --------------------------------------------------
        # Después de limpiar duplicados, intentamos recuperar locales
        # que quedaron sin parser asignado.
        #
        # Solo hacemos match si encontramos EXACTAMENTE un no_servicio
        # disponible en el parser.
        #
        # Reglas de rescate:
        # 1. No. de local aparece en file_path / dirección / nombre del archivo.
        # 2. Nombre comercial exacto y no ambiguo.
        #
        # No usamos CLIENTE solo, porque puede agrupar varias marcas
        # del mismo grupo corporativo.

        def servicio_parser_disponible(servicio):
            servicio_key = normalize_service_cc(
                pd.Series([servicio])
            ).iloc[0]

            if not servicio_key:
                return False

            servicios_asignados_actuales = set()

            if "parser_no_servicio_match" in general_ocupados_cc.columns:
                for value_asignado in general_ocupados_cc[
                    "parser_no_servicio_match"
                ].dropna().astype(str):

                    for raw in value_asignado.split("|"):
                        raw = raw.strip()

                        if raw.upper() in ["", "NAN", "NONE", "<NA>"]:
                            continue

                        servicios_asignados_actuales.add(
                            normalize_service_cc(
                                pd.Series([raw])
                            ).iloc[0]
                        )

            return servicio_key not in servicios_asignados_actuales

        def servicios_unicos_parser(mask_parser_candidato):
            if not mask_parser_candidato.any():
                return []

            if "no_servicio" not in parser_cc.columns:
                return []

            servicios = []

            for value_servicio in parser_cc.loc[
                mask_parser_candidato,
                "no_servicio"
            ].dropna().astype(str):

                servicio_key = normalize_service_cc(
                    pd.Series([value_servicio])
                ).iloc[0]

                if servicio_key and servicio_key.upper() not in [
                    "",
                    "NAN",
                    "NONE",
                    "<NA>"
                ]:
                    servicios.append(servicio_key)

            servicios = sorted(set(servicios))

            servicios = [
                s for s in servicios
                if servicio_parser_disponible(s)
            ]

            return servicios

        def mask_parser_por_servicios(servicios):
            if not servicios or "no_servicio" not in parser_cc.columns:
                return pd.Series(False, index=parser_cc.index)

            parser_servicio_key = normalize_service_cc(
                parser_cc["no_servicio"]
            )

            return parser_servicio_key.isin(servicios)

        def asignar_rescate_si_unico(idx_local, servicios, criterio):
            if len(servicios) != 1:
                return False

            servicio_unico = servicios[0]

            mask_parser_rescate = mask_parser_por_servicios(
                [servicio_unico]
            )

            if not mask_parser_rescate.any():
                return False

            mask_general_rescate = pd.Series(
                False,
                index=general_ocupados_cc.index
            )

            mask_general_rescate.loc[idx_local] = True

            copiar_parser_match_a_general(
                mask_general=mask_general_rescate,
                mask_parser=mask_parser_rescate,
                criterio=criterio
            )

            return True

        # Textos del parser donde puede aparecer el No. de Local
        parser_text_cols_rescate = [
            col for col in [
                "direccion_completa",
                "direccion_raw",
                "file_path",
                "file_name",
                "recibos_subgroup"
            ]
            if col in parser_cc.columns
        ]

        for idx_rescate, row_rescate in general_ocupados_cc.iterrows():

            # Si ya tiene algo asignado del parser, no lo tocamos.
            ya_tiene_parser = False

            for col_check_rescate in [
                "parser_no_servicio_match",
                "parser_medidor_match",
                "parser_tarifa_match",
                "parser_file_match"
            ]:
                if col_check_rescate not in general_ocupados_cc.columns:
                    continue

                value_check = row_rescate.get(col_check_rescate, "")

                if (
                    pd.notna(value_check)
                    and str(value_check).strip().upper()
                    not in ["", "NAN", "NONE", "<NA>", "SIN TARIFA"]
                ):
                    ya_tiene_parser = True
                    break

            if ya_tiene_parser:
                continue

            # ----------------------------------------------
            # 1) Rescate por No. de Local en file_path/dirección
            # ----------------------------------------------

            if general_local_col and parser_text_cols_rescate:

                no_local_val = row_rescate.get(general_local_col, "")

                variantes_local = generar_variantes_no_local(no_local_val)

                if variantes_local:
                    textos_parser_rescate = pd.Series(
                        "",
                        index=parser_cc.index,
                        dtype="object"
                    )

                    for col_texto_rescate in parser_text_cols_rescate:
                        textos_parser_rescate = (
                            textos_parser_rescate.astype(str)
                            + " "
                            + parser_cc[col_texto_rescate]
                            .fillna("")
                            .astype(str)
                        )

                    textos_parser_key_rescate = textos_parser_rescate.apply(
                        normalizar_texto_para_local
                    )

                    mask_parser_local_rescate = textos_parser_key_rescate.apply(
                        lambda texto: any(
                            variante in texto
                            for variante in variantes_local
                        )
                    )

                    servicios_local_rescate = servicios_unicos_parser(
                        mask_parser_local_rescate
                    )

                    if asignar_rescate_si_unico(
                        idx_local=idx_rescate,
                        servicios=servicios_local_rescate,
                        criterio="Rescate por No. local único en parser"
                    ):
                        continue

            # ----------------------------------------------
            # 2) Rescate por nombre comercial exacto no ambiguo
            # ----------------------------------------------
            # No usamos cliente corporativo.
            # Solo nombre comercial exacto normalizado.

            if (
                "_key_nombre_comercial" in general_ocupados_cc.columns
                and "_key_nombre" in parser_cc.columns
            ):
                nombre_key_rescate = row_rescate.get(
                    "_key_nombre_comercial",
                    ""
                )

                if (
                    pd.notna(nombre_key_rescate)
                    and str(nombre_key_rescate).strip().upper()
                    not in ["", "NAN", "NONE", "<NA>"]
                ):
                    mask_parser_nombre_rescate = (
                        parser_cc["_key_nombre"]
                        .astype(str)
                        .eq(str(nombre_key_rescate))
                    )

                    servicios_nombre_rescate = servicios_unicos_parser(
                        mask_parser_nombre_rescate
                    )

                    if asignar_rescate_si_unico(
                        idx_local=idx_rescate,
                        servicios=servicios_nombre_rescate,
                        criterio="Rescate por nombre comercial exacto único"
                    ):
                        continue

        # Después del rescate, volvemos a limpiar duplicados por si algún
        # no_servicio quedó asignado dos veces accidentalmente.
        servicio_a_indices_post_rescate = {}

        if "parser_no_servicio_match" in general_ocupados_cc.columns:
            for idx_servicio, value_servicio in general_ocupados_cc[
                "parser_no_servicio_match"
            ].dropna().items():

                servicios_idx = split_pipe_normalizado(
                    value_servicio,
                    tipo="servicio"
                )

                for servicio_idx in servicios_idx:
                    servicio_a_indices_post_rescate.setdefault(
                        servicio_idx,
                        []
                    ).append(idx_servicio)

        for servicio_dup, indices_dup in servicio_a_indices_post_rescate.items():

            indices_dup = sorted(set(indices_dup))

            if len(indices_dup) <= 1:
                continue

            candidatos_dup = general_ocupados_cc.loc[
                indices_dup
            ].copy()

            candidatos_dup["_score_match_servicio"] = (
                candidatos_dup.apply(
                    score_match_local,
                    axis=1
                )
            )

            candidatos_dup = candidatos_dup.sort_values(
                ["_score_match_servicio"],
                ascending=False
            )

            idx_ganador = candidatos_dup.index[0]

            indices_perdedores = [
                idx for idx in indices_dup
                if idx != idx_ganador
            ]

            if indices_perdedores:
                limpiar_match_parser_en_indices(
                    indices_perdedores,
                    f"no_servicio {servicio_dup} duplicado después de rescate"
                )

        # --------------------------------------------------
        # Decisión final de muestra global con recibo
        # --------------------------------------------------
        # Objetivo:
        # - No usar solo "_tiene_recibo_confirmado", porque baja demasiado la muestra.
        # - No usar todo "_tiene_recibo" amplio, porque mete falsos positivos.
        # - Usar una muestra intermedia:
        #   1) locales con parser realmente asignado
        #   2) matches fuertes por medidor
        #   3) matches fuertes por no. local / dirección
        #
        # Importante:
        # El match por CLIENTE por sí solo NO se considera suficiente.
        # El match por NOMBRE COMERCIAL sí ayuda, pero solo cuando logró
        # copiar algo concreto del parser.

        def valor_parser_valido(df, col):
            if col not in df.columns:
                return pd.Series(False, index=df.index)

            return (
                df[col].notna()
                & ~df[col]
                .astype(str)
                .str.upper()
                .str.strip()
                .isin([
                    "",
                    "NAN",
                    "NONE",
                    "<NA>",
                    "SIN TARIFA",
                    "SIN TARIFA / SIN DEMANDA ASOCIADA"
                ])
            )

        general_ocupados_cc["_tiene_recibo_original"] = (
            general_ocupados_cc["_tiene_recibo"]
        )

        tiene_parser_asignado = (
            valor_parser_valido(general_ocupados_cc, "parser_no_servicio_match")
            | valor_parser_valido(general_ocupados_cc, "parser_medidor_match")
            | valor_parser_valido(general_ocupados_cc, "parser_tarifa_match")
            | valor_parser_valido(general_ocupados_cc, "parser_file_match")
        )

        match_fuerte_directo = (
            general_ocupados_cc["_match_medidor"].fillna(False)
            | general_ocupados_cc["_match_no_local_direccion"].fillna(False)
        )

        match_nombre_con_parser = (
            general_ocupados_cc["_match_nombre_comercial"].fillna(False)
            & tiene_parser_asignado
        )

        # --------------------------------------------------
        # Regla final conservadora
        # --------------------------------------------------
        # Para evitar contar locales de más:
        # - NO basta con que haya match por cliente.
        # - NO basta con que haya match por nombre comercial.
        # - NO basta con que el no. local aparezca en algún texto.
        # - SÍ debe existir un dato concreto copiado desde el parser.
        #
        # La mejor evidencia es parser_no_servicio_match.
        # Aceptamos también medidor/tarifa/file del parser como respaldo,
        # pero ya no contamos "match fuerte sin parser".

        tiene_no_servicio_parser = valor_parser_valido(
            general_ocupados_cc,
            "parser_no_servicio_match"
        )

        general_ocupados_cc["_tiene_recibo"] = (
            tiene_no_servicio_parser
            | (
                tiene_parser_asignado
                & (
                    general_ocupados_cc["_match_medidor"].fillna(False)
                    | general_ocupados_cc["_match_no_local_direccion"].fillna(False)
                    | general_ocupados_cc["_match_nombre_comercial"].fillna(False)
                )
            )
        )

        # Diagnóstico: los que entraban por match amplio pero no pasan la
        # decisión final quedan como no confirmados.
        general_ocupados_cc["_match_no_confirmado"] = (
            general_ocupados_cc["_tiene_recibo_original"].fillna(False)
            & ~general_ocupados_cc["_tiene_recibo"].fillna(False)
        )

        general_ocupados_cc.loc[
            general_ocupados_cc["_match_no_confirmado"],
            "criterio_match"
        ] = (
            general_ocupados_cc.loc[
                general_ocupados_cc["_match_no_confirmado"],
                "criterio_match"
            ].astype(str)
            + " | NO CONFIRMADO"
        )

        general_ocupados_cc.loc[
            general_ocupados_cc["_tiene_recibo"] & tiene_parser_asignado,
            "criterio_match"
        ] = (
            general_ocupados_cc.loc[
                general_ocupados_cc["_tiene_recibo"] & tiene_parser_asignado,
                "criterio_match"
            ].astype(str)
            + " | PARSER ASIGNADO"
        )

        general_ocupados_cc.loc[
            (
                general_ocupados_cc["_tiene_recibo"]
                & tiene_no_servicio_parser
            ),
            "criterio_match"
        ] = (
            general_ocupados_cc.loc[
                (
                    general_ocupados_cc["_tiene_recibo"]
                    & tiene_no_servicio_parser
                ),
                "criterio_match"
            ].astype(str)
            + " | NO_SERVICIO ASIGNADO"
        )

        # --------------------------------------------------
        # Diagnóstico global de TARIFA_FINAL
        # --------------------------------------------------
        # No lo mostramos aquí porque esta función corre antes de los tabs.
        # Lo guardamos para mostrarlo después dentro de Calidad de Datos.

        debug_sin_tarifa_global = general_ocupados_cc[
            general_ocupados_cc["_match_no_confirmado"]
        ].copy()

        if not debug_sin_tarifa_global.empty:

            debug_sin_tarifa_global["Centro Comercial"] = limpiar_nombre_cc(
                mall_name
            )

            # Columna auxiliar para revisar no_servicio en el debug.
            # Prioriza el no_servicio copiado desde parser, pero también
            # revisa posibles columnas de DG si existieran.

            posibles_cols_no_servicio_debug = [
                "parser_no_servicio_match",
                "no_servicio",
                "No. Servicio",
                "No Servicio",
                "No. de Servicio",
                "No de Servicio",
                "SERVICIO",
                "Servicio"
            ]

            debug_sin_tarifa_global["NO_SERVICIO_DEBUG"] = pd.NA

            for col_servicio_debug in posibles_cols_no_servicio_debug:
                if col_servicio_debug not in debug_sin_tarifa_global.columns:
                    continue

                valores_servicio_debug = (
                    debug_sin_tarifa_global[col_servicio_debug]
                    .fillna("")
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .str.strip()
                )

                faltante_servicio_debug = (
                    debug_sin_tarifa_global["NO_SERVICIO_DEBUG"].isna()
                    | debug_sin_tarifa_global["NO_SERVICIO_DEBUG"]
                    .astype(str)
                    .str.upper()
                    .isin(["", "NAN", "NONE", "<NA>"])
                )

                valido_servicio_debug = (
                    valores_servicio_debug.notna()
                    & ~valores_servicio_debug
                    .astype(str)
                    .str.upper()
                    .isin(["", "NAN", "NONE", "<NA>"])
                )

                debug_sin_tarifa_global.loc[
                    faltante_servicio_debug & valido_servicio_debug,
                    "NO_SERVICIO_DEBUG"
                ] = valores_servicio_debug.loc[
                    faltante_servicio_debug & valido_servicio_debug
                ]

            # --------------------------------------------------
            # Candidatos encontrados en parser
            # --------------------------------------------------
            # Para los locales que hicieron match por cliente/nombre,
            # mostramos qué registros parecidos existen en el parser.
            # Esto ayuda a saber si el dato no existe o si no lo estamos jalando.

            def join_unique_debug_values(serie):
                vals = []

                for v in serie.dropna().astype(str).tolist():
                    v = v.strip()

                    if re.match(r"^\d+\.0$", v):
                        v = v.replace(".0", "")

                    if v.upper() in ["", "NAN", "NONE", "<NA>"]:
                        continue

                    vals.append(v)

                vals = sorted(set(vals))

                return " | ".join(vals) if vals else pd.NA

            for col_debug_parser in [
                "PARSER_CLIENTE_CANDIDATO",
                "PARSER_NOMBRE_COMERCIAL_CANDIDATO",
                "PARSER_NO_SERVICIO_CANDIDATO",
                "PARSER_MEDIDOR_CANDIDATO",
                "PARSER_TARIFA_CANDIDATO",
                "PARSER_FILE_CANDIDATO",
                "NO_SERVICIO_ASIGNADO_A_CLIENTE",
                "NO_SERVICIO_ASIGNADO_A_NOMBRE_COMERCIAL",
                "NO_SERVICIO_ASIGNADO_A_LOCAL"
            ]:
                debug_sin_tarifa_global[col_debug_parser] = pd.NA

            for idx_debug, row_debug in debug_sin_tarifa_global.iterrows():

                mask_parser_candidatos = pd.Series(
                    False,
                    index=parser_cc.index
                )

                cliente_debug = row_debug.get("CLIENTE", "")
                nombre_debug = row_debug.get("NOMBRE COMERCIAL", "")

                # Candidatos por cliente
                if (
                    pd.notna(cliente_debug)
                    and str(cliente_debug).strip() != ""
                    and "_key_cliente" in parser_cc.columns
                ):
                    mask_cliente_debug = parser_cc["_key_cliente"].apply(
                        lambda x: has_partial_match_cc(
                            cliente_debug,
                            [x]
                        )
                    )

                    mask_parser_candidatos = (
                        mask_parser_candidatos | mask_cliente_debug
                    )

                # Candidatos por nombre comercial
                if (
                    pd.notna(nombre_debug)
                    and str(nombre_debug).strip() != ""
                    and "_key_nombre" in parser_cc.columns
                ):
                    mask_nombre_debug = parser_cc["_key_nombre"].apply(
                        lambda x: has_partial_match_cc(
                            nombre_debug,
                            [x]
                        )
                    )

                    mask_parser_candidatos = (
                        mask_parser_candidatos | mask_nombre_debug
                    )

                parser_candidatos = parser_cc.loc[
                    mask_parser_candidatos
                ].copy()

                if parser_candidatos.empty:
                    continue

                if "cliente_nombre" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_CLIENTE_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["cliente_nombre"]
                    )

                if "recibos_subgroup" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_NOMBRE_COMERCIAL_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["recibos_subgroup"]
                    )

                if "no_servicio" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_NO_SERVICIO_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["no_servicio"]
                    )

                # --------------------------------------------------
                # Revisar si el no_servicio candidato ya fue asignado
                # a otro local de Datos Generales
                # --------------------------------------------------
                servicio_candidato_debug = debug_sin_tarifa_global.loc[
                    idx_debug,
                    "PARSER_NO_SERVICIO_CANDIDATO"
                ]

                if (
                    pd.notna(servicio_candidato_debug)
                    and str(servicio_candidato_debug).strip() != ""
                    and "parser_no_servicio_match" in general_ocupados_cc.columns
                ):
                    servicios_debug = [
                        normalize_service_cc(pd.Series([x.strip()])).iloc[0]
                        for x in str(servicio_candidato_debug).split("|")
                        if str(x).strip().upper() not in ["", "NAN", "NONE", "<NA>"]
                    ]

                    parser_servicios_asignados = (
                        general_ocupados_cc["parser_no_servicio_match"]
                        .fillna("")
                        .astype(str)
                    )

                    mask_asignado_debug = parser_servicios_asignados.apply(
                        lambda value: any(
                            servicio in [
                                normalize_service_cc(pd.Series([raw.strip()])).iloc[0]
                                for raw in str(value).split("|")
                                if str(raw).strip().upper() not in ["", "NAN", "NONE", "<NA>"]
                            ]
                            for servicio in servicios_debug
                        )
                    )

                    asignados_debug = general_ocupados_cc.loc[
                        mask_asignado_debug
                    ].copy()

                    if not asignados_debug.empty:
                        debug_sin_tarifa_global.loc[
                            idx_debug,
                            "NO_SERVICIO_ASIGNADO_A_CLIENTE"
                        ] = " | ".join(
                            asignados_debug.get("CLIENTE", pd.Series(dtype=str))
                            .dropna()
                            .astype(str)
                            .unique()
                        )

                        debug_sin_tarifa_global.loc[
                            idx_debug,
                            "NO_SERVICIO_ASIGNADO_A_NOMBRE_COMERCIAL"
                        ] = " | ".join(
                            asignados_debug.get("NOMBRE COMERCIAL", pd.Series(dtype=str))
                            .dropna()
                            .astype(str)
                            .unique()
                        )

                        debug_sin_tarifa_global.loc[
                            idx_debug,
                            "NO_SERVICIO_ASIGNADO_A_LOCAL"
                        ] = " | ".join(
                            asignados_debug.get(general_local_col, pd.Series(dtype=str))
                            .dropna()
                            .astype(str)
                            .unique()
                        )

                if "medidor" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_MEDIDOR_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["medidor"]
                    )

                if "tarifa_norm" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_TARIFA_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["tarifa_norm"]
                    )

                elif "tarifa" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_TARIFA_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["tarifa"]
                    )

                if "file_path" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_FILE_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["file_path"]
                    )

                elif "file_name" in parser_candidatos.columns:
                    debug_sin_tarifa_global.loc[
                        idx_debug,
                        "PARSER_FILE_CANDIDATO"
                    ] = join_unique_debug_values(
                        parser_candidatos["file_name"]
                    )

            cols_debug_global_tarifa = [
                c for c in [
                    "Centro Comercial",
                    "CLIENTE",
                    "NOMBRE COMERCIAL",
                    general_local_col,
                    "NO_SERVICIO_DEBUG",
                    "criterio_match",
                    "parser_tarifa_match",
                    "parser_no_servicio_match",
                    "parser_medidor_match",
                    "parser_file_match",
                    "PARSER_CLIENTE_CANDIDATO",
                    "PARSER_NOMBRE_COMERCIAL_CANDIDATO",
                    "PARSER_NO_SERVICIO_CANDIDATO",
                    "PARSER_MEDIDOR_CANDIDATO",
                    "PARSER_TARIFA_CANDIDATO",
                    "PARSER_FILE_CANDIDATO",
                    "NO_SERVICIO_ASIGNADO_A_CLIENTE",
                    "NO_SERVICIO_ASIGNADO_A_NOMBRE_COMERCIAL",
                    "NO_SERVICIO_ASIGNADO_A_LOCAL",
                    "TARIFA_FINAL"
                ]
                if c in debug_sin_tarifa_global.columns
            ]

            debug_sin_tarifa_rows.append(
                debug_sin_tarifa_global[cols_debug_global_tarifa].copy()
            )

        # --------------------------------------------------
        # Guardar muestra validada para benchmark
        # --------------------------------------------------

        muestra_cc = general_ocupados_cc[
            general_ocupados_cc["_tiene_recibo"]
        ].copy()

        muestra_cc["_centro_comercial_parser"] = mall_name
        muestra_cc["_centro_comercial_limpio"] = limpiar_nombre_cc(mall_name)
        muestra_cc["_match_source"] = muestra_cc["criterio_match"]

        muestra_con_recibo_rows.append(muestra_cc)

        # --------------------------------------------------
        # Resumen de cobertura
        # --------------------------------------------------

        locales_ocupados = len(general_ocupados_cc)
        locales_con_recibo = int(general_ocupados_cc["_tiene_recibo"].sum())
        locales_sin_recibo = int(locales_ocupados - locales_con_recibo)

        cobertura = (
            locales_con_recibo / locales_ocupados * 100
            if locales_ocupados > 0
            else 0
        )

        coverage_rows.append({
            "Centro Comercial": limpiar_nombre_cc(mall_name),
            "Locales ocupados": locales_ocupados,
            "Locales ocupados con recibo": locales_con_recibo,
            "Locales ocupados sin recibo": locales_sin_recibo,
            "Cobertura de muestra (%)": round(cobertura, 1),
        })

        match_summary_rows.append({
            "Centro Comercial": limpiar_nombre_cc(mall_name),
            "Match por medidor": int(general_ocupados_cc["_match_medidor"].sum()),
            "Match por cliente": int(general_ocupados_cc["_match_cliente"].sum()),
            "Match por nombre comercial": int(general_ocupados_cc["_match_nombre_comercial"].sum()),
            "Match por no. local en dirección": int(general_ocupados_cc["_match_no_local_direccion"].sum()),
            "Match total": int(general_ocupados_cc["_tiene_recibo"].sum()),
            "Locales ocupados": len(general_ocupados_cc)
        })

        sin_match_cc = general_ocupados_cc[
            ~general_ocupados_cc["_tiene_recibo"]
        ].copy()

        for _, row in sin_match_cc.iterrows():
            sin_match_rows.append({
                "Centro Comercial": limpiar_nombre_cc(mall_name),
                "Cliente": row.get("CLIENTE", ""),
                "Nombre comercial": row.get("NOMBRE COMERCIAL", ""),
                "No de local": row.get(general_local_col, "") if general_local_col else "",
                "Medidor datos generales": row.get(general_meter_col, "") if general_meter_col else "",
                "Key cliente": row.get("_key_cliente", ""),
                "Key nombre comercial": row.get("_key_nombre_comercial", ""),
                "Key medidor": row.get("_key_medidor", ""),
                "Criterio de match": row.get("criterio_match", "")
            })

    muestra_con_recibo = (
        pd.concat(muestra_con_recibo_rows, ignore_index=True)
        if muestra_con_recibo_rows
        else pd.DataFrame()
    )

    if not muestra_con_recibo.empty:
        dedup_muestra_cols = [
            col for col in [
                "_centro_comercial_limpio",
                "CLIENTE",
                "NOMBRE COMERCIAL",
                "No de Local",
            ]
            if col in muestra_con_recibo.columns
        ]

        if dedup_muestra_cols:
            muestra_con_recibo = muestra_con_recibo.drop_duplicates(
                subset=dedup_muestra_cols,
                keep="first"
            )

    coverage_by_mall = pd.DataFrame(coverage_rows)

    if not coverage_by_mall.empty:
        coverage_by_mall = coverage_by_mall.sort_values(
            "Centro Comercial",
            ascending=True
        ).reset_index(drop=True)

        total_locales = coverage_by_mall["Locales ocupados"].sum()
        total_con_recibo = coverage_by_mall["Locales ocupados con recibo"].sum()
        total_sin_recibo = coverage_by_mall["Locales ocupados sin recibo"].sum()

        total_cobertura = (
            total_con_recibo / total_locales * 100
            if total_locales > 0
            else 0
        )

        total_row = pd.DataFrame([{
            "Centro Comercial": "TOTAL",
            "Locales ocupados": total_locales,
            "Locales ocupados con recibo": total_con_recibo,
            "Locales ocupados sin recibo": total_sin_recibo,
            "Cobertura de muestra (%)": round(total_cobertura, 1)
        }])

        coverage_by_mall = pd.concat(
            [coverage_by_mall, total_row],
            ignore_index=True
        )

    match_summary_df = pd.DataFrame(match_summary_rows)
    sin_match_df = pd.DataFrame(sin_match_rows)
    parser_sin_match_df = pd.DataFrame(parser_sin_match_rows)

    debug_sin_tarifa_df = (
        pd.concat(debug_sin_tarifa_rows, ignore_index=True)
        if debug_sin_tarifa_rows
        else pd.DataFrame()
    )

    return (
        muestra_con_recibo,
        coverage_by_mall,
        match_summary_df,
        sin_match_df,
        parser_sin_match_df,
        debug_sin_tarifa_df
    )



muestra_con_recibo, coverage_by_mall, match_summary_df, sin_match_df, parser_sin_match_df, debug_sin_tarifa_df = construir_muestra_con_recibo_global(
    parsed=parsed,
    general_data=general_data,
    mall_col=mall_col
)

# ============================================================
# Base maestra global DG-parser
# ============================================================
# Esta es la muestra más completa disponible.
# Ningún tab debe recalcular ni sobrescribir esta base.

muestra_con_recibo_global = muestra_con_recibo.copy()

# ------------------------------------------------------------
# Regla PDBT: demanda contratada por default
# ------------------------------------------------------------
# Para locales ocupados con recibo:
# - Si la tarifa es PDBT
# - Y el parser NO trae demanda_contratada_kw
# Entonces se asignan 25 kW.
# Si el parser sí trae demanda_contratada_kw, se conserva.

pdbt_default_25_count = 0

if "demanda_contratada_kw" not in muestra_con_recibo_global.columns:
    muestra_con_recibo_global["demanda_contratada_kw"] = pd.NA

muestra_con_recibo_global["demanda_contratada_kw"] = pd.to_numeric(
    muestra_con_recibo_global["demanda_contratada_kw"],
    errors="coerce"
)

if "TARIFA_FINAL" in muestra_con_recibo_global.columns:
    tarifa_para_demanda = muestra_con_recibo_global["TARIFA_FINAL"]
elif "parser_tarifa_match" in muestra_con_recibo_global.columns:
    tarifa_para_demanda = muestra_con_recibo_global["parser_tarifa_match"]
else:
    tarifa_para_demanda = pd.Series(
        "",
        index=muestra_con_recibo_global.index
    )

mask_pdbt = (
    tarifa_para_demanda
    .fillna("")
    .astype(str)
    .str.upper()
    .str.strip()
    .eq("PDBT")
)

mask_sin_demanda_parser = (
    muestra_con_recibo_global["demanda_contratada_kw"].isna()
)

pdbt_default_25_count = int(
    (mask_pdbt & mask_sin_demanda_parser).sum()
)

muestra_con_recibo_global.loc[
    mask_pdbt & mask_sin_demanda_parser,
    "demanda_contratada_kw"
] = 25

# ============================================================
# Estimación PDBT con NREL y promedio de últimos 12 meses
# ============================================================

parsed = aplicar_estimacion_pdbt_nrel_a_parsed(
    parsed=parsed,
    muestra_con_recibo=muestra_con_recibo_global,
    data_dir=DATA_DIR
)

parsed = calcular_demanda_promedio_ultimos_12_meses(parsed)

tab_resumen, tab_general, tab_cc, tab_sg, tab_calidad, tab_anexo = st.tabs([
    "Resumen Ejecutivo",
    "Portafolio",
    "Por Centro Comercial",
    "Servicios Generales",
    "Calidad de Datos",
    "Anexo"
])

def obtener_tipo_perfil_nrel(subgiro_comercial, tipo_local):
    subgiro = normalizar_texto_simple(subgiro_comercial)
    tipo = normalizar_texto_simple(tipo_local)

    if "ALIMENTOS" in subgiro or "BEBIDAS" in subgiro:
        if "FOOD COURT" in tipo:
            return "quickservicerestaurant", "Comida rápida (Quick Service Restaurant)"
        else:
            return "fullservicerestaurant", "Restaurante (Full Service Restaurant)"

    if "TIENDAS DEPARTAMENTALES" in subgiro:
        return "retailstandalone", "Tienda departamental (Retail Standalone)"

    return "retailstripmall", "Local comercial (Retail Strip Mall)"


def obtener_zona_nrel_por_cc(centro_comercial, climate_mapping_df):
    centro_key = normalizar_texto_simple(centro_comercial)

    if climate_mapping_df.empty:
        return None

    for _, row in climate_mapping_df.iterrows():
        cc_map_key = normalizar_texto_simple(row.get("centro_comercial", ""))

        if cc_map_key and (cc_map_key in centro_key or centro_key in cc_map_key):
            return row.get("zona_nrel")

    return None

def crear_demanda_real_anual_template(muestra_con_recibo, climate_mapping_df):
    rows = []

    for _, row in muestra_con_recibo.iterrows():

        centro_comercial = row.get("NOMBRE DEL CC", row.get("mall_folder", ""))
        subgiro = row.get("SUBGIRO_COMERCIAL", "")
        tipo_local = row.get("TIPO LOCAL", "")
        tarifa = row.get("TARIFA_ANALISIS", row.get("TARIFA", ""))

        zona_nrel = obtener_zona_nrel_por_cc(
            centro_comercial,
            climate_mapping_df
        )

        perfil_code, perfil_nombre = obtener_tipo_perfil_nrel(
            subgiro,
            tipo_local
        )

        rows.append({
            "centro_comercial": centro_comercial,
            "cliente": row.get("CLIENTE", ""),
            "nombre_comercial": row.get("NOMBRE COMERCIAL", ""),
            "no_local": row.get("No de Local", ""),
            "subgiro_comercial": subgiro,
            "tipo_local": tipo_local,
            "tarifa": tarifa,
            "zona_nrel": zona_nrel,
            "perfil_nrel_code": perfil_code,
            "perfil_nrel_nombre": perfil_nombre,
            "demanda_real_anual_kw": pd.NA,
            "criterio_demanda": pd.NA
        })
    return pd.DataFrame(rows)

def enriquecer_muestra_con_demanda(muestra_con_recibo: pd.DataFrame, parsed: pd.DataFrame) -> pd.DataFrame:
    """
    Toma la muestra validada de Calidad de Datos y le agrega la demanda promedio anual
    calculada desde el parser.

    Importante:
    Esta función NO vuelve a definir qué locales tienen recibo.
    Solo usa la muestra ya validada y le busca su demanda correspondiente.
    """

    if muestra_con_recibo.empty or parsed.empty:
        return pd.DataFrame()

    df = muestra_con_recibo.copy()
    # --------------------------------------------------
    # 0) Usar el match original guardado desde Calidad de Datos
    # --------------------------------------------------
    # Si muestra_con_recibo ya trae datos del parser original,
    # los usamos antes de cualquier nuevo intento de cruce.

    if "parser_no_servicio_match" in df.columns:
        df["no_servicio"] = df["parser_no_servicio_match"]

    if "parser_medidor_match" in df.columns:
        df["medidor"] = df["parser_medidor_match"]

    if "parser_tarifa_match" in df.columns:
        df["tarifa_norm"] = df["parser_tarifa_match"]

    if "parser_criterio_match" in df.columns:
        df["criterio_union_demanda"] = df["parser_criterio_match"]

    if "parser_medidor_match" in df.columns:
        df["_key_medidor"] = normalize_meter_cc(
            df["parser_medidor_match"]
        )

    p = parsed.copy()

    # --------------------------------------------------
    # Llave de centro comercial
    # --------------------------------------------------

    if "_centro_comercial_limpio" in df.columns:
        df["_cc_key"] = df["_centro_comercial_limpio"].apply(cc_key)

    elif "NOMBRE DEL CC" in df.columns:
        df["_cc_key"] = df["NOMBRE DEL CC"].apply(cc_key)

    elif "CENTRO COMERCIAL" in df.columns:
        df["_cc_key"] = df["CENTRO COMERCIAL"].apply(cc_key)

    elif "source_sheet" in df.columns:
        df["_cc_key"] = df["source_sheet"].apply(cc_key)

    else:
        df["_cc_key"] = ""

    if "mall_folder" in p.columns:
        p["_cc_key"] = p["mall_folder"].apply(cc_key)
    else:
        p["_cc_key"] = ""

    # --------------------------------------------------
    # Llave de medidor
    # --------------------------------------------------

    general_meter_col = first_existing_column(
        df,
        [
            "No. De medidor",
            "No. de medidor",
            "No de medidor",
            "No. Medidor",
            "No Medidor",
            "No. medidor",
            "MEDIDOR",
            "Medidor",
            "medidor"
        ]
    )

    if "_key_medidor" not in df.columns:
        df["_key_medidor"] = (
            normalize_meter_cc(df[general_meter_col])
            if general_meter_col
            else ""
        )

    p["_key_medidor"] = (
        normalize_meter_cc(p["medidor"])
        if "medidor" in p.columns
        else ""
    )

    # --------------------------------------------------
    # Llaves de cliente y nombre comercial
    # --------------------------------------------------

    if "_key_cliente" not in df.columns:
        df["_key_cliente"] = (
            normalize_cc_key(df["CLIENTE"])
            if "CLIENTE" in df.columns
            else ""
        )

    if "_key_nombre_comercial" not in df.columns:
        df["_key_nombre_comercial"] = (
            normalize_cc_key(df["NOMBRE COMERCIAL"])
            if "NOMBRE COMERCIAL" in df.columns
            else ""
        )

    p["_key_cliente"] = (
        normalize_cc_key(p["cliente_nombre"])
        if "cliente_nombre" in p.columns
        else ""
    )

    p["_key_nombre"] = (
        normalize_cc_key(p["recibos_subgroup"])
        if "recibos_subgroup" in p.columns
        else ""
    )

    # --------------------------------------------------
    # Tabla lookup de demanda desde parser
    # --------------------------------------------------

    demanda_cols = [
        col for col in [
            "_cc_key",
            "_key_medidor",
            "_key_cliente",
            "_key_nombre",
            "mall_folder",
            "cliente_nombre",
            "recibos_subgroup",
            "no_servicio",
            "medidor",
            "tarifa_norm",
            "demanda_promedio_anual_kw",
            "demanda_maxima_anual_kw",
            "meses_con_demanda",
            "periodo_12m_inicio",
            "periodo_12m_fin",
            "kwh_12m",
            "zona_nrel",
            "perfil_nrel_code",
            "perfil_nrel_nombre",
            "profile_path_nrel",
            "pdbt_nrel_status",
            "centro_comercial_nrel_input",
            "subgiro_nrel_input",
            "tipo_local_nrel_input",
            "criterio_demanda_mensual",
            "direccion_completa",
            "direccion_raw",
            "file_path",
            "file_name"
        ]
        if col in p.columns
    ]

    demanda_lookup = p[demanda_cols].copy()

    # --------------------------------------------------
    # Lookup preferente por no_servicio
    # --------------------------------------------------
    # Esta es la unión correcta cuando un mismo servicio tuvo varios medidores.
    # Agrega todos los medidores y conserva la demanda/kWh anual ya calculada
    # a nivel servicio.

    if "no_servicio" in p.columns:
        p["_key_no_servicio"] = normalize_service_cc(p["no_servicio"])
    else:
        p["_key_no_servicio"] = ""

    def split_service_keys_for_match(value):
        """
        Convierte valores como:
        '671221152141 | 673201100237'
        en:
        ['671221152141', '673201100237']
        """

        if pd.isna(value):
            return []

        raw_values = str(value).split("|")

        keys = []

        for raw in raw_values:
            raw = raw.strip()

            if raw.upper() in ["", "NAN", "NONE", "<NA>"]:
                continue

            key = normalize_service_cc(pd.Series([raw])).iloc[0]

            if key and key.upper() not in ["", "NAN", "NONE", "<NA>"]:
                keys.append(key)

        return sorted(set(keys))

    if "parser_no_servicio_match" in df.columns:
        df["_key_no_servicio_match_list"] = (
            df["parser_no_servicio_match"]
            .apply(split_service_keys_for_match)
        )

    elif "no_servicio" in df.columns:
        df["_key_no_servicio_match_list"] = (
            df["no_servicio"]
            .apply(split_service_keys_for_match)
        )

    else:
        df["_key_no_servicio_match_list"] = [[] for _ in range(len(df))]

    df["_row_id_servicio_match"] = df.index

    if "_key_no_servicio" not in demanda_lookup.columns:
        demanda_lookup["_key_no_servicio"] = (
            p["_key_no_servicio"].values
            if len(p) == len(demanda_lookup)
            else ""
        )

    def join_unique_values(serie, upper=False):
        vals = []

        for v in serie.dropna().astype(str).tolist():
            v = v.strip()

            if re.match(r"^\d+\.0$", v):
                v = v.replace(".0", "")

            if upper:
                v = v.upper()

            if v in ["", "nan", "None", "NONE", "NAN", "<NA>"]:
                continue

            vals.append(v)

        vals = sorted(set(vals))

        return " | ".join(vals) if vals else pd.NA

    if (
        "_key_no_servicio_match_list" in df.columns
        and "_key_no_servicio" in demanda_lookup.columns
    ):
        demanda_lookup_servicio = demanda_lookup[
            demanda_lookup["_key_no_servicio"].astype(str).str.strip() != ""
        ].copy()

        if not demanda_lookup_servicio.empty:
            agg_dict_servicio = {}

            if "no_servicio" in demanda_lookup_servicio.columns:
                agg_dict_servicio["no_servicio"] = (
                    "no_servicio",
                    lambda x: join_unique_values(x)
                )

            if "_key_medidor" in demanda_lookup_servicio.columns:
                agg_dict_servicio["_key_medidor_parser_servicio"] = (
                    "_key_medidor",
                    lambda x: join_unique_values(x, upper=True)
                )

            if "tarifa_norm" in demanda_lookup_servicio.columns:
                agg_dict_servicio["tarifa_norm"] = (
                    "tarifa_norm",
                    lambda x: join_unique_values(x, upper=True)
                )

            if "demanda_promedio_anual_kw" in demanda_lookup_servicio.columns:
                agg_dict_servicio["demanda_promedio_anual_kw"] = (
                    "demanda_promedio_anual_kw",
                    "max"
                )

            if "demanda_maxima_anual_kw" in demanda_lookup_servicio.columns:
                agg_dict_servicio["demanda_maxima_anual_kw"] = (
                    "demanda_maxima_anual_kw",
                    "max"
                )

            if "meses_con_demanda" in demanda_lookup_servicio.columns:
                agg_dict_servicio["meses_con_demanda"] = (
                    "meses_con_demanda",
                    "max"
                )

            if "periodo_12m_inicio" in demanda_lookup_servicio.columns:
                agg_dict_servicio["periodo_12m_inicio"] = (
                    "periodo_12m_inicio",
                    "min"
                )

            if "periodo_12m_fin" in demanda_lookup_servicio.columns:
                agg_dict_servicio["periodo_12m_fin"] = (
                    "periodo_12m_fin",
                    "max"
                )

            if "kwh_12m" in demanda_lookup_servicio.columns:
                agg_dict_servicio["kwh_12m"] = (
                    "kwh_12m",
                    "max"
                )

            if "zona_nrel" in demanda_lookup_servicio.columns:
                agg_dict_servicio["zona_nrel"] = (
                    "zona_nrel",
                    lambda x: join_unique_values(x)
                )

            if "perfil_nrel_code" in demanda_lookup_servicio.columns:
                agg_dict_servicio["perfil_nrel_code"] = (
                    "perfil_nrel_code",
                    lambda x: join_unique_values(x)
                )

            if "perfil_nrel_nombre" in demanda_lookup_servicio.columns:
                agg_dict_servicio["perfil_nrel_nombre"] = (
                    "perfil_nrel_nombre",
                    lambda x: join_unique_values(x)
                )

            if "profile_path_nrel" in demanda_lookup_servicio.columns:
                agg_dict_servicio["profile_path_nrel"] = (
                    "profile_path_nrel",
                    lambda x: join_unique_values(x)
                )

            if "pdbt_nrel_status" in demanda_lookup_servicio.columns:
                agg_dict_servicio["pdbt_nrel_status"] = (
                    "pdbt_nrel_status",
                    lambda x: join_unique_values(x)
                )

            demanda_lookup_servicio = (
                demanda_lookup_servicio
                .groupby(["_cc_key", "_key_no_servicio"], dropna=False)
                .agg(**agg_dict_servicio)
                .reset_index()
            )

            # --------------------------------------------------
            # Expandir servicios múltiples del match global
            # --------------------------------------------------
            # Si un local tiene:
            # 671221152141 | 673201100237
            # lo convertimos en dos filas temporales para poder unir
            # contra ambos servicios del parser.

            df_servicio_keys = (
                df[
                    [
                        "_row_id_servicio_match",
                        "_cc_key",
                        "_key_no_servicio_match_list"
                    ]
                ]
                .explode("_key_no_servicio_match_list")
                .rename(
                    columns={
                        "_key_no_servicio_match_list": "_key_no_servicio_match"
                    }
                )
            )

            df_servicio_keys = df_servicio_keys[
                df_servicio_keys["_key_no_servicio_match"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
            ].copy()

            if not df_servicio_keys.empty:

                df_servicio_match = df_servicio_keys.merge(
                    demanda_lookup_servicio,
                    left_on=[
                        "_cc_key",
                        "_key_no_servicio_match"
                    ],
                    right_on=[
                        "_cc_key",
                        "_key_no_servicio"
                    ],
                    how="left"
                )

                agg_back = {}

                for col in [
                    "no_servicio",
                    "medidor",
                    "_key_medidor_parser_servicio",
                    "tarifa_norm",
                    "zona_nrel",
                    "perfil_nrel_code",
                    "perfil_nrel_nombre",
                    "profile_path_nrel",
                    "pdbt_nrel_status",
                    "centro_comercial_nrel_input",
                    "subgiro_nrel_input",
                    "tipo_local_nrel_input",
                    "criterio_demanda_mensual"
                ]:
                    if col in df_servicio_match.columns:
                        agg_back[col] = (
                            col,
                            lambda x: join_unique_values(x)
                        )

                for col in [
                    "demanda_promedio_anual_kw",
                    "demanda_maxima_anual_kw",
                    "meses_con_demanda",
                    "kwh_12m"
                ]:
                    if col in df_servicio_match.columns:
                        agg_back[col] = (
                            col,
                            "max"
                        )

                if "periodo_12m_inicio" in df_servicio_match.columns:
                    agg_back["periodo_12m_inicio"] = (
                        "periodo_12m_inicio",
                        "min"
                    )

                if "periodo_12m_fin" in df_servicio_match.columns:
                    agg_back["periodo_12m_fin"] = (
                        "periodo_12m_fin",
                        "max"
                    )

                if agg_back:
                    df_servicio_match_agg = (
                        df_servicio_match
                        .groupby("_row_id_servicio_match", dropna=False)
                        .agg(**agg_back)
                        .reset_index()
                    )

                    df = df.merge(
                        df_servicio_match_agg,
                        on="_row_id_servicio_match",
                        how="left",
                        suffixes=("", "_servicio")
                    )

                    # --------------------------------------------------
                    # Validación de conflicto entre servicio y medidor DG
                    # --------------------------------------------------
                    # Si el match por no_servicio trae un medidor distinto
                    # al medidor de DG, no copiamos datos de ese servicio.
                    # Esto evita casos como Smart Fit tomando el servicio
                    # de Top Jump.
                    if "_key_medidor_parser_servicio_servicio" in df.columns:
                        medidores_servicio_match = (
                            df["_key_medidor_parser_servicio_servicio"]
                            .fillna("")
                            .astype(str)
                            .str.upper()
                            .str.split("|")
                            .apply(
                                lambda vals: {
                                    v.strip()
                                    for v in vals
                                    if str(v).strip()
                                    and str(v).strip().upper()
                                    not in ["", "NAN", "NONE", "<NA>"]
                                }
                            )
                        )
                    else:
                        medidores_servicio_match = pd.Series(
                            [set() for _ in range(len(df))],
                            index=df.index
                        )

                    if "_key_medidor_dg_validacion" not in df.columns:
                        if "_key_medidor_dg_refuerzo" in df.columns:
                            df["_key_medidor_dg_validacion"] = (
                                df["_key_medidor_dg_refuerzo"]
                                .fillna("")
                                .astype(str)
                                .str.upper()
                                .str.strip()
                            )
                        else:
                            df["_key_medidor_dg_validacion"] = ""

                    medidor_dg_validacion = (
                        df["_key_medidor_dg_validacion"]
                        .fillna("")
                        .astype(str)
                        .str.upper()
                        .str.strip()
                    )

                    df["_servicio_match_valido_por_medidor"] = [
                        (
                            # Si DG no tiene medidor, no bloqueamos el match.
                            medidor_dg in ["", "NAN", "NONE", "<NA>"]
                            or
                            # Si el servicio no trae medidor parser, tampoco bloqueamos.
                            len(medidores_parser) == 0
                            or
                            # Si DG sí tiene medidor, debe estar dentro de los
                            # medidores asociados al no_servicio del parser.
                            medidor_dg in medidores_parser
                        )
                        for medidor_dg, medidores_parser
                        in zip(
                            medidor_dg_validacion,
                            medidores_servicio_match
                        )
                    ]

                    for col in [
                        "no_servicio",
                        "medidor",
                        "tarifa_norm",
                        "demanda_promedio_anual_kw",
                        "demanda_maxima_anual_kw",
                        "meses_con_demanda",
                        "periodo_12m_inicio",
                        "periodo_12m_fin",
                        "kwh_12m",
                        "zona_nrel",
                        "perfil_nrel_code",
                        "perfil_nrel_nombre",
                        "profile_path_nrel",
                        "pdbt_nrel_status",
                        "centro_comercial_nrel_input",
                        "subgiro_nrel_input",
                        "tipo_local_nrel_input",
                        "criterio_demanda_mensual"
                    ]:
                        col_servicio = col + "_servicio"

                        if col_servicio not in df.columns:
                            continue

                        if col not in df.columns:
                            df[col] = pd.NA

                        faltante = (
                            df[col].isna()
                            | df[col].astype(str).str.upper().isin(
                                ["", "NAN", "NONE", "<NA>", "SIN TARIFA"]
                            )
                        )

                        valido_servicio = (
                            df[col_servicio].notna()
                            & ~df[col_servicio].astype(str).str.upper().isin(
                                ["", "NAN", "NONE", "<NA>", "SIN TARIFA"]
                            )
                        )

                        mask_copiar_servicio = (
                            faltante
                            & valido_servicio
                            & df["_servicio_match_valido_por_medidor"]
                        )

                        df.loc[
                            mask_copiar_servicio,
                            col
                        ] = df.loc[
                            mask_copiar_servicio,
                            col_servicio
                        ]

                    if "criterio_union_demanda" not in df.columns:
                        df["criterio_union_demanda"] = pd.NA

                    mask_servicio_match = (
                        (
                            df["demanda_maxima_anual_kw"].notna()
                            | df["tarifa_norm"].notna()
                            | df["profile_path_nrel"].notna()
                        )
                        & df["_servicio_match_valido_por_medidor"]
                        & (
                            df["criterio_union_demanda"].isna()
                            | df["criterio_union_demanda"].astype(str).str.upper().isin(
                                ["", "NAN", "NONE", "<NA>"]
                            )
                        )
                    )

                    df.loc[
                        mask_servicio_match,
                        "criterio_union_demanda"
                    ] = "No. servicio"

    # Evitar repetir meses del mismo servicio.
    dedup_cols = [
        col for col in [
            "_cc_key",
            "_key_medidor",
            "_key_cliente",
            "_key_nombre",
            "no_servicio",
            "tarifa_norm",
            "anio"
        ]
        if col in demanda_lookup.columns
    ]

    if dedup_cols:
        demanda_lookup = demanda_lookup.drop_duplicates(
            subset=dedup_cols,
            keep="first"
        )

    # --------------------------------------------------
    # 1) Match preferente por medidor
    # --------------------------------------------------

    lookup_medidor = demanda_lookup[
        demanda_lookup["_key_medidor"].astype(str).str.strip() != ""
    ].copy()

    lookup_medidor = lookup_medidor.drop_duplicates(
        subset=["_cc_key", "_key_medidor"],
        keep="first"
    )

    cols_lookup_medidor = [
        col for col in [
            "_cc_key",
            "_key_medidor",
            "no_servicio",
            "medidor",
            "tarifa_norm",
            "demanda_promedio_anual_kw",
            "demanda_maxima_anual_kw",
            "meses_con_demanda",
            "periodo_12m_inicio",
            "periodo_12m_fin",
            "kwh_12m",
            "zona_nrel",
            "perfil_nrel_code",
            "perfil_nrel_nombre",
            "pdbt_nrel_status"
        ]
        if col in lookup_medidor.columns
    ]

    df = df.merge(
        lookup_medidor[cols_lookup_medidor],
        on=["_cc_key", "_key_medidor"],
        how="left",
        suffixes=("", "_parser")
    )

    df["criterio_union_demanda"] = pd.NA

    df.loc[
        df["demanda_promedio_anual_kw"].notna(),
        "criterio_union_demanda"
    ] = "Medidor"

    # --------------------------------------------------
    # 1B) Refuerzo por medidor directo
    # --------------------------------------------------
    # Este bloque asegura que si Datos Generales y parser tienen el mismo
    # medidor, se copie la información del parser aunque todavía no exista
    # demanda calculada.

    general_meter_col_refuerzo = first_existing_column(
        df,
        [
            "No. De medidor",
            "No. de medidor",
            "No de medidor",
            "No. Medidor",
            "No Medidor",
            "No. medidor",
            "MEDIDOR",
            "Medidor",
            "medidor"
        ]
    )

    if general_meter_col_refuerzo:
        df["_key_medidor_dg_refuerzo"] = normalize_meter_cc(
            df[general_meter_col_refuerzo]
        )
    else:
        df["_key_medidor_dg_refuerzo"] = ""

    if "_key_medidor" not in df.columns:
        df["_key_medidor"] = ""

    df["_key_medidor"] = df["_key_medidor"].fillna("").astype(str)

    # Medidor de Datos Generales como referencia de validación.
    # No tiene prioridad sobre el parser enriquecido, solo se usa para
    # evitar matches cruzados cuando el no_servicio viene mal asignado.
    df["_key_medidor_dg_validacion"] = (
        df["_key_medidor_dg_refuerzo"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    mask_key_medidor_vacia = (
        df["_key_medidor"].str.upper().isin(
            ["", "NAN", "NONE", "<NA>"]
        )
    )

    df.loc[
        mask_key_medidor_vacia,
        "_key_medidor"
    ] = df.loc[
        mask_key_medidor_vacia,
        "_key_medidor_dg_refuerzo"
    ]

    if "_key_medidor" in demanda_lookup.columns:
        demanda_medidor_refuerzo = demanda_lookup.copy()

        demanda_medidor_refuerzo["_key_medidor"] = (
            demanda_medidor_refuerzo["_key_medidor"]
            .fillna("")
            .astype(str)
        )

        demanda_medidor_refuerzo = demanda_medidor_refuerzo[
            ~demanda_medidor_refuerzo["_key_medidor"]
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
        ].copy()

        demanda_medidor_refuerzo = demanda_medidor_refuerzo[
            ~demanda_medidor_refuerzo["_cc_key"]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
        ].copy()

        cols_a_copiar_por_medidor = [
            col for col in [
                "no_servicio",
                "medidor",
                "tarifa_norm",
                "demanda_promedio_anual_kw",
                "demanda_maxima_anual_kw",
                "meses_con_demanda",
                "periodo_12m_inicio",
                "periodo_12m_fin",
                "kwh_12m",
                "zona_nrel",
                "perfil_nrel_code",
                "perfil_nrel_nombre",
                "profile_path_nrel",
                "pdbt_nrel_status",
                "centro_comercial_nrel_input",
                "subgiro_nrel_input",
                "tipo_local_nrel_input"
            ]
            if col in demanda_medidor_refuerzo.columns
        ]

        demanda_medidor_refuerzo["_tiene_demanda_refuerzo"] = (
            demanda_medidor_refuerzo["demanda_maxima_anual_kw"].notna()
            if "demanda_maxima_anual_kw" in demanda_medidor_refuerzo.columns
            else False
        )

        demanda_medidor_refuerzo["_tiene_tarifa_refuerzo"] = (
            demanda_medidor_refuerzo["tarifa_norm"].notna()
            & ~demanda_medidor_refuerzo["tarifa_norm"]
            .astype(str)
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
            if "tarifa_norm" in demanda_medidor_refuerzo.columns
            else False
        )

        demanda_medidor_refuerzo["_tiene_kwh_refuerzo"] = (
            demanda_medidor_refuerzo["kwh_12m"].notna()
            if "kwh_12m" in demanda_medidor_refuerzo.columns
            else False
        )

        demanda_medidor_refuerzo = demanda_medidor_refuerzo.sort_values(
            [
                "_tiene_demanda_refuerzo",
                "_tiene_tarifa_refuerzo",
                "_tiene_kwh_refuerzo"
            ],
            ascending=[
                False,
                False,
                False
            ]
        )

        demanda_medidor_refuerzo = demanda_medidor_refuerzo.drop_duplicates(
            subset=[
                "_cc_key",
                "_key_medidor"
            ],
            keep="first"
        )

        cols_merge_medidor_refuerzo = [
            "_cc_key",
            "_key_medidor"
        ] + cols_a_copiar_por_medidor

        df = df.merge(
            demanda_medidor_refuerzo[cols_merge_medidor_refuerzo],
            on=[
                "_cc_key",
                "_key_medidor"
            ],
            how="left",
            suffixes=(
                "",
                "_medidor_refuerzo"
            )
        )

        if "criterio_union_demanda" not in df.columns:
            df["criterio_union_demanda"] = pd.NA

        encontro_match_medidor = pd.Series(
            False,
            index=df.index
        )

        for col in cols_a_copiar_por_medidor:
            col_ref = col + "_medidor_refuerzo"

            if col_ref not in df.columns:
                continue

            if col not in df.columns:
                df[col] = pd.NA

            ref_tiene_valor = (
                df[col_ref].notna()
                & ~df[col_ref]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
            )

            if col in ["tarifa_norm", "medidor", "no_servicio"]:
                encontro_match_medidor = encontro_match_medidor | ref_tiene_valor

            faltantes_col = (
                df[col].isna()
                | df[col]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
            )

            mask_copiar = faltantes_col & ref_tiene_valor

            df.loc[
                mask_copiar,
                col
            ] = df.loc[
                mask_copiar,
                col_ref
            ]

            df = df.drop(columns=[col_ref])

        mask_union_medidor_refuerzo = (
            encontro_match_medidor
            & (
                df["criterio_union_demanda"].isna()
                | df["criterio_union_demanda"]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>"])
            )
        )

        df.loc[
            mask_union_medidor_refuerzo,
            "criterio_union_demanda"
        ] = "Medidor"

    # --------------------------------------------------
    # 1C) Último fallback por medidor único en parser
    # --------------------------------------------------
    # Si no se logró copiar tarifa/medidor usando centro comercial + medidor,
    # intentamos por medidor solamente, PERO solo cuando el medidor es único
    # en el parser. Esto resuelve casos donde el nombre del CC no cruza igual,
    # pero el medidor es inequívoco.

    if "_key_medidor" in df.columns and "_key_medidor" in demanda_lookup.columns:

        demanda_medidor_unico = demanda_lookup.copy()

        demanda_medidor_unico["_key_medidor"] = (
            demanda_medidor_unico["_key_medidor"]
            .fillna("")
            .astype(str)
        )

        demanda_medidor_unico = demanda_medidor_unico[
            ~demanda_medidor_unico["_key_medidor"]
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
        ].copy()

        conteo_medidor_parser = (
            demanda_medidor_unico
            .groupby("_key_medidor")
            .size()
            .reset_index(name="_conteo_medidor_parser")
        )

        medidores_unicos_parser = conteo_medidor_parser[
            conteo_medidor_parser["_conteo_medidor_parser"] == 1
        ]["_key_medidor"]

        demanda_medidor_unico = demanda_medidor_unico[
            demanda_medidor_unico["_key_medidor"].isin(
                medidores_unicos_parser
            )
        ].copy()

        cols_a_copiar_medidor_unico = [
            col for col in [
                "no_servicio",
                "medidor",
                "tarifa_norm",
                "demanda_promedio_anual_kw",
                "demanda_maxima_anual_kw",
                "meses_con_demanda",
                "periodo_12m_inicio",
                "periodo_12m_fin",
                "kwh_12m",
                "zona_nrel",
                "perfil_nrel_code",
                "perfil_nrel_nombre",
                "profile_path_nrel",
                "pdbt_nrel_status",
                "centro_comercial_nrel_input",
                "subgiro_nrel_input",
                "tipo_local_nrel_input"
            ]
            if col in demanda_medidor_unico.columns
        ]

        cols_merge_medidor_unico = [
            "_key_medidor"
        ] + cols_a_copiar_medidor_unico

        df = df.merge(
            demanda_medidor_unico[cols_merge_medidor_unico],
            on="_key_medidor",
            how="left",
            suffixes=(
                "",
                "_medidor_unico"
            )
        )

        encontro_match_medidor_unico = pd.Series(
            False,
            index=df.index
        )

        for col in cols_a_copiar_medidor_unico:
            col_unico = col + "_medidor_unico"

            if col_unico not in df.columns:
                continue

            if col not in df.columns:
                df[col] = pd.NA

            valor_unico_valido = (
                df[col_unico].notna()
                & ~df[col_unico]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
            )

            if col in ["tarifa_norm", "medidor", "no_servicio"]:
                encontro_match_medidor_unico = (
                    encontro_match_medidor_unico
                    | valor_unico_valido
                )

            valor_actual_faltante = (
                df[col].isna()
                | df[col]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
            )

            mask_copiar_unico = (
                valor_actual_faltante
                & valor_unico_valido
            )

            df.loc[
                mask_copiar_unico,
                col
            ] = df.loc[
                mask_copiar_unico,
                col_unico
            ]

            df = df.drop(columns=[col_unico])

        if "criterio_union_demanda" not in df.columns:
            df["criterio_union_demanda"] = pd.NA

        criterio_faltante_o_falso = (
            df["criterio_union_demanda"].isna()
            | df["criterio_union_demanda"]
            .astype(str)
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
            | (
                df["tarifa_norm"].isna()
                | df["tarifa_norm"]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>", "SIN TARIFA"])
            )
        )

        mask_union_medidor_unico = (
            encontro_match_medidor_unico
            & criterio_faltante_o_falso
        )

        df.loc[
            mask_union_medidor_unico,
            "criterio_union_demanda"
        ] = "Medidor único parser"

    # --------------------------------------------------
    # 2) Fallback por cliente
    # --------------------------------------------------

    faltantes = df["demanda_promedio_anual_kw"].isna()

    if faltantes.any():
        lookup_cliente = demanda_lookup[
            demanda_lookup["_key_cliente"].astype(str).str.strip() != ""
        ].copy()

        lookup_cliente = lookup_cliente.drop_duplicates(
            subset=["_cc_key", "_key_cliente"],
            keep="first"
        )

        cols_lookup_cliente = [
            col for col in [
                "_cc_key",
                "_key_cliente",
                "no_servicio",
                "medidor",
                "tarifa_norm",
                "demanda_promedio_anual_kw",
                "demanda_maxima_anual_kw",
                "meses_con_demanda",
                "periodo_12m_inicio",
                "periodo_12m_fin",
                "kwh_12m",
                "zona_nrel",
                "perfil_nrel_code",
                "perfil_nrel_nombre",
                "pdbt_nrel_status"
            ]
            if col in lookup_cliente.columns
        ]

        temp = df.loc[faltantes].merge(
            lookup_cliente[cols_lookup_cliente],
            on=["_cc_key", "_key_cliente"],
            how="left",
            suffixes=("", "_cliente")
        )

        for col in [
            "no_servicio",
            "medidor",
            "tarifa_norm",
            "demanda_promedio_anual_kw",
            "demanda_maxima_anual_kw",
            "meses_con_demanda",
            "periodo_12m_inicio",
            "periodo_12m_fin",
            "kwh_12m",
            "zona_nrel",
            "perfil_nrel_code",
            "perfil_nrel_nombre",
            "pdbt_nrel_status"
        ]:
            col_cliente = col + "_cliente"

            if col_cliente in temp.columns:
                df.loc[faltantes, col] = temp[col_cliente].values

        df.loc[
            faltantes & df["demanda_promedio_anual_kw"].notna(),
            "criterio_union_demanda"
        ] = "Cliente"

    # --------------------------------------------------
    # 3) Fallback por nombre comercial
    # --------------------------------------------------

    faltantes = df["demanda_promedio_anual_kw"].isna()

    if faltantes.any():
        lookup_nombre = demanda_lookup[
            demanda_lookup["_key_nombre"].astype(str).str.strip() != ""
        ].copy()

        lookup_nombre = lookup_nombre.drop_duplicates(
            subset=["_cc_key", "_key_nombre"],
            keep="first"
        )

        cols_lookup_nombre = [
            col for col in [
                "_cc_key",
                "_key_nombre",
                "no_servicio",
                "medidor",
                "tarifa_norm",
                "demanda_promedio_anual_kw",
                "demanda_maxima_anual_kw",
                "meses_con_demanda",
                "periodo_12m_inicio",
                "periodo_12m_fin",
                "kwh_12m",
                "zona_nrel",
                "perfil_nrel_code",
                "perfil_nrel_nombre",
                "pdbt_nrel_status"
            ]
            if col in lookup_nombre.columns
        ]

        temp = df.loc[faltantes].merge(
            lookup_nombre[cols_lookup_nombre],
            left_on=["_cc_key", "_key_nombre_comercial"],
            right_on=["_cc_key", "_key_nombre"],
            how="left",
            suffixes=("", "_nombre")
        )

        for col in [
            "no_servicio",
            "medidor",
            "tarifa_norm",
            "demanda_promedio_anual_kw",
            "demanda_maxima_anual_kw",
            "meses_con_demanda",
            "periodo_12m_inicio",
            "periodo_12m_fin",
            "kwh_12m",
            "zona_nrel",
            "perfil_nrel_code",
            "perfil_nrel_nombre",
            "pdbt_nrel_status"
        ]:
            col_nombre = col + "_nombre"

            if col_nombre in temp.columns:
                df.loc[faltantes, col] = temp[col_nombre].values

        df.loc[
            faltantes & df["demanda_promedio_anual_kw"].notna(),
            "criterio_union_demanda"
        ] = "Nombre comercial"

    # --------------------------------------------------
    # 4) Fallback por No. de local en dirección/textos del parser
    # --------------------------------------------------
    # Este fallback replica el criterio usado en Calidad de Datos.
    # Sirve para locales que sí entraron a muestra_con_recibo por
    # No. local / dirección, pero que no lograron traer parser por
    # medidor, cliente o nombre comercial.

    local_col_muestra = first_existing_column(
        df,
        [
            "No de Local",
            "No de local",
            "No. de Local",
            "No. de local",
            "No Local",
            "No. Local",
            "LOCAL",
            "Local"
        ]
    )

    parser_text_cols = [
        col for col in [
            "direccion_completa",
            "direccion_raw",
            "file_path",
            "file_name",
            "recibos_subgroup"
        ]
        if col in demanda_lookup.columns
    ]

    def generar_variantes_no_local_enriquecimiento(no_local):
        raw = str(no_local).upper().strip()

        if raw in ["", "NAN", "NONE"]:
            return []

        raw = (
            raw
            .replace("LOCAL", "")
            .replace("LOC.", "")
            .replace("LOC", "")
            .replace("NO.", "")
            .replace("NO", "")
            .replace("#", "")
            .strip()
        )

        raw = re.sub(r"\s+", " ", raw)

        variantes = set()

        prefijo_match = re.match(r"^([A-Z0-9]+)[\-\s]*", raw)
        prefijo_base = prefijo_match.group(1) if prefijo_match else ""

        partes = re.split(r",|/|\bY\b|\bE\b|;", raw)

        for parte in partes:
            parte = parte.strip()

            if not parte:
                continue

            compact = re.sub(r"[^A-Z0-9]", "", parte)

            if len(compact) >= 3:
                variantes.add(compact)

            if prefijo_base and re.fullmatch(r"[0-9]+[A-Z]?", compact):
                variantes.add(f"{prefijo_base}{compact}")

            m = re.match(r"^([A-Z0-9]+)[\-\s]*([0-9]+[A-Z]?)$", parte)

            if m:
                variantes.add(f"{m.group(1)}{m.group(2)}")

        full_compact = re.sub(r"[^A-Z0-9]", "", raw)

        if len(full_compact) >= 3:
            variantes.add(full_compact)

        variantes = {
            v for v in variantes
            if len(v) >= 3
            and v not in ["LOCAL", "NAN", "NONE"]
        }

        return list(variantes)

    def normalizar_texto_para_local_enriquecimiento(value):
        return re.sub(
            r"[^A-Z0-9]",
            "",
            normalizar_texto_simple(value)
        )

    cols_a_llenar_desde_parser = [
        col for col in [
            "no_servicio",
            "medidor",
            "tarifa_norm",
            "demanda_promedio_anual_kw",
            "demanda_maxima_anual_kw",
            "meses_con_demanda",
            "periodo_12m_inicio",
            "periodo_12m_fin",
            "kwh_12m",
            "zona_nrel",
            "perfil_nrel_code",
            "perfil_nrel_nombre",
            "profile_path_nrel",
            "pdbt_nrel_status",
            "centro_comercial_nrel_input",
            "subgiro_nrel_input",
            "tipo_local_nrel_input"
        ]
        if col in demanda_lookup.columns
    ]

    if local_col_muestra and parser_text_cols:

        # Solo intentamos este fallback en locales que siguen sin tarifa
        # o sin demanda después de los matches anteriores.
        faltantes_parser = (
            df["tarifa_norm"].isna()
            | df["tarifa_norm"].astype(str).str.upper().isin(
                ["", "NAN", "NONE", "SIN TARIFA"]
            )
            | df["demanda_maxima_anual_kw"].isna()
        )

        for idx, row in df[faltantes_parser].iterrows():

            variantes_local = generar_variantes_no_local_enriquecimiento(
                row.get(local_col_muestra, "")
            )

            if not variantes_local:
                continue

            candidatos = demanda_lookup[
                demanda_lookup["_cc_key"].eq(row.get("_cc_key", ""))
            ].copy()

            if candidatos.empty:
                continue

            candidatos["_texto_local_parser"] = ""

            for col_texto in parser_text_cols:
                candidatos["_texto_local_parser"] = (
                    candidatos["_texto_local_parser"].astype(str)
                    + " "
                    + candidatos[col_texto].fillna("").astype(str)
                )

            candidatos["_texto_local_parser_key"] = candidatos[
                "_texto_local_parser"
            ].apply(normalizar_texto_para_local_enriquecimiento)

            mask_local = candidatos["_texto_local_parser_key"].apply(
                lambda texto: any(
                    variante in texto
                    for variante in variantes_local
                )
            )

            candidatos_match = candidatos[mask_local].copy()

            if candidatos_match.empty:
                continue

            # Si hay varios candidatos, preferimos:
            # 1. el que tenga demanda máxima anual calculada
            # 2. el que tenga tarifa
            # 3. el más reciente por periodo_12m_fin, si existe
            if "demanda_maxima_anual_kw" in candidatos_match.columns:
                candidatos_match["_tiene_demanda"] = (
                    candidatos_match["demanda_maxima_anual_kw"].notna()
                )
            else:
                candidatos_match["_tiene_demanda"] = False

            if "tarifa_norm" in candidatos_match.columns:
                candidatos_match["_tiene_tarifa"] = (
                    candidatos_match["tarifa_norm"].notna()
                    & ~candidatos_match["tarifa_norm"]
                    .astype(str)
                    .str.upper()
                    .isin(["", "NAN", "NONE", "SIN TARIFA"])
                )
            else:
                candidatos_match["_tiene_tarifa"] = False

            if "periodo_12m_fin" in candidatos_match.columns:
                candidatos_match["_periodo_sort"] = pd.to_datetime(
                    candidatos_match["periodo_12m_fin"],
                    errors="coerce"
                )
            else:
                candidatos_match["_periodo_sort"] = pd.NaT

            candidatos_match = candidatos_match.sort_values(
                [
                    "_tiene_demanda",
                    "_tiene_tarifa",
                    "_periodo_sort"
                ],
                ascending=[
                    False,
                    False,
                    False
                ]
            )

            mejor_match = candidatos_match.iloc[0]

            for col in cols_a_llenar_desde_parser:
                if col in df.columns:
                    df.at[idx, col] = mejor_match.get(col, pd.NA)

            df.at[idx, "criterio_union_demanda"] = "No. local / dirección"

    return df



@st.cache_data(show_spinner=False)
def construir_base_densidad_demanda(
    muestra_con_recibo: pd.DataFrame,
    parsed: pd.DataFrame
) -> pd.DataFrame:
    """
    Construye la base maestra de densidad de demanda.

    Esta base debe ser la única fuente para:
    - Benchmark de densidad en Resumen Ejecutivo
    - Densidad de demanda en Centro Comercial

    Universo:
    - Locales ocupados con recibo de muestra_con_recibo

    Demanda:
    - demanda_maxima_anual_kw, proveniente del flujo ya calculado:
        - kwmax para GDMTH/GDMTO/GDBT
        - NREL para PDBT cuando exista perfil disponible
    """

    if muestra_con_recibo.empty or parsed.empty:
        return pd.DataFrame()

    df = enriquecer_muestra_con_demanda(
        muestra_con_recibo=muestra_con_recibo,
        parsed=parsed
    )

    if df.empty:
        return df

    # --------------------------------------------------------
    # Centro comercial normalizado
    # --------------------------------------------------------

    if "_centro_comercial_limpio" in df.columns:
        df["_cc_key_reporte"] = df["_centro_comercial_limpio"].apply(cc_key)

    elif "NOMBRE DEL CC" in df.columns:
        df["_cc_key_reporte"] = df["NOMBRE DEL CC"].apply(cc_key)

    elif "CENTRO COMERCIAL" in df.columns:
        df["_cc_key_reporte"] = df["CENTRO COMERCIAL"].apply(cc_key)

    elif "source_sheet" in df.columns:
        df["_cc_key_reporte"] = df["source_sheet"].apply(cc_key)

    else:
        df["_cc_key_reporte"] = ""

    # --------------------------------------------------------
    # Nombre comercial
    # --------------------------------------------------------

    if "NOMBRE COMERCIAL" in df.columns:
        df["Nombre Comercial"] = df["NOMBRE COMERCIAL"]

    elif "CLIENTE" in df.columns:
        df["Nombre Comercial"] = df["CLIENTE"]

    elif "cliente_nombre" in df.columns:
        df["Nombre Comercial"] = df["cliente_nombre"]

    else:
        df["Nombre Comercial"] = ""

    df["Nombre Comercial"] = (
        df["Nombre Comercial"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Tarifa
    # --------------------------------------------------------
    # La base maestra de densidad debe usar TARIFA_FINAL.
    #
    # TARIFA_FINAL ya fue calculada en el match global con la regla:
    # 1. parser_tarifa_match / tarifa del parser
    # 2. tarifa_norm / tarifa normalizada del parser
    # 3. respaldo DG solo si no hubo tarifa del parser
    #
    # Aquí NO volvemos a decidir la prioridad; solo normalizamos
    # visualmente y dejamos una columna estándar llamada "Tarifa".

    if "TARIFA_FINAL" in df.columns:
        df["Tarifa"] = normalize_tarifa_series(
            df["TARIFA_FINAL"]
        )

    else:
        # Respaldo defensivo por si algún día la base llega sin TARIFA_FINAL.
        # No debería pasar si muestra_con_recibo_global está bien construida.
        df["Tarifa"] = pd.NA

        posibles_cols_tarifa_respaldo = [
            "parser_tarifa_match",
            "tarifa_norm",
            "tarifa",
            "TARIFA_ANALISIS",
            "TARIFA",
            "Tarifa"
        ]

        for col_tarifa in posibles_cols_tarifa_respaldo:
            if col_tarifa not in df.columns:
                continue

            valores_tarifa = normalize_tarifa_series(
                df[col_tarifa]
            )

            valores_tarifa = valores_tarifa.replace({
                "": pd.NA,
                "NAN": pd.NA,
                "NONE": pd.NA,
                "<NA>": pd.NA,
                "SIN TARIFA": pd.NA,
                "SIN TARIFA / SIN DEMANDA ASOCIADA": pd.NA
            })

            faltantes_tarifa = (
                df["Tarifa"].isna()
                | df["Tarifa"]
                .astype(str)
                .str.upper()
                .str.strip()
                .isin([
                    "",
                    "NAN",
                    "NONE",
                    "<NA>",
                    "SIN TARIFA",
                    "SIN TARIFA / SIN DEMANDA ASOCIADA"
                ])
            )

            df.loc[
                faltantes_tarifa,
                "Tarifa"
            ] = valores_tarifa.loc[faltantes_tarifa]

    df["Tarifa"] = (
        df["Tarifa"]
        .fillna("SIN TARIFA")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Medidor y número de servicio para visualización
    # --------------------------------------------------------

    if "medidor" not in df.columns:
        df["medidor"] = pd.NA

    if "parser_medidor_match" in df.columns:
        medidor_match_valido = (
            df["parser_medidor_match"].notna()
            & ~df["parser_medidor_match"]
            .astype(str)
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
        )

        medidor_actual_faltante = (
            df["medidor"].isna()
            | df["medidor"]
            .astype(str)
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
        )

        df.loc[
            medidor_actual_faltante & medidor_match_valido,
            "medidor"
        ] = df.loc[
            medidor_actual_faltante & medidor_match_valido,
            "parser_medidor_match"
        ]

    if "no_servicio" not in df.columns:
        df["no_servicio"] = pd.NA

    if "parser_no_servicio_match" in df.columns:
        servicio_match_valido = (
            df["parser_no_servicio_match"].notna()
            & ~df["parser_no_servicio_match"]
            .astype(str)
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
        )

        servicio_actual_faltante = (
            df["no_servicio"].isna()
            | df["no_servicio"]
            .astype(str)
            .str.upper()
            .isin(["", "NAN", "NONE", "<NA>"])
        )

        df.loc[
            servicio_actual_faltante & servicio_match_valido,
            "no_servicio"
        ] = df.loc[
            servicio_actual_faltante & servicio_match_valido,
            "parser_no_servicio_match"
        ]

    # --------------------------------------------------------
    # Giro comercial
    # --------------------------------------------------------

    giro_col = first_existing_column(
        df,
        [
            "SUBGIRO_COMERCIAL",
            "SUBGIRO COMERCIAL",
            "GIRO_COMERCIAL",
            "GIRO COMERCIAL",
            "GIRO",
            "Giro"
        ]
    )

    if giro_col is None:
        df["Giro comercial densidad"] = "Sin giro"
    else:
        df["Giro comercial densidad"] = (
            df[giro_col]
            .fillna("Sin giro")
            .astype(str)
            .str.strip()
            .replace({"": "Sin giro", "nan": "Sin giro", "None": "Sin giro"})
        )

    # --------------------------------------------------------
    # Área m²
    # --------------------------------------------------------

    area_col = first_existing_column(
        df,
        [
            "MTS2_num",
            "M2_num",
            "m2_num",
            "MTS 2_num",
            "MTS²_num",
            "AREA_M2_num",
            "AREA M2_num",
            "SUPERFICIE_num",
            "SUPERFICIE M2_num",
            "MTS2",
            "M2",
            "m2",
            "MTS 2",
            "MTS²",
            "AREA_M2",
            "AREA M2",
            "SUPERFICIE",
            "SUPERFICIE M2",
            "Área",
            "Area",
            "AREA",
            "Superficie"
        ]
    )

    if area_col is None:
        df["Area m2"] = pd.NA

    elif str(area_col).endswith("_num"):
        df["Area m2"] = pd.to_numeric(
            df[area_col],
            errors="coerce"
        )

    else:
        df["Area m2"] = clean_number_series(
            df[area_col]
        )

    # --------------------------------------------------------
    # Demanda promedio anual
    # --------------------------------------------------------
    # Concepto estándar de toda la app:
    # promedio de las demandas máximas de los recibos más recientes disponibles.
    #
    # Si los recibos son bimestrales, se usan hasta 6.
    # Si los recibos son mensuales, se usan hasta 12.
    # Si hay menos recibos disponibles, se promedia con los existentes.

    if "demanda_promedio_anual_kw" in df.columns:
        df["Demanda promedio anual (kW)"] = pd.to_numeric(
            df["demanda_promedio_anual_kw"],
            errors="coerce"
        )
    else:
        df["Demanda promedio anual (kW)"] = pd.NA

    # Conservamos esta columna solo como respaldo/auditoría si existe,
    # pero ya no será la métrica principal de la app.
    if "demanda_maxima_anual_kw" in df.columns:
        df["Demanda máxima anual (kW)"] = pd.to_numeric(
            df["demanda_maxima_anual_kw"],
            errors="coerce"
        )
    else:
        df["Demanda máxima anual (kW)"] = pd.NA

    # --------------------------------------------------------
    # Densidad de demanda
    # --------------------------------------------------------
    # Se reporta en W/m².
    # Fórmula:
    # Demanda promedio anual (kW) / Área m² × 1,000

    df["Densidad de demanda W/m2"] = pd.NA

    mask_densidad = (
        df["Demanda promedio anual (kW)"].notna()
        & df["Area m2"].notna()
        & (df["Area m2"] > 0)
    )

    df.loc[
        mask_densidad,
        "Densidad de demanda W/m2"
    ] = (
        df.loc[
            mask_densidad,
            "Demanda promedio anual (kW)"
        ]
        / df.loc[
            mask_densidad,
            "Area m2"
        ]
        * 1000
    )

    # Columna de compatibilidad si otras secciones todavía la buscan.
    df["Densidad de demanda kW/m2"] = (
        pd.to_numeric(
            df["Densidad de demanda W/m2"],
            errors="coerce"
        )
        / 1000
    )

    # --------------------------------------------------------
    # Estatus de demanda
    # --------------------------------------------------------

    df["estatus_demanda"] = "Pendiente"

    df.loc[
        df["Demanda promedio anual (kW)"].notna(),
        "estatus_demanda"
    ] = "Calculada"

    df.loc[
        df["Tarifa"].isin(["GDMTH", "GDMTO", "GDBT"])
        & df["Demanda promedio anual (kW)"].notna(),
        "estatus_demanda"
    ] = "Calculada con kwmax"

    df.loc[
        df["Tarifa"].eq("PDBT")
        & df["Demanda promedio anual (kW)"].notna(),
        "estatus_demanda"
    ] = "Estimada PDBT con NREL"

    df.loc[
        df["Tarifa"].eq("PDBT")
        & df["Demanda promedio anual (kW)"].isna(),
        "estatus_demanda"
    ] = "Pendiente perfil NREL"

    return df

def aplicar_calibracion_hibrida_pdbt_allux(
    benchmark_df: pd.DataFrame,
    min_medidos_giro_clima: int = 5,
    min_pdbt_giro_clima: int = 3,
    min_medidos_giro: int = 5,
    min_pdbt_giro: int = 3,
    min_medidos_clima: int = 10,
    min_pdbt_clima: int = 5,
    factor_min: float = 0.75,
    factor_max: float = 4.00
):
    """
    Calibra la demanda estimada PDBT con factores Allux por giro.

    Metodología:
    1) Para locales medidos GDMTH/GDMTO/GDBT:
       ratio_real = demanda_promedio_anual_kw / kWh_día_promedio

    2) Para PDBT con NREL:
       ratio_nrel = demanda_pdbt_nrel_kw / kWh_día_promedio

    3) Por giro:
       factor_allux = mediana(ratio_real medido) / mediana(ratio_nrel PDBT)

    4) Para PDBT:
       demanda_híbrida = demanda_nrel_original × factor_allux

    Si no hay muestra suficiente por giro, se usa factor general del portafolio.
    """

    if benchmark_df is None or benchmark_df.empty:
        return benchmark_df, pd.DataFrame()

    df = benchmark_df.copy()

    demanda_col = first_existing_column(
        df,
        [
            "Demanda promedio anual (kW)",
            "demanda_promedio_anual_kw"
        ]
    )

    densidad_col = first_existing_column(
        df,
        [
            "Densidad de demanda W/m2",
            "densidad_demanda_promedio_anual_w_m2"
        ]
    )

    area_col = first_existing_column(
        df,
        [
            "Area m2",
            "Área m2",
            "MTS2",
            "M2",
            "m2",
            "AREA_M2",
            "AREA M2"
        ]
    )

    tarifa_col = first_existing_column(
        df,
        [
            "Tarifa",
            "tarifa_norm",
            "TARIFA_FINAL"
        ]
    )

    giro_col = first_existing_column(
        df,
        [
            "Giro comercial densidad",
            "SUBGIRO_COMERCIAL",
            "SUBGIRO COMERCIAL",
            "GIRO_COMERCIAL",
            "GIRO COMERCIAL",
            "GIRO",
            "Giro"
        ]
    )

    clima_col = first_existing_column(
        df,
        [
            "Clima benchmark",
            "clima_benchmark",
            "Clima",
            "CLIMA",
            "zona_nrel",
            "Zona NREL"
        ]
    )

    kwh_col = first_existing_column(
        df,
        [
            "kwh_12m",
            "KWh 12m",
            "kWh 12m"
        ]
    )

    inicio_col = first_existing_column(
        df,
        [
            "periodo_12m_inicio",
            "Periodo 12m inicio"
        ]
    )

    fin_col = first_existing_column(
        df,
        [
            "periodo_12m_fin",
            "Periodo 12m fin"
        ]
    )

    if (
        demanda_col is None
        or area_col is None
        or tarifa_col is None
        or giro_col is None
        or kwh_col is None
    ):
        factores_vacio = pd.DataFrame({
            "Estatus": [
                "No se pudo calcular factor híbrido: faltan columnas base"
            ]
        })
        return df, factores_vacio

    df["_demanda_factor_kw"] = pd.to_numeric(
        df[demanda_col],
        errors="coerce"
    )

    df["_area_factor_m2"] = pd.to_numeric(
        df[area_col],
        errors="coerce"
    )

    df["_kwh_factor_12m"] = pd.to_numeric(
        df[kwh_col],
        errors="coerce"
    )

    df["_tarifa_factor"] = normalize_tarifa_series(
        df[tarifa_col]
    ).fillna("SIN TARIFA").astype(str).str.upper().str.strip()

    df["_giro_factor"] = (
        df[giro_col]
        .fillna("Sin giro")
        .astype(str)
        .str.strip()
        .replace({"": "Sin giro", "nan": "Sin giro", "None": "Sin giro"})
    )

    if clima_col is None:
        df["_clima_factor"] = "Sin clima"
    else:
        df["_clima_factor"] = (
            df[clima_col]
            .fillna("Sin clima")
            .astype(str)
            .str.strip()
            .replace({"": "Sin clima", "nan": "Sin clima", "None": "Sin clima"})
        )

    # ------------------------------------------------------------
    # Días usados para convertir kWh 12m a kWh/día
    # ------------------------------------------------------------

    if inicio_col and fin_col:
        inicio = pd.to_datetime(
            df[inicio_col],
            errors="coerce"
        )
        fin = pd.to_datetime(
            df[fin_col],
            errors="coerce"
        )

        dias = (fin - inicio).dt.days.abs() + 1

    else:
        dias = pd.Series(pd.NA, index=df.index)

    if "meses_con_demanda" in df.columns:
        dias_fallback_meses = (
            pd.to_numeric(
                df["meses_con_demanda"],
                errors="coerce"
            )
            * 30.4375
        )
        dias = dias.fillna(dias_fallback_meses)

    if "recibos_usados_demanda" in df.columns:
        dias_fallback_recibos = (
            pd.to_numeric(
                df["recibos_usados_demanda"],
                errors="coerce"
            )
            * 30.4375
        )
        dias = dias.fillna(dias_fallback_recibos)

    df["_dias_factor"] = pd.to_numeric(
        dias,
        errors="coerce"
    )

    df["_kwh_dia_factor"] = (
        df["_kwh_factor_12m"]
        / df["_dias_factor"]
    )

    df["_ratio_kw_por_kwh_dia"] = (
        df["_demanda_factor_kw"]
        / df["_kwh_dia_factor"]
    )

    mask_ratio_valido = (
        df["_ratio_kw_por_kwh_dia"].notna()
        & (df["_ratio_kw_por_kwh_dia"] > 0)
        & df["_demanda_factor_kw"].notna()
        & (df["_demanda_factor_kw"] > 0)
        & df["_kwh_dia_factor"].notna()
        & (df["_kwh_dia_factor"] > 0)
    )

    mask_medidos = (
        mask_ratio_valido
        & df["_tarifa_factor"].isin(["GDMTH", "GDMTO", "GDBT"])
    )

    mask_pdbt = (
        mask_ratio_valido
        & df["_tarifa_factor"].eq("PDBT")
    )

    medidos = df[mask_medidos].copy()
    pdbt_nrel = df[mask_pdbt].copy()

    if medidos.empty or pdbt_nrel.empty:
        factores_vacio = pd.DataFrame({
            "Estatus": [
                "No se pudo calcular factor híbrido: no hay muestra medida o PDBT suficiente"
            ]
        })
        return df, factores_vacio

    ratio_medido_general = medidos["_ratio_kw_por_kwh_dia"].median()
    ratio_nrel_general = pdbt_nrel["_ratio_kw_por_kwh_dia"].median()

    if (
        pd.isna(ratio_medido_general)
        or pd.isna(ratio_nrel_general)
        or ratio_nrel_general <= 0
    ):
        factor_general = 1.0
    else:
        factor_general = ratio_medido_general / ratio_nrel_general

    factor_general_acotado = min(
        max(factor_general, factor_min),
        factor_max
    )

    # ============================================================
    # Factores Nivel 1: giro + clima
    # ============================================================

    factores_gc_medidos = (
        medidos
        .groupby(["_giro_factor", "_clima_factor"], dropna=False)
        .agg(
            locales_medidos_giro_clima=(
                "_ratio_kw_por_kwh_dia",
                "count"
            ),
            ratio_allux_giro_clima=(
                "_ratio_kw_por_kwh_dia",
                "median"
            )
        )
        .reset_index()
    )

    factores_gc_pdbt = (
        pdbt_nrel
        .groupby(["_giro_factor", "_clima_factor"], dropna=False)
        .agg(
            locales_pdbt_giro_clima=(
                "_ratio_kw_por_kwh_dia",
                "count"
            ),
            ratio_nrel_giro_clima=(
                "_ratio_kw_por_kwh_dia",
                "median"
            )
        )
        .reset_index()
    )

    factores_gc = factores_gc_medidos.merge(
        factores_gc_pdbt,
        on=["_giro_factor", "_clima_factor"],
        how="outer"
    )

    factores_gc["factor_giro_clima_sin_acotar"] = (
        factores_gc["ratio_allux_giro_clima"]
        / factores_gc["ratio_nrel_giro_clima"]
    )

    factores_gc["factor_giro_clima"] = factores_gc[
        "factor_giro_clima_sin_acotar"
    ].clip(
        lower=factor_min,
        upper=factor_max
    )

    factores_gc["usar_giro_clima"] = (
        factores_gc["locales_medidos_giro_clima"].fillna(0).ge(min_medidos_giro_clima)
        & factores_gc["locales_pdbt_giro_clima"].fillna(0).ge(min_pdbt_giro_clima)
        & factores_gc["factor_giro_clima"].notna()
        & factores_gc["factor_giro_clima"].gt(0)
    )

    # ============================================================
    # Factores Nivel 2: giro
    # ============================================================

    factores_giro_medidos = (
        medidos
        .groupby("_giro_factor", dropna=False)
        .agg(
            locales_medidos_giro=(
                "_ratio_kw_por_kwh_dia",
                "count"
            ),
            ratio_allux_giro=(
                "_ratio_kw_por_kwh_dia",
                "median"
            )
        )
        .reset_index()
    )

    factores_giro_pdbt = (
        pdbt_nrel
        .groupby("_giro_factor", dropna=False)
        .agg(
            locales_pdbt_giro=(
                "_ratio_kw_por_kwh_dia",
                "count"
            ),
            ratio_nrel_giro=(
                "_ratio_kw_por_kwh_dia",
                "median"
            )
        )
        .reset_index()
    )

    factores_giro = factores_giro_medidos.merge(
        factores_giro_pdbt,
        on="_giro_factor",
        how="outer"
    )

    factores_giro["factor_giro_sin_acotar"] = (
        factores_giro["ratio_allux_giro"]
        / factores_giro["ratio_nrel_giro"]
    )

    factores_giro["factor_giro"] = factores_giro[
        "factor_giro_sin_acotar"
    ].clip(
        lower=factor_min,
        upper=factor_max
    )

    factores_giro["usar_giro"] = (
        factores_giro["locales_medidos_giro"].fillna(0).ge(min_medidos_giro)
        & factores_giro["locales_pdbt_giro"].fillna(0).ge(min_pdbt_giro)
        & factores_giro["factor_giro"].notna()
        & factores_giro["factor_giro"].gt(0)
    )

    # ============================================================
    # Factores Nivel 3: clima
    # ============================================================

    factores_clima_medidos = (
        medidos
        .groupby("_clima_factor", dropna=False)
        .agg(
            locales_medidos_clima=(
                "_ratio_kw_por_kwh_dia",
                "count"
            ),
            ratio_allux_clima=(
                "_ratio_kw_por_kwh_dia",
                "median"
            )
        )
        .reset_index()
    )

    factores_clima_pdbt = (
        pdbt_nrel
        .groupby("_clima_factor", dropna=False)
        .agg(
            locales_pdbt_clima=(
                "_ratio_kw_por_kwh_dia",
                "count"
            ),
            ratio_nrel_clima=(
                "_ratio_kw_por_kwh_dia",
                "median"
            )
        )
        .reset_index()
    )

    factores_clima = factores_clima_medidos.merge(
        factores_clima_pdbt,
        on="_clima_factor",
        how="outer"
    )

    factores_clima["factor_clima_sin_acotar"] = (
        factores_clima["ratio_allux_clima"]
        / factores_clima["ratio_nrel_clima"]
    )

    factores_clima["factor_clima"] = factores_clima[
        "factor_clima_sin_acotar"
    ].clip(
        lower=factor_min,
        upper=factor_max
    )

    factores_clima["usar_clima"] = (
        factores_clima["locales_medidos_clima"].fillna(0).ge(min_medidos_clima)
        & factores_clima["locales_pdbt_clima"].fillna(0).ge(min_pdbt_clima)
        & factores_clima["factor_clima"].notna()
        & factores_clima["factor_clima"].gt(0)
    )

    # ============================================================
    # Mapas de factores
    # ============================================================

    factor_gc_map = {
        (row["_giro_factor"], row["_clima_factor"]): row["factor_giro_clima"]
        for _, row in factores_gc[factores_gc["usar_giro_clima"]].iterrows()
    }

    factor_giro_map = {
        row["_giro_factor"]: row["factor_giro"]
        for _, row in factores_giro[factores_giro["usar_giro"]].iterrows()
    }

    factor_clima_map = {
        row["_clima_factor"]: row["factor_clima"]
        for _, row in factores_clima[factores_clima["usar_clima"]].iterrows()
    }

    def obtener_factor_aplicable(row):
        giro = row.get("_giro_factor", "Sin giro")
        clima = row.get("_clima_factor", "Sin clima")

        if (giro, clima) in factor_gc_map:
            return factor_gc_map[(giro, clima)], "Giro + clima"

        if giro in factor_giro_map:
            return factor_giro_map[giro], "Giro comercial"

        if clima in factor_clima_map:
            return factor_clima_map[clima], "Clima"

        return factor_general_acotado, "General portafolio"

    mask_pdbt_aplicar = (
        df["_tarifa_factor"].eq("PDBT")
        & df["_demanda_factor_kw"].notna()
        & df["_demanda_factor_kw"].gt(0)
    )

    df["demanda_pdbt_nrel_original_kw"] = pd.NA
    df["factor_ajuste_allux_pdbt"] = pd.NA
    df["nivel_factor_ajuste_allux_pdbt"] = pd.NA

    df.loc[
        mask_pdbt_aplicar,
        "demanda_pdbt_nrel_original_kw"
    ] = df.loc[
        mask_pdbt_aplicar,
        "_demanda_factor_kw"
    ]

    factores_aplicados = df.loc[
        mask_pdbt_aplicar
    ].apply(
        obtener_factor_aplicable,
        axis=1
    )

    df.loc[
        mask_pdbt_aplicar,
        "factor_ajuste_allux_pdbt"
    ] = factores_aplicados.apply(lambda x: x[0]).values

    df.loc[
        mask_pdbt_aplicar,
        "nivel_factor_ajuste_allux_pdbt"
    ] = factores_aplicados.apply(lambda x: x[1]).values

    demanda_hibrida = (
        pd.to_numeric(
            df.loc[
                mask_pdbt_aplicar,
                "demanda_pdbt_nrel_original_kw"
            ],
            errors="coerce"
        )
        * pd.to_numeric(
            df.loc[
                mask_pdbt_aplicar,
                "factor_ajuste_allux_pdbt"
            ],
            errors="coerce"
        )
    )

    # Actualizar demanda en columnas principales.
    df.loc[
        mask_pdbt_aplicar,
        demanda_col
    ] = demanda_hibrida.values

    if "demanda_promedio_anual_kw" in df.columns:
        df.loc[
            mask_pdbt_aplicar,
            "demanda_promedio_anual_kw"
        ] = demanda_hibrida.values

    if "Demanda promedio anual (kW)" in df.columns:
        df.loc[
            mask_pdbt_aplicar,
            "Demanda promedio anual (kW)"
        ] = demanda_hibrida.values

    # Recalcular densidad PDBT ajustada.
    area_pdbt = pd.to_numeric(
        df.loc[
            mask_pdbt_aplicar,
            "_area_factor_m2"
        ],
        errors="coerce"
    )

    densidad_hibrida = (
        demanda_hibrida
        / area_pdbt
        * 1000
    )

    densidad_hibrida = densidad_hibrida.where(
        area_pdbt.notna() & area_pdbt.gt(0),
        pd.NA
    )

    if densidad_col:
        df.loc[
            mask_pdbt_aplicar,
            densidad_col
        ] = densidad_hibrida.values

    if "densidad_demanda_promedio_anual_w_m2" in df.columns:
        df.loc[
            mask_pdbt_aplicar,
            "densidad_demanda_promedio_anual_w_m2"
        ] = densidad_hibrida.values

    if "Densidad de demanda W/m2" in df.columns:
        df.loc[
            mask_pdbt_aplicar,
            "Densidad de demanda W/m2"
        ] = densidad_hibrida.values

    if "estatus_demanda" in df.columns:
        df.loc[
            mask_pdbt_aplicar,
            "estatus_demanda"
        ] = "Estimada PDBT híbrida NREL + factor Allux"

    # Tabla final para Anexo.
    # ============================================================
    # Tabla final para Anexo
    # ============================================================

    factores_gc_anexo = factores_gc.copy()
    factores_gc_anexo["Nivel evaluado"] = "Giro + clima"
    factores_gc_anexo = factores_gc_anexo.rename(columns={
        "_giro_factor": "Giro comercial",
        "_clima_factor": "Clima",
        "locales_medidos_giro_clima": "Locales medidos calibración",
        "locales_pdbt_giro_clima": "Locales PDBT base NREL",
        "ratio_allux_giro_clima": "Ratio Allux medido kW/(kWh/día)",
        "ratio_nrel_giro_clima": "Ratio NREL PDBT kW/(kWh/día)",
        "factor_giro_clima_sin_acotar": "Factor sin acotar",
        "factor_giro_clima": "Factor calculado",
        "usar_giro_clima": "Muestra suficiente"
    })

    factores_giro_anexo = factores_giro.copy()
    factores_giro_anexo["Clima"] = "Todos"
    factores_giro_anexo["Nivel evaluado"] = "Giro comercial"
    factores_giro_anexo = factores_giro_anexo.rename(columns={
        "_giro_factor": "Giro comercial",
        "locales_medidos_giro": "Locales medidos calibración",
        "locales_pdbt_giro": "Locales PDBT base NREL",
        "ratio_allux_giro": "Ratio Allux medido kW/(kWh/día)",
        "ratio_nrel_giro": "Ratio NREL PDBT kW/(kWh/día)",
        "factor_giro_sin_acotar": "Factor sin acotar",
        "factor_giro": "Factor calculado",
        "usar_giro": "Muestra suficiente"
    })

    factores_clima_anexo = factores_clima.copy()
    factores_clima_anexo["Giro comercial"] = "Todos"
    factores_clima_anexo["Nivel evaluado"] = "Clima"
    factores_clima_anexo = factores_clima_anexo.rename(columns={
        "_clima_factor": "Clima",
        "locales_medidos_clima": "Locales medidos calibración",
        "locales_pdbt_clima": "Locales PDBT base NREL",
        "ratio_allux_clima": "Ratio Allux medido kW/(kWh/día)",
        "ratio_nrel_clima": "Ratio NREL PDBT kW/(kWh/día)",
        "factor_clima_sin_acotar": "Factor sin acotar",
        "factor_clima": "Factor calculado",
        "usar_clima": "Muestra suficiente"
    })

    factores_anexo = pd.concat(
        [
            factores_gc_anexo,
            factores_giro_anexo,
            factores_clima_anexo
        ],
        ignore_index=True
    )

    columnas_factores_anexo = [
        "Nivel evaluado",
        "Giro comercial",
        "Clima",
        "Locales medidos calibración",
        "Locales PDBT base NREL",
        "Ratio Allux medido kW/(kWh/día)",
        "Ratio NREL PDBT kW/(kWh/día)",
        "Factor sin acotar",
        "Factor calculado",
        "Muestra suficiente"
    ]

    factores_anexo = factores_anexo[
        [
            col for col in columnas_factores_anexo
            if col in factores_anexo.columns
        ]
    ].copy()

    factores_anexo["Factor general portafolio"] = factor_general_acotado

    factores_anexo["Mín. medidos giro+clima"] = min_medidos_giro_clima
    factores_anexo["Mín. PDBT giro+clima"] = min_pdbt_giro_clima
    factores_anexo["Mín. medidos giro"] = min_medidos_giro
    factores_anexo["Mín. PDBT giro"] = min_pdbt_giro
    factores_anexo["Mín. medidos clima"] = min_medidos_clima
    factores_anexo["Mín. PDBT clima"] = min_pdbt_clima

    cols_round = [
        "Ratio Allux medido kW/(kWh/día)",
        "Ratio NREL PDBT kW/(kWh/día)",
        "Factor sin acotar",
        "Factor calculado",
        "Factor general portafolio"
    ]

    for col in cols_round:
        if col in factores_anexo.columns:
            factores_anexo[col] = pd.to_numeric(
                factores_anexo[col],
                errors="coerce"
            ).round(3)

    for col in [
        "Locales medidos calibración",
        "Locales PDBT base NREL",
        "Mín. medidos giro+clima",
        "Mín. PDBT giro+clima",
        "Mín. medidos giro",
        "Mín. PDBT giro",
        "Mín. medidos clima",
        "Mín. PDBT clima"
    ]:
        if col in factores_anexo.columns:
            factores_anexo[col] = pd.to_numeric(
                factores_anexo[col],
                errors="coerce"
            ).fillna(0).astype(int)

    df = df.drop(
        columns=[
            "_demanda_factor_kw",
            "_area_factor_m2",
            "_kwh_factor_12m",
            "_tarifa_factor",
            "_giro_factor",
            "_clima_factor",
            "_dias_factor",
            "_kwh_dia_factor",
            "_ratio_kw_por_kwh_dia"
        ],
        errors="ignore"
    )

    return df, factores_anexo

# ============================================================
# Base maestra de densidad de demanda
# ============================================================

benchmark_densidad_base = construir_base_densidad_demanda(
    muestra_con_recibo=muestra_con_recibo_global,
    parsed=parsed
)

benchmark_densidad_base, factores_ajuste_allux_pdbt_df = (
    aplicar_calibracion_hibrida_pdbt_allux(
        benchmark_densidad_base,
        min_medidos_giro_clima=5,
        min_pdbt_giro_clima=3,
        min_medidos_giro=5,
        min_pdbt_giro=3,
        min_medidos_clima=10,
        min_pdbt_clima=5,
        factor_min=0.75,
        factor_max=4.00
    )
)

with tab_resumen:
    # ============================================================
    # Resumen ejecutivo
    # ============================================================

    st.markdown(
        '<div class="section-title">Resumen ejecutivo</div>',
        unsafe_allow_html=True
    )

    # ------------------------------------------------------------
    # Base DG
    # ------------------------------------------------------------

    resumen_general = general_data.copy()

    # Total de centros comerciales analizados
    if "coverage_by_mall" in globals() and not coverage_by_mall.empty:
        total_cc_analizados = (
            coverage_by_mall[
                ~coverage_by_mall["Centro Comercial"]
                .astype(str)
                .str.upper()
                .str.strip()
                .eq("TOTAL")
            ]["Centro Comercial"]
            .nunique()
        )
    else:
        general_cc_col_resumen = first_existing_column(
            resumen_general,
            [
                "NOMBRE DEL CC",
                "CENTRO COMERCIAL",
                "CC",
                "PLAZA",
                "source_sheet"
            ]
        )

        total_cc_analizados = (
            resumen_general[general_cc_col_resumen]
            .dropna()
            .astype(str)
            .apply(cc_key)
            .nunique()
            if general_cc_col_resumen
            else 0
        )

    # ------------------------------------------------------------
    # Excluir Servicios Generales del conteo de locales
    # ------------------------------------------------------------
    # Para Total de locales y Locales ocupados, no contamos filas cuyo
    # giro comercial sea Servicios Generales.

    giro_col_resumen = first_existing_column(
        resumen_general,
        [
            "Giro comercial densidad",
            "SUBGIRO_COMERCIAL",
            "SUBGIRO COMERCIAL",
            "GIRO_COMERCIAL",
            "GIRO COMERCIAL",
            "GIRO",
            "Giro"
        ]
    )

    if giro_col_resumen:
        giro_resumen = (
            resumen_general[giro_col_resumen]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        mask_servicios_generales_resumen = (
            giro_resumen.str.contains("SERVICIOS GENERALES", na=False)
            | giro_resumen.str.contains("SERVICIO GENERAL", na=False)
        )
    else:
        mask_servicios_generales_resumen = pd.Series(
            False,
            index=resumen_general.index
        )

    resumen_general_sin_sg = resumen_general[
        ~mask_servicios_generales_resumen
    ].copy()

    # Total de locales = filas útiles de DG, excluyendo Servicios Generales
    total_locales_dg = len(resumen_general_sin_sg)

    # ------------------------------------------------------------
    # Locales ocupados
    # ------------------------------------------------------------
    # Regla:
    # Un local está ocupado si tiene CLIENTE o NOMBRE COMERCIAL válido,
    # y no dice DISPONIBLE.

    cliente_col_resumen = first_existing_column(
        resumen_general,
        [
            "CLIENTE",
            "Cliente",
            "cliente",
            "NOMBRE CLIENTE",
            "Nombre Cliente"
        ]
    )

    nombre_comercial_col_resumen = first_existing_column(
        resumen_general,
        [
            "NOMBRE COMERCIAL",
            "Nombre Comercial",
            "Nombre comercial",
            "nombre_comercial",
            "NOMBRE_COMERCIAL"
        ]
    )

    if cliente_col_resumen:
        cliente_resumen = (
            resumen_general_sin_sg[cliente_col_resumen]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        cliente_resumen = pd.Series(
            "",
            index=resumen_general_sin_sg.index
        )

    if nombre_comercial_col_resumen:
        nombre_comercial_resumen = (
            resumen_general_sin_sg[nombre_comercial_col_resumen]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )
    else:
        nombre_comercial_resumen = pd.Series(
            "",
            index=resumen_general_sin_sg.index
        )

    cliente_valido_resumen = (
        cliente_resumen.ne("")
        & ~cliente_resumen.isin([
            "NAN",
            "NONE",
            "<NA>",
            "DISPONIBLE",
            "LOCAL DISPONIBLE",
            "VACANTE"
        ])
        & ~cliente_resumen.str.contains("DISPONIBLE", na=False)
    )

    nombre_valido_resumen = (
        nombre_comercial_resumen.ne("")
        & ~nombre_comercial_resumen.isin([
            "NAN",
            "NONE",
            "<NA>",
            "DISPONIBLE",
            "LOCAL DISPONIBLE",
            "VACANTE"
        ])
        & ~nombre_comercial_resumen.str.contains("DISPONIBLE", na=False)
    )

    locales_ocupados_total = int(
        (cliente_valido_resumen | nombre_valido_resumen).sum()
    )

    # ------------------------------------------------------------
    # Locales ocupados con recibo
    # ------------------------------------------------------------

    locales_ocupados_con_recibo_total = pd.NA

    if "coverage_by_mall" in globals() and not coverage_by_mall.empty:

        coverage_total_row = coverage_by_mall[
            coverage_by_mall["Centro Comercial"]
            .astype(str)
            .str.upper()
            .str.strip()
            .eq("TOTAL")
        ]

        if not coverage_total_row.empty:
            locales_ocupados_con_recibo_total = coverage_total_row.iloc[0].get(
                "Locales ocupados con recibo",
                pd.NA
            )

    if pd.isna(locales_ocupados_con_recibo_total):
        locales_ocupados_con_recibo_total = (
            len(muestra_con_recibo_global)
            if "muestra_con_recibo_global" in globals()
            and not muestra_con_recibo_global.empty
            else 0
        )

    locales_ocupados_con_recibo_total = int(locales_ocupados_con_recibo_total)

    # ------------------------------------------------------------
    # Consumo y facturación de locales ocupados con recibo
    # ------------------------------------------------------------
    # Usamos el parser completo como fuente de kWh e importe,
    # consistente con recibos encontrados.

    total_kwh_ocupados_con_recibo = (
        parsed["kwh_total_num"].sum(skipna=True)
        if "kwh_total_num" in parsed.columns
        else pd.NA
    )

    total_importe_ocupados_con_recibo = (
        parsed["importe_total_num"].sum(skipna=True)
        if "importe_total_num" in parsed.columns
        else pd.NA
    )

    costo_promedio_mxn_kwh = pd.NA

    if (
        pd.notna(total_kwh_ocupados_con_recibo)
        and total_kwh_ocupados_con_recibo > 0
        and pd.notna(total_importe_ocupados_con_recibo)
    ):
        costo_promedio_mxn_kwh = (
            total_importe_ocupados_con_recibo
            / total_kwh_ocupados_con_recibo
        )

    # ------------------------------------------------------------
    # kW contratados y demanda promedio anual total
    # ------------------------------------------------------------

    kw_contratados_total = pd.NA

    if (
        "muestra_con_recibo_global" in globals()
        and not muestra_con_recibo_global.empty
        and "demanda_contratada_kw" in muestra_con_recibo_global.columns
    ):
        kw_contratados_total = pd.to_numeric(
            muestra_con_recibo_global["demanda_contratada_kw"],
            errors="coerce"
        ).sum(skipna=True)

    demanda_promedio_anual_total_kw = pd.NA

    if (
        "benchmark_densidad_base" in globals()
        and not benchmark_densidad_base.empty
    ):
        if "demanda_promedio_anual_kw" in benchmark_densidad_base.columns:
            demanda_promedio_anual_total_kw = pd.to_numeric(
                benchmark_densidad_base["demanda_promedio_anual_kw"],
                errors="coerce"
            ).sum(skipna=True)

    # ------------------------------------------------------------
    # Tarjetas principales
    # ------------------------------------------------------------

    cobertura_muestra_pct = (
        locales_ocupados_con_recibo_total / locales_ocupados_total * 100
        if locales_ocupados_total > 0
        else pd.NA
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric(
        "Centros comerciales analizados",
        f"{total_cc_analizados:,}"
    )

    col2.metric(
        "Total de locales",
        f"{total_locales_dg:,}"
    )

    col3.metric(
        "Locales ocupados",
        f"{locales_ocupados_total:,}"
    )

    col4.metric(
        "Locales ocupados con recibo",
        f"{locales_ocupados_con_recibo_total:,}"
    )

    col5.metric(
        "Cobertura de muestra",
        "—" if pd.isna(cobertura_muestra_pct) else f"{cobertura_muestra_pct:.1f}%"
    )

    col5, col6, col7, col8, col9 = st.columns(5)

    col5.metric(
        "Consumo total anual (kWh)",
        format_number(total_kwh_ocupados_con_recibo, 0)
    )

    col6.metric(
        "Demanda contratada (kW)",
        format_number(kw_contratados_total, 0)
    )

    col7.metric(
        "Demanda promedio anual (kW)",
        format_number(demanda_promedio_anual_total_kw, 0)
    )

    col8.metric(
        "Facturación anual",
        format_money_compact(total_importe_ocupados_con_recibo)
    )

    col9.metric(
        "Costo promedio (MXN/kWh)",
        format_mxn_per_kwh(costo_promedio_mxn_kwh)
    )

    # ============================================================
    # Distribución por clima y tipo de centro comercial
    # ============================================================

    st.markdown(
        '<div class="subsection-title">Distribución de centros comerciales y locales</div>',
        unsafe_allow_html=True
    )

    cc_master_path = DATA_DIR / "profiles" / "cc_master_data.csv"

    if not cc_master_path.exists():
        st.warning(
            f"No encontré el archivo maestro de centros comerciales: {cc_master_path}"
        )

    else:
        cc_master_resumen = pd.read_csv(cc_master_path)
        cc_master_resumen.columns = cc_master_resumen.columns.str.strip()

        cc_master_cc_col = first_existing_column(
            cc_master_resumen,
            [
                "Nombre Comercial",
                "centro_comercial",
                "Centro Comercial",
                "NOMBRE DEL CC",
                "Nombre del CC",
                "CC",
                "Plaza",
                "PLAZA"
            ]
        )

        cc_master_tipo_col = first_existing_column(
            cc_master_resumen,
            [
                "Tipo de Mall",
                "Tipo de centro comercial",
                "tipo_cc",
                "TIPO_CC",
                "Tipo CC"
            ]
        )

        cc_master_zona_col = first_existing_column(
            cc_master_resumen,
            [
                "zona_nrel",
                "Zona NREL",
                "ZONA_NREL",
                "climate_zone",
                "Climate Zone",
                "CLIMATE_ZONE",
                "zona_climatica",
                "Zona climática"
            ]
        )

        general_cc_col_resumen = first_existing_column(
            resumen_general,
            [
                "NOMBRE DEL CC",
                "CENTRO COMERCIAL",
                "CC",
                "PLAZA",
                "source_sheet"
            ]
        )

        if (
            cc_master_cc_col is None
            or general_cc_col_resumen is None
        ):
            st.info(
                "No pude cruzar DG contra cc_master_data.csv porque falta la columna de centro comercial."
            )

        else:
            # --------------------------------------------------------
            # Base de locales DG con clasificación de CC
            # --------------------------------------------------------

            locales_resumen = resumen_general.copy()

            locales_resumen["_cc_key_resumen"] = (
                locales_resumen[general_cc_col_resumen]
                .apply(cc_key)
            )

            cc_master_resumen["_cc_key_resumen"] = (
                cc_master_resumen[cc_master_cc_col]
                .apply(cc_key)
            )

            cols_master_resumen = [
                "_cc_key_resumen"
            ]

            if cc_master_tipo_col:
                cols_master_resumen.append(cc_master_tipo_col)

            if cc_master_zona_col:
                cols_master_resumen.append(cc_master_zona_col)

            locales_resumen = locales_resumen.merge(
                cc_master_resumen[cols_master_resumen].drop_duplicates(
                    subset=["_cc_key_resumen"]
                ),
                on="_cc_key_resumen",
                how="left"
            )

            # --------------------------------------------------------
            # Clima
            # --------------------------------------------------------

            if cc_master_zona_col:

                def clasificar_clima_resumen(zona):
                    zona_upper = str(zona).upper()

                    if "HOT" in zona_upper or "CALIDO" in zona_upper or "CÁLIDO" in zona_upper:
                        return "Cálido"

                    if "MIXED" in zona_upper or "TEMPLADO" in zona_upper:
                        return "Templado"

                    if "COLD" in zona_upper or "FRIO" in zona_upper or "FRÍO" in zona_upper:
                        return "Frío"

                    return "Sin clasificar"

                locales_resumen["Clima"] = locales_resumen[
                    cc_master_zona_col
                ].apply(clasificar_clima_resumen)

                # Nombre limpio del CC para mostrar en la tabla
                if "Centro Comercial" not in locales_resumen.columns:
                    if general_cc_col_resumen:
                        locales_resumen["Centro Comercial"] = (
                            locales_resumen[general_cc_col_resumen]
                            .astype(str)
                            .apply(limpiar_nombre_cc)
                        )
                    else:
                        locales_resumen["Centro Comercial"] = ""

                resumen_clima = (
                    locales_resumen
                    .groupby("Clima", dropna=False)
                    .agg(
                        Centros_comerciales=(
                            "_cc_key_resumen",
                            "nunique"
                        ),
                        Locales=(
                            "_cc_key_resumen",
                            "size"
                        ),
                        Centros_comerciales_incluidos=(
                            "Centro Comercial",
                            lambda x: ", ".join(
                                sorted(
                                    set(
                                        x.dropna()
                                        .astype(str)
                                        .str.strip()
                                    )
                                )
                            )
                        )
                    )
                    .reset_index()
                    .rename(columns={
                        "Centros_comerciales": "Centros comerciales",
                        "Centros_comerciales_incluidos": "Centros comerciales incluidos"
                    })
                )

                orden_clima = {
                    "Cálido": 1,
                    "Templado": 2,
                    "Frío": 3,
                    "Sin clasificar": 99
                }

                resumen_clima["_orden"] = (
                    resumen_clima["Clima"]
                    .map(orden_clima)
                    .fillna(99)
                )

                resumen_clima = (
                    resumen_clima
                    .sort_values("_orden")
                    .drop(columns="_orden")
                )


            else:
                st.info(
                    "No encontré columna de zona climática / zona NREL en cc_master_data.csv."
                )

            # --------------------------------------------------------
            # Tipo de centro comercial
            # --------------------------------------------------------

            if cc_master_tipo_col:

                locales_resumen["Tipo de centro comercial"] = (
                    locales_resumen[cc_master_tipo_col]
                    .fillna("Sin clasificar")
                    .astype(str)
                    .str.strip()
                    .replace({
                        "": "Sin clasificar",
                        "nan": "Sin clasificar",
                        "None": "Sin clasificar"
                    })
                )

                resumen_tipo_cc = (
                    locales_resumen
                    .groupby("Tipo de centro comercial", dropna=False)
                    .agg(
                        Centros_comerciales=(
                            "_cc_key_resumen",
                            "nunique"
                        ),
                        Locales=(
                            "_cc_key_resumen",
                            "size"
                        ),
                        Centros_comerciales_incluidos=(
                            "Centro Comercial",
                            lambda x: ", ".join(
                                sorted(
                                    set(
                                        x.dropna()
                                        .astype(str)
                                        .str.strip()
                                    )
                                )
                            )
                        )
                    )
                    .reset_index()
                    .rename(columns={
                        "Centros_comerciales": "Centros comerciales",
                        "Centros_comerciales_incluidos": "Centros comerciales incluidos"
                    })
                )

                orden_tipo_cc = {
                    "Luxury Fashion Mall": 1,
                    "Fashion Mall": 2,
                    "Regional Mall": 3,
                    "Power Center": 4,
                    "Strip Mall": 5,
                    "Sin clasificar": 99
                }

                resumen_tipo_cc["_orden"] = (
                    resumen_tipo_cc["Tipo de centro comercial"]
                    .map(orden_tipo_cc)
                    .fillna(99)
                )

                resumen_tipo_cc = (
                    resumen_tipo_cc
                    .sort_values("_orden")
                    .drop(columns="_orden")
                )



            else:
                st.info(
                    "No encontré columna de tipo de centro comercial en cc_master_data.csv."
                )

            # --------------------------------------------------------
            # Mostrar tablas lado a lado
            # --------------------------------------------------------

            col_clima, col_tipo_cc = st.columns(2)

            with col_clima:
                st.markdown(
                    '<div style="font-size:1.15rem; font-weight:700; color:#2F6FB2; margin-bottom:0.6rem;">Por clima</div>',
                    unsafe_allow_html=True
                )

                if not resumen_clima.empty:
                    st.dataframe(
                        resumen_clima,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Centros comerciales incluidos": st.column_config.TextColumn(
                                "Centros comerciales incluidos",
                                width="large"
                            )
                        }
                    )
                else:
                    st.info("No se pudo construir la tabla por clima.")

            with col_tipo_cc:
                st.markdown(
                    '<div style="font-size:1.15rem; font-weight:700; color:#2F6FB2; margin-bottom:0.6rem;">Por tipo de Centro Comercial</div>',
                    unsafe_allow_html=True
                )

                if not resumen_tipo_cc.empty:
                    st.dataframe(
                        resumen_tipo_cc,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Centros comerciales incluidos": st.column_config.TextColumn(
                                "Centros comerciales incluidos",
                                width="large"
                            )
                        }
                    )
                else:
                    st.info("No se pudo construir la tabla por tipo de Centro Comercial.")

    # ============================================================
    # Benchmark de densidad de demanda
    # ============================================================

    st.markdown(
        '<div class="section-title">Benchmark de densidad de demanda</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Esta sección calcula la densidad de demanda promedio anual por giro comercial y tarifa. "
        "El universo base es la muestra global validada de locales ocupados con recibo."
    )

    cc_master_path = DATA_DIR / "profiles" / "cc_master_data.csv"

    if not cc_master_path.exists():
        st.warning(
            f"No encontré el archivo `cc_master_data.csv` en la carpeta `data/profiles`: {cc_master_path}"
        )

    elif (
        "benchmark_densidad_base" not in globals()
        or benchmark_densidad_base.empty
    ):
        st.info(
            "No se pudo construir benchmark_densidad_base. "
            "Revisa la construcción de muestra_con_recibo_global y parsed."
        )

    else:
        cc_master_df = pd.read_csv(cc_master_path)
        cc_master_df.columns = cc_master_df.columns.str.strip()



        # ------------------------------------------------------------
        # Base benchmark: usar la base maestra, no recalcular desde cero
        # ------------------------------------------------------------

        benchmark_base = benchmark_densidad_base.copy()

        # ------------------------------------------------------------
        # Centro comercial en benchmark
        # ------------------------------------------------------------

        if "_cc_key_reporte" not in benchmark_base.columns:

            if "_centro_comercial_limpio" in benchmark_base.columns:
                benchmark_base["_cc_key_reporte"] = (
                    benchmark_base["_centro_comercial_limpio"].apply(cc_key)
                )

            elif "NOMBRE DEL CC" in benchmark_base.columns:
                benchmark_base["_cc_key_reporte"] = (
                    benchmark_base["NOMBRE DEL CC"].apply(cc_key)
                )

            elif "source_sheet" in benchmark_base.columns:
                benchmark_base["_cc_key_reporte"] = (
                    benchmark_base["source_sheet"].apply(cc_key)
                )

            else:
                benchmark_base["_cc_key_reporte"] = ""

        # ------------------------------------------------------------
        # Cruzar benchmark contra cc_master_data.csv para clima y tipo de CC
        # ------------------------------------------------------------

        cc_master_cc_col = first_existing_column(
            cc_master_df,
            [
                "Nombre Comercial",
                "centro_comercial",
                "Centro Comercial",
                "NOMBRE DEL CC",
                "Nombre del CC",
                "CC",
                "Plaza",
                "PLAZA"
            ]
        )

        cc_master_tipo_col = first_existing_column(
            cc_master_df,
            [
                "Tipo de Mall",
                "Tipo de centro comercial",
                "tipo_cc",
                "TIPO_CC",
                "Tipo CC"
            ]
        )

        cc_master_zona_col = first_existing_column(
            cc_master_df,
            [
                "zona_nrel",
                "Zona NREL",
                "ZONA_NREL",
                "climate_zone",
                "Climate Zone",
                "CLIMATE_ZONE",
                "zona_climatica",
                "Zona climática"
            ]
        )

        if cc_master_cc_col is None:
            st.info(
                "No pude filtrar benchmark por clima/tipo de CC porque falta la columna de centro comercial en cc_master_data.csv."
            )

        else:
            cc_master_df["_cc_key_reporte"] = (
                cc_master_df[cc_master_cc_col].apply(cc_key)
            )

            # ------------------------------------------------------------
            # Preparar maestro de CC con nombres únicos para evitar columnas
            # duplicadas tipo zona_nrel_x / zona_nrel_y
            # ------------------------------------------------------------

            cc_master_join = cc_master_df.copy()

            cc_master_join["_cc_key_reporte"] = (
                cc_master_join[cc_master_cc_col].apply(cc_key)
            )

            if cc_master_tipo_col:
                cc_master_join["_tipo_cc_benchmark"] = (
                    cc_master_join[cc_master_tipo_col]
                    .fillna("Sin clasificar")
                    .astype(str)
                    .str.strip()
                )
            else:
                cc_master_join["_tipo_cc_benchmark"] = "Sin clasificar"

            if cc_master_zona_col:
                cc_master_join["_zona_nrel_benchmark"] = (
                    cc_master_join[cc_master_zona_col]
                    .fillna("Sin clasificar")
                    .astype(str)
                    .str.strip()
                )
            else:
                cc_master_join["_zona_nrel_benchmark"] = "Sin clasificar"

            benchmark_base = benchmark_base.merge(
                cc_master_join[
                    [
                        "_cc_key_reporte",
                        "_tipo_cc_benchmark",
                        "_zona_nrel_benchmark"
                    ]
                ].drop_duplicates(
                    subset=["_cc_key_reporte"]
                ),
                on="_cc_key_reporte",
                how="left"
            )

        # ------------------------------------------------------------
        # Clasificación de clima
        # ------------------------------------------------------------

        def clasificar_clima_benchmark(zona):
            zona_upper = str(zona).upper()

            if (
                "HOT" in zona_upper
                or "CALIDO" in zona_upper
                or "CÁLIDO" in zona_upper
            ):
                return "Cálido"

            if (
                "MIXED" in zona_upper
                or "TEMPLADO" in zona_upper
            ):
                return "Templado"

            if (
                "COLD" in zona_upper
                or "FRIO" in zona_upper
                or "FRÍO" in zona_upper
            ):
                return "Frío"

            return "Sin clasificar"

        benchmark_base["Clima benchmark"] = (
            benchmark_base["_zona_nrel_benchmark"]
            .fillna("Sin clasificar")
            .apply(clasificar_clima_benchmark)
        )

        benchmark_base["Tipo de centro comercial benchmark"] = (
            benchmark_base["_tipo_cc_benchmark"]
            .fillna("Sin clasificar")
            .astype(str)
            .str.strip()
            .replace({
                "": "Sin clasificar",
                "nan": "Sin clasificar",
                "None": "Sin clasificar"
            })
        )

        # ------------------------------------------------------------
        # Columnas estándar para demanda, densidad, tarifa y giro
        # ------------------------------------------------------------
        # IMPORTANTE:
        # Estas columnas se preparan sobre la base GLOBAL.
        # Las métricas de tamaño de muestra NO deben depender de los dropdowns.

        benchmark_base_global = benchmark_base.copy()

        demanda_col_benchmark = first_existing_column(
            benchmark_base_global,
            [
                "Demanda promedio anual (kW)",
                "demanda_promedio_anual_kw"
            ]
        )

        densidad_col_benchmark = first_existing_column(
            benchmark_base_global,
            [
                "densidad_demanda_promedio_anual_w_m2",
                "Densidad de demanda W/m2"
            ]
        )

        tarifa_col_benchmark = first_existing_column(
            benchmark_base_global,
            [
                "Tarifa",
                "tarifa_norm",
                "TARIFA_FINAL"
            ]
        )

        giro_col_benchmark = first_existing_column(
            benchmark_base_global,
            [
                "Giro comercial densidad",
                "SUBGIRO_COMERCIAL",
                "SUBGIRO COMERCIAL",
                "GIRO_COMERCIAL",
                "GIRO COMERCIAL",
                "GIRO",
                "Giro"
            ]
        )

        if demanda_col_benchmark is None:
            benchmark_base_global["demanda_benchmark_kw"] = pd.NA
        else:
            benchmark_base_global["demanda_benchmark_kw"] = pd.to_numeric(
                benchmark_base_global[demanda_col_benchmark],
                errors="coerce"
            )

        if densidad_col_benchmark is None:
            benchmark_base_global["densidad_benchmark_w_m2"] = pd.NA
        else:
            benchmark_base_global["densidad_benchmark_w_m2"] = pd.to_numeric(
                benchmark_base_global[densidad_col_benchmark],
                errors="coerce"
            )

        if tarifa_col_benchmark is None:
            benchmark_base_global["tarifa_benchmark"] = "SIN TARIFA"
        else:
            benchmark_base_global["tarifa_benchmark"] = (
                normalize_tarifa_series(
                    benchmark_base_global[tarifa_col_benchmark]
                )
                .fillna("SIN TARIFA")
                .astype(str)
                .str.upper()
                .str.strip()
            )

        if giro_col_benchmark is None:
            benchmark_base_global["giro_benchmark"] = "Sin giro"
        else:
            benchmark_base_global["giro_benchmark"] = (
                benchmark_base_global[giro_col_benchmark]
                .fillna("Sin giro")
                .astype(str)
                .str.strip()
                .replace({
                    "": "Sin giro",
                    "nan": "Sin giro",
                    "None": "Sin giro"
                })
            )

        # ------------------------------------------------------------
        # Tamaño de muestra GLOBAL para benchmark
        # ------------------------------------------------------------
        # Estas métricas se calculan sobre TODA la base de densidad,
        # es decir, todos los locales ocupados con recibo de los 19 CCs.
        # No dependen de clima_selector ni tipo_cc_selector.
        #
        # Conciliación esperada:
        # Servicios Generales
        # + Locales con densidad calculable
        # + Locales pendientes por demanda
        # + Locales sin densidad por área
        # = Locales ocupados con recibo

        mask_servicios_generales_global = (
            benchmark_base_global["giro_benchmark"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .str.contains("SERVICIOS GENERALES", na=False)
        )

        # Locales ocupados con recibo:
        # debe venir del match global / coverage_by_mall,
        # no del largo de benchmark_base_global, porque esa base puede perder
        # algún registro si no logró enriquecer demanda.
        if (
            "locales_ocupados_con_recibo_total" in locals()
            and pd.notna(locales_ocupados_con_recibo_total)
        ):
            locales_ocupados_con_recibo_benchmark_global = int(
                locales_ocupados_con_recibo_total
            )

        elif "coverage_by_mall" in globals() and not coverage_by_mall.empty:

            coverage_total_row_benchmark = coverage_by_mall[
                coverage_by_mall["Centro Comercial"]
                .astype(str)
                .str.upper()
                .str.strip()
                .eq("TOTAL")
            ]

            if not coverage_total_row_benchmark.empty:
                locales_ocupados_con_recibo_benchmark_global = int(
                    coverage_total_row_benchmark.iloc[0].get(
                        "Locales ocupados con recibo",
                        len(benchmark_base_global)
                    )
                )
            else:
                locales_ocupados_con_recibo_benchmark_global = len(
                    benchmark_base_global
                )

        else:
            locales_ocupados_con_recibo_benchmark_global = len(
                benchmark_base_global
            )

        servicios_generales_global = int(
            mask_servicios_generales_global.sum()
        )

        # ------------------------------------------------------------
        # Locales con densidad calculable
        # ------------------------------------------------------------
        # Excluimos Servicios Generales porque se reportan aparte.

        mask_densidad_calculable_global = (
            ~mask_servicios_generales_global
            & benchmark_base_global["densidad_benchmark_w_m2"].notna()
            & (benchmark_base_global["densidad_benchmark_w_m2"] > 0)
            & benchmark_base_global["demanda_benchmark_kw"].notna()
            & (benchmark_base_global["demanda_benchmark_kw"] > 0)
        )

        benchmark_calculable_global = benchmark_base_global[
            mask_densidad_calculable_global
        ].copy()

        locales_con_densidad_calculable_global = int(
            mask_densidad_calculable_global.sum()
        )

        # ------------------------------------------------------------
        # Locales pendientes por demanda
        # ------------------------------------------------------------
        # Excluimos Servicios Generales porque se reportan aparte.

        mask_pendiente_demanda_global = (
            ~mask_servicios_generales_global
            & (
                benchmark_base_global["demanda_benchmark_kw"].isna()
                | (benchmark_base_global["demanda_benchmark_kw"] <= 0)
            )
        )

        locales_pendientes_demanda_global = int(
            mask_pendiente_demanda_global.sum()
        )

        # ------------------------------------------------------------
        # Locales sin densidad por área
        # ------------------------------------------------------------
        # Son locales que:
        # - NO son Servicios Generales
        # - SÍ tienen demanda calculada
        # - Pero NO tienen densidad calculable
        #
        # Normalmente esto debería deberse a:
        # - área m² vacía
        # - área m² igual a cero
        # - área m² no numérica
        # - densidad no calculada aunque haya demanda

        area_col_benchmark = first_existing_column(
            benchmark_base_global,
            [
                "Area m2",
                "m2_num",
                "MTS2",
                "M2",
                "m2",
                "Área",
                "Area",
                "AREA",
                "Superficie",
                "SUPERFICIE",
                "MTS 2",
                "MTS²",
                "AREA_M2",
                "AREA M2",
                "SUPERFICIE M2"
            ]
        )

        if area_col_benchmark:
            benchmark_base_global["area_benchmark_m2"] = pd.to_numeric(
                benchmark_base_global[area_col_benchmark],
                errors="coerce"
            )
        else:
            benchmark_base_global["area_benchmark_m2"] = pd.NA

        mask_sin_densidad_por_area_global = (
            ~mask_servicios_generales_global
            & benchmark_base_global["demanda_benchmark_kw"].notna()
            & (benchmark_base_global["demanda_benchmark_kw"] > 0)
            & (
                benchmark_base_global["densidad_benchmark_w_m2"].isna()
                | (benchmark_base_global["densidad_benchmark_w_m2"] <= 0)
            )
        )

        locales_sin_densidad_por_area_global = int(
            mask_sin_densidad_por_area_global.sum()
        )

        diagnostico_sin_densidad_area = benchmark_base_global[
            mask_sin_densidad_por_area_global
        ].copy()

        diagnostico_sin_densidad_area["Motivo"] = "Sin densidad calculable"

        diagnostico_sin_densidad_area.loc[
            diagnostico_sin_densidad_area["area_benchmark_m2"].isna(),
            "Motivo"
        ] = "Sin área m² válida"

        diagnostico_sin_densidad_area.loc[
            diagnostico_sin_densidad_area["area_benchmark_m2"].notna()
            & (diagnostico_sin_densidad_area["area_benchmark_m2"] <= 0),
            "Motivo"
        ] = "Área m² igual a cero o negativa"

        diagnostico_sin_densidad_area.loc[
            diagnostico_sin_densidad_area["area_benchmark_m2"].notna()
            & (diagnostico_sin_densidad_area["area_benchmark_m2"] > 0)
            & (
                diagnostico_sin_densidad_area["densidad_benchmark_w_m2"].isna()
                | (diagnostico_sin_densidad_area["densidad_benchmark_w_m2"] <= 0)
            ),
            "Motivo"
        ] = "Tiene área y demanda, pero densidad no calculada"

        # ------------------------------------------------------------
        # Métricas
        # ------------------------------------------------------------

        st.markdown(
            '<div class="subsection-title">Tamaño de muestra para benchmark</div>',
            unsafe_allow_html=True
        )

        col_a, col_b, col_c, col_d, col_e = st.columns(5)

        col_a.metric(
            "Locales ocupados con recibo",
            f"{locales_ocupados_con_recibo_benchmark_global:,}"
        )

        col_b.metric(
            "No. de Servicios Generales",
            f"{servicios_generales_global:,}"
        )

        col_c.metric(
            "Locales con densidad calculable",
            f"{locales_con_densidad_calculable_global:,}"
        )

        col_d.metric(
            "Locales pendientes por demanda",
            f"{locales_pendientes_demanda_global:,}"
        )

        col_e.metric(
            "Locales sin densidad por área",
            f"{locales_sin_densidad_por_area_global:,}"
        )

        st.caption(
            "El tamaño de muestra se calcula sobre toda la base de densidad de demanda: "
            "todos los locales ocupados con recibo de los 19 centros comerciales. "
            "Los filtros de clima y tipo de centro comercial solo afectan la tabla inferior. "
            "La suma de Servicios Generales, locales con densidad calculable, "
            "locales pendientes por demanda y locales sin densidad por área debe coincidir "
            "con el total de locales ocupados con recibo."
        )

        with st.expander(
            "Diagnóstico: locales sin densidad por área",
            expanded=False
        ):

            if diagnostico_sin_densidad_area.empty:
                st.success(
                    "No hay locales sin densidad por área."
                )

            else:
                cols_diag_area = [
                    col for col in [
                        "_centro_comercial_limpio",
                        "NOMBRE DEL CC",
                        "Centro Comercial",
                        "CLIENTE",
                        "NOMBRE COMERCIAL",
                        "Giro comercial densidad",
                        "SUBGIRO_COMERCIAL",
                        "GIRO_COMERCIAL",
                        "Tarifa",
                        "tarifa_benchmark",
                        "no_servicio",
                        "parser_no_servicio_match",
                        "medidor",
                        "parser_medidor_match",
                        "Demanda promedio anual (kW)",
                        "demanda_benchmark_kw",
                        "Area m2",
                        "area_benchmark_m2",
                        "Densidad de demanda W/m2",
                        "densidad_benchmark_w_m2",
                        "criterio_union_demanda",
                        "estatus_demanda",
                        "Motivo"
                    ]
                    if col in diagnostico_sin_densidad_area.columns
                ]

                st.warning(
                    f"{len(diagnostico_sin_densidad_area):,} locales tienen demanda calculada, "
                    "pero no tienen densidad calculable."
                )

                st.dataframe(
                    diagnostico_sin_densidad_area[cols_diag_area],
                    use_container_width=True,
                    hide_index=True,
                    height=500
                )

        # ------------------------------------------------------------
        # Diagnóstico Servicios Generales
        # ------------------------------------------------------------

        with st.expander(
            "Diagnóstico Servicios Generales: DG vs match global vs benchmark",
            expanded=False
        ):

            def texto_ocupado_valido_debug(series):
                texto = (
                    series
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

                return (
                    texto.ne("")
                    & ~texto.isin([
                        "NAN",
                        "NONE",
                        "<NA>",
                        "DISPONIBLE",
                        "LOCAL DISPONIBLE",
                        "VACANTE"
                    ])
                    & ~texto.str.contains("DISPONIBLE", na=False)
                    & ~texto.str.contains("VACANTE", na=False)
                )

            def normalizar_debug(value):
                if pd.isna(value):
                    return ""

                return (
                    str(value)
                    .upper()
                    .strip()
                    .replace("Á", "A")
                    .replace("É", "E")
                    .replace("Í", "I")
                    .replace("Ó", "O")
                    .replace("Ú", "U")
                    .replace("Ñ", "N")
                )

            def crear_key_local_debug(df_debug):
                df_debug = df_debug.copy()

                cc_col_debug = first_existing_column(
                    df_debug,
                    [
                        "_centro_comercial_limpio",
                        "NOMBRE DEL CC",
                        "CENTRO COMERCIAL",
                        "Centro Comercial",
                        "CC",
                        "PLAZA",
                        "source_sheet"
                    ]
                )

                cliente_col_debug = first_existing_column(
                    df_debug,
                    [
                        "CLIENTE",
                        "Cliente",
                        "cliente",
                        "cliente_nombre"
                    ]
                )

                nombre_col_debug = first_existing_column(
                    df_debug,
                    [
                        "NOMBRE COMERCIAL",
                        "Nombre Comercial",
                        "Nombre comercial",
                        "recibos_subgroup",
                        "tenant",
                        "locatario"
                    ]
                )

                local_col_debug = first_existing_column(
                    df_debug,
                    [
                        "No de Local",
                        "No. de Local",
                        "No Local",
                        "LOCAL",
                        "Local"
                    ]
                )

                medidor_col_debug = first_existing_column(
                    df_debug,
                    [
                        "No. De medidor",
                        "No. de medidor",
                        "No de medidor",
                        "MEDIDOR",
                        "Medidor",
                        "medidor",
                        "parser_medidor_match"
                    ]
                )

                if cc_col_debug:
                    cc_key_debug = df_debug[cc_col_debug].apply(cc_key)
                else:
                    cc_key_debug = pd.Series(
                        "",
                        index=df_debug.index
                    )

                if cliente_col_debug:
                    cliente_key_debug = df_debug[cliente_col_debug].apply(
                        normalizar_debug
                    )
                else:
                    cliente_key_debug = pd.Series(
                        "",
                        index=df_debug.index
                    )

                if nombre_col_debug:
                    nombre_key_debug = df_debug[nombre_col_debug].apply(
                        normalizar_debug
                    )
                else:
                    nombre_key_debug = pd.Series(
                        "",
                        index=df_debug.index
                    )

                if local_col_debug:
                    local_key_debug = df_debug[local_col_debug].apply(
                        normalizar_debug
                    )
                else:
                    local_key_debug = pd.Series(
                        "",
                        index=df_debug.index
                    )

                if medidor_col_debug:
                    medidor_key_debug = df_debug[medidor_col_debug].apply(
                        normalizar_debug
                    )
                else:
                    medidor_key_debug = pd.Series(
                        "",
                        index=df_debug.index
                    )

                return (
                    cc_key_debug.astype(str)
                    + " | "
                    + cliente_key_debug.astype(str)
                    + " | "
                    + nombre_key_debug.astype(str)
                    + " | "
                    + local_key_debug.astype(str)
                    + " | "
                    + medidor_key_debug.astype(str)
                )

            # --------------------------------------------------------
            # 1. Servicios Generales ocupados en DG
            # --------------------------------------------------------

            sg_dg_debug = general_data.copy()

            giro_col_sg_debug = first_existing_column(
                sg_dg_debug,
                [
                    "SUBGIRO_COMERCIAL",
                    "SUBGIRO COMERCIAL",
                    "GIRO_COMERCIAL",
                    "GIRO COMERCIAL",
                    "GIRO",
                    "Giro",
                    "Giro comercial"
                ]
            )

            cliente_col_sg_debug = first_existing_column(
                sg_dg_debug,
                [
                    "CLIENTE",
                    "Cliente",
                    "cliente",
                    "cliente_nombre"
                ]
            )

            nombre_col_sg_debug = first_existing_column(
                sg_dg_debug,
                [
                    "NOMBRE COMERCIAL",
                    "Nombre Comercial",
                    "Nombre comercial",
                    "recibos_subgroup",
                    "tenant",
                    "locatario"
                ]
            )

            if giro_col_sg_debug is None:
                st.warning(
                    "No encontré columna de giro en DG para diagnosticar Servicios Generales."
                )

            else:
                giro_text_sg_debug = (
                    sg_dg_debug[giro_col_sg_debug]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

                mask_sg_dg_debug = giro_text_sg_debug.str.contains(
                    r"SERVICIO[S]?\s+GENERAL(ES)?",
                    regex=True,
                    na=False
                )

                if cliente_col_sg_debug:
                    cliente_ocupado_debug = texto_ocupado_valido_debug(
                        sg_dg_debug[cliente_col_sg_debug]
                    )
                else:
                    cliente_ocupado_debug = pd.Series(
                        False,
                        index=sg_dg_debug.index
                    )

                if nombre_col_sg_debug:
                    nombre_ocupado_debug = texto_ocupado_valido_debug(
                        sg_dg_debug[nombre_col_sg_debug]
                    )
                else:
                    nombre_ocupado_debug = pd.Series(
                        False,
                        index=sg_dg_debug.index
                    )

                mask_ocupado_dg_debug = (
                    cliente_ocupado_debug
                    | nombre_ocupado_debug
                )

                sg_dg_ocupados_debug = sg_dg_debug[
                    mask_sg_dg_debug
                    & mask_ocupado_dg_debug
                ].copy()

                sg_dg_ocupados_debug["_key_local_debug"] = crear_key_local_debug(
                    sg_dg_ocupados_debug
                )

                # ----------------------------------------------------
                # 2. Servicios Generales con match global
                # ----------------------------------------------------

                if (
                    "muestra_con_recibo_global" in globals()
                    and not muestra_con_recibo_global.empty
                ):
                    muestra_sg_debug = muestra_con_recibo_global.copy()

                    giro_col_muestra_sg_debug = first_existing_column(
                        muestra_sg_debug,
                        [
                            "SUBGIRO_COMERCIAL",
                            "SUBGIRO COMERCIAL",
                            "GIRO_COMERCIAL",
                            "GIRO COMERCIAL",
                            "GIRO",
                            "Giro",
                            "Giro comercial",
                            "Giro comercial densidad"
                        ]
                    )

                    if giro_col_muestra_sg_debug:
                        mask_sg_muestra_debug = (
                            muestra_sg_debug[giro_col_muestra_sg_debug]
                            .fillna("")
                            .astype(str)
                            .str.upper()
                            .str.strip()
                            .str.contains(
                                r"SERVICIO[S]?\s+GENERAL(ES)?",
                                regex=True,
                                na=False
                            )
                        )
                    else:
                        mask_sg_muestra_debug = pd.Series(
                            False,
                            index=muestra_sg_debug.index
                        )

                    muestra_sg_debug = muestra_sg_debug[
                        mask_sg_muestra_debug
                    ].copy()

                    muestra_sg_debug["_key_local_debug"] = crear_key_local_debug(
                        muestra_sg_debug
                    )

                    keys_sg_muestra_debug = set(
                        muestra_sg_debug["_key_local_debug"]
                        .dropna()
                        .astype(str)
                    )

                else:
                    muestra_sg_debug = pd.DataFrame()
                    keys_sg_muestra_debug = set()

                # ----------------------------------------------------
                # 3. Servicios Generales contados en benchmark
                # ----------------------------------------------------

                benchmark_sg_debug = benchmark_base_global.copy()

                if "giro_benchmark" in benchmark_sg_debug.columns:
                    mask_sg_benchmark_debug = (
                        benchmark_sg_debug["giro_benchmark"]
                        .fillna("")
                        .astype(str)
                        .str.upper()
                        .str.strip()
                        .str.contains(
                            r"SERVICIO[S]?\s+GENERAL(ES)?",
                            regex=True,
                            na=False
                        )
                    )
                else:
                    mask_sg_benchmark_debug = pd.Series(
                        False,
                        index=benchmark_sg_debug.index
                    )

                benchmark_sg_debug = benchmark_sg_debug[
                    mask_sg_benchmark_debug
                ].copy()

                benchmark_sg_debug["_key_local_debug"] = crear_key_local_debug(
                    benchmark_sg_debug
                )

                keys_sg_benchmark_debug = set(
                    benchmark_sg_debug["_key_local_debug"]
                    .dropna()
                    .astype(str)
                )

                # ----------------------------------------------------
                # 4. Marcar estatus en DG
                # ----------------------------------------------------

                sg_dg_ocupados_debug["Tiene match global"] = (
                    sg_dg_ocupados_debug["_key_local_debug"]
                    .astype(str)
                    .isin(keys_sg_muestra_debug)
                )

                sg_dg_ocupados_debug["Contado en benchmark como SG"] = (
                    sg_dg_ocupados_debug["_key_local_debug"]
                    .astype(str)
                    .isin(keys_sg_benchmark_debug)
                )

                sg_dg_ocupados_debug["Estatus"] = "OK"

                sg_dg_ocupados_debug.loc[
                    ~sg_dg_ocupados_debug["Tiene match global"],
                    "Estatus"
                ] = "Sin match global"

                sg_dg_ocupados_debug.loc[
                    sg_dg_ocupados_debug["Tiene match global"]
                    & ~sg_dg_ocupados_debug["Contado en benchmark como SG"],
                    "Estatus"
                ] = "Con match, pero no contado como SG en benchmark"

                col_diag_1, col_diag_2, col_diag_3 = st.columns(3)

                col_diag_1.metric(
                    "SG ocupados en DG",
                    f"{len(sg_dg_ocupados_debug):,}"
                )

                col_diag_2.metric(
                    "SG con match global",
                    f"{len(muestra_sg_debug):,}"
                )

                col_diag_3.metric(
                    "SG contados en benchmark",
                    f"{len(benchmark_sg_debug):,}"
                )

                # ----------------------------------------------------
                # 5. Tabla de diagnóstico
                # ----------------------------------------------------

                cols_diag_sg = [
                    col for col in [
                        "source_sheet",
                        "_centro_comercial_limpio",
                        "NOMBRE DEL CC",
                        "CLIENTE",
                        "NOMBRE COMERCIAL",
                        "No de Local",
                        "No. de Local",
                        "No. De medidor",
                        "No. de medidor",
                        giro_col_sg_debug,
                        "Tiene match global",
                        "Contado en benchmark como SG",
                        "Estatus"
                    ]
                    if col in sg_dg_ocupados_debug.columns
                ]

                st.dataframe(
                    sg_dg_ocupados_debug[cols_diag_sg],
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )

        # ------------------------------------------------------------
        # Selectores
        # ------------------------------------------------------------
        # Agregamos "Todos" para que el benchmark pueda mostrar la base global
        # o filtrarse por clima / tipo de centro comercial.

        clima_selector = st.selectbox(
            "Selecciona clima",
            ["Todos", "Cálido", "Templado", "Frío"],
            key="benchmark_clima"
        )

        tipo_cc_selector = st.selectbox(
            "Selecciona tipo de centro comercial",
            [
                "Todos",
                "Luxury Fashion Mall",
                "Fashion Mall",
                "Regional Mall",
                "Power Center",
                "Strip Mall"
            ],
            key="benchmark_tipo_cc"
        )

        # Opciones dinámicas para giro y tarifa.
        # Siempre ponemos "Todos" como primera opción.
        giros_benchmark_opciones = (
            benchmark_base_global["giro_benchmark"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
            if "giro_benchmark" in benchmark_base_global.columns
            else []
        )

        tarifas_benchmark_opciones = (
            benchmark_base_global["tarifa_benchmark"]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .sort_values()
            .unique()
            .tolist()
            if "tarifa_benchmark" in benchmark_base_global.columns
            else []
        )

        giro_selector = st.selectbox(
            "Selecciona giro comercial",
            ["Todos"] + giros_benchmark_opciones,
            key="benchmark_giro"
        )

        tarifa_selector = st.selectbox(
            "Selecciona tarifa",
            ["Todos"] + tarifas_benchmark_opciones,
            key="benchmark_tarifa"
        )

        # ------------------------------------------------------------
        # Aplicar filtros de dropdown SOLO para la tabla
        # ------------------------------------------------------------

        benchmark_filtrado = benchmark_base_global.copy()

        if clima_selector != "Todos":
            benchmark_filtrado = benchmark_filtrado[
                benchmark_filtrado["Clima benchmark"].eq(clima_selector)
            ].copy()

        if tipo_cc_selector != "Todos":
            benchmark_filtrado = benchmark_filtrado[
                benchmark_filtrado["Tipo de centro comercial benchmark"].eq(
                    tipo_cc_selector
                )
            ].copy()

        if giro_selector != "Todos":
            benchmark_filtrado = benchmark_filtrado[
                benchmark_filtrado["giro_benchmark"].eq(giro_selector)
            ].copy()

        if tarifa_selector != "Todos":
            benchmark_filtrado = benchmark_filtrado[
                benchmark_filtrado["tarifa_benchmark"].eq(tarifa_selector)
            ].copy()

        benchmark_calculable = benchmark_filtrado[
            ~benchmark_filtrado["giro_benchmark"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
            .str.contains("SERVICIOS GENERALES", na=False)
            & benchmark_filtrado["densidad_benchmark_w_m2"].notna()
            & (benchmark_filtrado["densidad_benchmark_w_m2"] > 0)
            & benchmark_filtrado["demanda_benchmark_kw"].notna()
            & (benchmark_filtrado["demanda_benchmark_kw"] > 0)
        ].copy()

        # ------------------------------------------------------------
        # Resumen de densidad por giro y tarifa
        # ------------------------------------------------------------

        if benchmark_calculable.empty:
            st.info(
                "No hay locales con densidad calculable para la selección de clima y tipo de centro comercial."
            )

        else:
            st.markdown(
                '<div class="subsection-title">Resumen de densidad por giro y tarifa</div>',
                unsafe_allow_html=True
            )

            resumen_benchmark = (
                benchmark_calculable
                .groupby(
                    [
                        "giro_benchmark",
                        "tarifa_benchmark"
                    ],
                    dropna=False
                )
                .agg(
                    locales=(
                        "densidad_benchmark_w_m2",
                        "count"
                    ),
                    area_promedio_m2=(
                        "area_benchmark_m2",
                        "mean"
                    ),
                    demanda_promedio_anual_promedio_kw=(
                        "demanda_benchmark_kw",
                        "mean"
                    ),
                    densidad_promedio_w_m2=(
                        "densidad_benchmark_w_m2",
                        "mean"
                    ),
                    densidad_mediana_w_m2=(
                        "densidad_benchmark_w_m2",
                        "median"
                    ),
                    p25_w_m2=(
                        "densidad_benchmark_w_m2",
                        lambda x: x.quantile(0.25)
                    ),
                    p75_w_m2=(
                        "densidad_benchmark_w_m2",
                        lambda x: x.quantile(0.75)
                    ),
                    p90_w_m2=(
                        "densidad_benchmark_w_m2",
                        lambda x: x.quantile(0.90)
                    )
                )
                .reset_index()
                .rename(columns={
                    "giro_benchmark": "Giro comercial",
                    "tarifa_benchmark": "Tarifa",
                    "locales": "No. de locales",
                    "area_promedio_m2": "Área promedio (m²)",
                    "demanda_promedio_anual_promedio_kw": "Demanda promedio anual (kW)",
                    "densidad_promedio_w_m2": "Densidad promedio (W/m²)",
                    "densidad_mediana_w_m2": "Densidad mediana (W/m²)",
                    "p25_w_m2": "P25 (W/m²)",
                    "p75_w_m2": "P75 (W/m²)",
                    "p90_w_m2": "P90 (W/m²)"
                })
            )

            for col in [
                "Área promedio (m²)",
                "Demanda promedio anual (kW)",
                "Densidad promedio (W/m²)",
                "Densidad mediana (W/m²)",
                "P25 (W/m²)",
                "P75 (W/m²)",
                "P90 (W/m²)"
            ]:
                resumen_benchmark[col] = pd.to_numeric(
                    resumen_benchmark[col],
                    errors="coerce"
                ).round(1)

            resumen_benchmark = resumen_benchmark.sort_values(
                [
                    "Tarifa",
                    "Densidad promedio (W/m²)"
                ],
                ascending=[
                    True,
                    False
                ]
            )

            resumen_benchmark = resumen_benchmark[
                [
                    "Giro comercial",
                    "Tarifa",
                    "No. de locales",
                    "Área promedio (m²)",
                    "Demanda promedio anual (kW)",
                    "Densidad promedio (W/m²)",
                    "Densidad mediana (W/m²)",
                    "P25 (W/m²)",
                    "P75 (W/m²)",
                    "P90 (W/m²)"
                ]
            ].copy()

            st.dataframe(
                resumen_benchmark,
                use_container_width=True,
                hide_index=True
            )

            # ------------------------------------------------------------
            # Detalle de locales según filtros seleccionados
            # ------------------------------------------------------------

            st.markdown(
                '<div class="subsection-title">Detalle de locales filtrados</div>',
                unsafe_allow_html=True
            )

            detalle_locales_filtrado = benchmark_filtrado.copy()

            # Excluir Servicios Generales igual que en el resumen de benchmark.
            detalle_locales_filtrado = detalle_locales_filtrado[
                ~detalle_locales_filtrado["giro_benchmark"]
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
                .str.contains("SERVICIOS GENERALES", na=False)
            ].copy()

            # Si quieres que esta tabla muestre solo locales con densidad calculable,
            # deja activo este bloque. Si quieres que muestre todos los locales con recibo
            # aunque tengan demanda/densidad pendiente, déjalo comentado.
            #
            # detalle_locales_filtrado = detalle_locales_filtrado[
            #     detalle_locales_filtrado["demanda_benchmark_kw"].notna()
            #     & (detalle_locales_filtrado["demanda_benchmark_kw"] > 0)
            #     & detalle_locales_filtrado["densidad_benchmark_w_m2"].notna()
            #     & (detalle_locales_filtrado["densidad_benchmark_w_m2"] > 0)
            # ].copy()

            centro_col_detalle = first_existing_column(
                detalle_locales_filtrado,
                [
                    "_centro_comercial_limpio",
                    "NOMBRE DEL CC",
                    "Centro Comercial",
                    "CENTRO COMERCIAL",
                    "source_sheet"
                ]
            )

            nombre_col_detalle = first_existing_column(
                detalle_locales_filtrado,
                [
                    "NOMBRE COMERCIAL",
                    "Nombre Comercial",
                    "CLIENTE",
                    "cliente_nombre",
                    "recibos_subgroup"
                ]
            )

            servicio_col_detalle = first_existing_column(
                detalle_locales_filtrado,
                [
                    "no_servicio",
                    "No. servicio",
                    "No servicio",
                    "parser_no_servicio_match"
                ]
            )

            detalle_tabla = pd.DataFrame()

            detalle_tabla["Centro comercial"] = (
                detalle_locales_filtrado[centro_col_detalle]
                if centro_col_detalle
                else pd.NA
            )

            detalle_tabla["Nombre Comercial"] = (
                detalle_locales_filtrado[nombre_col_detalle]
                if nombre_col_detalle
                else pd.NA
            )

            detalle_tabla["No. de servicio"] = (
                detalle_locales_filtrado[servicio_col_detalle]
                if servicio_col_detalle
                else pd.NA
            )

            detalle_tabla["Área (m²)"] = pd.to_numeric(
                detalle_locales_filtrado["area_benchmark_m2"],
                errors="coerce"
            )

            detalle_tabla["Demanda promedio anual (kW)"] = pd.to_numeric(
                detalle_locales_filtrado["demanda_benchmark_kw"],
                errors="coerce"
            )

            if "demanda_pdbt_nrel_original_kw" in detalle_locales_filtrado.columns:
                detalle_tabla["Demanda NREL original (kW)"] = pd.to_numeric(
                    detalle_locales_filtrado["demanda_pdbt_nrel_original_kw"],
                    errors="coerce"
                )

            if "factor_ajuste_allux_pdbt" in detalle_locales_filtrado.columns:
                detalle_tabla["Factor ajuste Allux PDBT"] = pd.to_numeric(
                    detalle_locales_filtrado["factor_ajuste_allux_pdbt"],
                    errors="coerce"
                )

            if "nivel_factor_ajuste_allux_pdbt" in detalle_locales_filtrado.columns:
                detalle_tabla["Nivel factor Allux"] = (
                    detalle_locales_filtrado["nivel_factor_ajuste_allux_pdbt"]
                )

            detalle_tabla["Densidad promedio (W/m²)"] = pd.to_numeric(
                detalle_locales_filtrado["densidad_benchmark_w_m2"],
                errors="coerce"
            )

            detalle_tabla = (
                detalle_tabla
                .drop_duplicates()
                .sort_values(
                    [
                        "Centro comercial",
                        "Nombre Comercial",
                        "No. de servicio"
                    ],
                    na_position="last"
                )
                .reset_index(drop=True)
            )

            for col_num in [
                "Área (m²)",
                "Demanda promedio anual (kW)",
                "Demanda NREL original (kW)",
                "Factor ajuste Allux PDBT",
                "Densidad promedio (W/m²)"
            ]:
                if col_num in detalle_tabla.columns:
                    detalle_tabla[col_num] = pd.to_numeric(
                        detalle_tabla[col_num],
                        errors="coerce"
                    ).round(2)

            st.caption(
                f"{len(detalle_tabla):,} locales coinciden con los filtros seleccionados."
            )

            st.dataframe(
                detalle_tabla,
                use_container_width=True,
                hide_index=True,
                height=500
            )

            st.caption(
                "La Demanda promedio anual (kW) se calcula como el promedio de la demanda anual "
                "de los locales dentro del grupo seleccionado. "
                "El Área promedio (m²) se calcula como el promedio del área de esos mismos locales. "
                "La Densidad promedio (W/m²) se calcula como el promedio de la densidad individual "
                "de cada local, no como demanda agregada / área agregada. "
                + NOTA_DEMANDA_PROMEDIO_ANUAL
            )



with tab_general:
    st.markdown(
        '<div class="section-title">Distribución MT vs BT</div>',
        unsafe_allow_html=True
    )

    universo_usuarios = (
        muestra_con_recibo_global.copy()
        if "muestra_con_recibo_global" in locals()
        and not muestra_con_recibo_global.empty
        else general_data.copy()
    )

    universo_usuarios["Tarifa análisis"] = (
        universo_usuarios["TARIFA_ANALISIS"]
        if "TARIFA_ANALISIS" in universo_usuarios.columns
        else universo_usuarios["TARIFA"]
    )

    universo_usuarios["Nivel de tensión"] = universo_usuarios["Tarifa análisis"].apply(
        lambda x: "MT" if str(x).upper().strip() in ["GDMTH", "GDMTO"] else (
            "BT" if str(x).upper().strip() in ["PDBT", "GDBT"] else "Sin clasificar"
        )
    )

    usuarios_mt_bt = (
        universo_usuarios[
            universo_usuarios["Nivel de tensión"].isin(["MT", "BT"])
        ]
        .groupby("Nivel de tensión")
        .size()
        .reset_index(name="Número de usuarios")
    )

    total_usuarios_mt_bt = usuarios_mt_bt["Número de usuarios"].sum()

    usuarios_mt_bt["(%)"] = (
        usuarios_mt_bt["Número de usuarios"] / total_usuarios_mt_bt * 100
    ).round(1)

    usuarios_mt_bt_display = usuarios_mt_bt.copy()
    usuarios_mt_bt_display["(%)"] = usuarios_mt_bt_display["(%)"].map(lambda x: f"{x:.1f}%")

    col_usuarios, col_demanda = st.columns(2)

    with col_usuarios:
        st.markdown(
            '<div class="subsection-title">Usuarios por nivel de tensión</div>',
            unsafe_allow_html=True
        )

        fig_usuarios = usuarios_mt_bt.set_index("Nivel de tensión").plot.pie(
            y="Número de usuarios",
            autopct="%1.1f%%",
            figsize=(4, 4),
            legend=False
        ).figure

        st.pyplot(fig_usuarios)

        st.dataframe(
            usuarios_mt_bt_display,
            use_container_width=True
        )

    with col_demanda:
        st.markdown(
            '<div class="subsection-title">Demanda promedio anual (kW) por nivel de tensión</div>',
            unsafe_allow_html=True
        )

        if "benchmark_densidad_base" not in globals() or benchmark_densidad_base.empty:
            st.info(
                "No está disponible la base validada para calcular Demanda promedio anual (kW) por nivel de tensión."
            )

        else:
            demanda_portafolio = benchmark_densidad_base.copy()

            tarifa_col = first_existing_column(
                demanda_portafolio,
                [
                    "Tarifa",
                    "tarifa",
                    "tarifa_norm",
                    "TARIFA_FINAL",
                    "tarifa_benchmark"
                ]
            )

            demanda_promedio_col = first_existing_column(
                demanda_portafolio,
                [
                    "Demanda promedio anual (kW)",
                    "demanda_promedio_anual_kw"
                ]
            )

            if tarifa_col is None:
                st.info(
                    "No encontré columna de tarifa en la base validada para clasificar BT/MT."
                )

            elif demanda_promedio_col is None:
                st.info(
                    "No encontré columna de Demanda promedio anual (kW) en la base validada."
                )

            else:
                demanda_portafolio["Tarifa nivel tensión"] = (
                    demanda_portafolio[tarifa_col]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

                demanda_portafolio["Nivel de tensión"] = pd.NA

                demanda_portafolio.loc[
                    demanda_portafolio["Tarifa nivel tensión"].isin(["PDBT", "GDBT"]),
                    "Nivel de tensión"
                ] = "BT"

                demanda_portafolio.loc[
                    demanda_portafolio["Tarifa nivel tensión"].isin(["GDMTH", "GDMTO"]),
                    "Nivel de tensión"
                ] = "MT"

                demanda_portafolio = demanda_portafolio[
                    demanda_portafolio["Nivel de tensión"].notna()
                ].copy()

                demanda_portafolio = demanda_portafolio[
                    demanda_portafolio["Demanda promedio anual (kW)"].notna()
                    & demanda_portafolio["Demanda promedio anual (kW)"].gt(0)
                ].copy()

                if demanda_portafolio.empty:
                    st.info(
                        "No hay Demanda promedio anual (kW) calculable para graficar por nivel de tensión."
                    )

                else:
                    demanda_mt_bt = (
                        demanda_portafolio
                        .groupby("Nivel de tensión", dropna=False)["Demanda promedio anual (kW)"]
                        .sum()
                        .reset_index()
                    )

                    total_demanda_mt_bt = demanda_mt_bt[
                        "Demanda promedio anual (kW)"
                    ].sum(skipna=True)

                    demanda_mt_bt["(%)"] = (
                        demanda_mt_bt["Demanda promedio anual (kW)"]
                        / total_demanda_mt_bt
                        * 100
                        if total_demanda_mt_bt > 0
                        else pd.NA
                    )

                    demanda_mt_bt_display = demanda_mt_bt.copy()

                    demanda_mt_bt_display["Demanda promedio anual (kW)"] = (
                        demanda_mt_bt_display["Demanda promedio anual (kW)"]
                        .round(1)
                    )

                    demanda_mt_bt_display["(%)"] = (
                        demanda_mt_bt_display["(%)"]
                        .map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
                    )

                    fig_demanda = demanda_mt_bt.set_index("Nivel de tensión").plot.pie(
                        y="Demanda promedio anual (kW)",
                        autopct="%1.1f%%",
                        figsize=(4, 4),
                        legend=False
                    ).figure

                    st.pyplot(fig_demanda)

                    st.dataframe(
                        demanda_mt_bt_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Esta gráfica suma la Demanda promedio anual (kW) de todos los locales ocupados con recibo "
                        "en todos los centros comerciales, agrupada por nivel de tensión derivado de la tarifa: "
                        "BT = PDBT + GDBT; MT = GDMTH + GDMTO. "
                        + NOTA_DEMANDA_PROMEDIO_ANUAL
                    )

    # ============================================================
    # Distribución por Clima
    # ============================================================

    st.markdown(
        '<div class="section-title">Distribución por Clima</div>',
        unsafe_allow_html=True
    )

    climate_mapping_path = DATA_DIR / "profiles" / "cc_climate_zone_mapping.csv"

    if climate_mapping_path.exists():

        climate_mapping_df = pd.read_csv(
            climate_mapping_path,
            encoding="latin-1"
        )

        climate_mapping_df.columns = (
            climate_mapping_df.columns
            .str.strip()
            .str.lower()
        )

        def clasificar_macro_clima(zona):
            zona = str(zona).strip()

            if zona in ["Hot-Humid", "Hot-Dry"]:
                return "Cálido"

            if zona in ["Mixed-Humid", "Mixed-Dry"]:
                return "Templado"

            if zona in ["Cold", "Cold / Very Cold"]:
                return "Frío"

            return "Sin clasificar"

        climate_mapping_df["Macro clima"] = climate_mapping_df["zona_nrel"].apply(
            clasificar_macro_clima
        )

        clima_dist = (
            climate_mapping_df
            .groupby("Macro clima")
            .size()
            .reset_index(name="Centros comerciales")
        )

        total_cc_clima = clima_dist["Centros comerciales"].sum()

        clima_dist["(%)"] = (
            clima_dist["Centros comerciales"] / total_cc_clima * 100
        ).round(1)

        orden_clima = {
            "Cálido": 1,
            "Templado": 2,
            "Frío": 3,
            "Sin clasificar": 4
        }

        clima_dist["_orden"] = clima_dist["Macro clima"].map(orden_clima).fillna(999)

        clima_dist = clima_dist.sort_values("_orden").drop(columns=["_orden"])

        # Data para barra horizontal apilada 100%
        clima_bar = pd.DataFrame({
            row["Macro clima"]: [row["(%)"]]
            for _, row in clima_dist.iterrows()
        })

        fig, ax = plt.subplots(figsize=(8, 1.6))

        clima_bar.plot(
            kind="barh",
            stacked=True,
            ax=ax,
            legend=True
        )

        ax.set_xlim(0, 100)
        ax.set_yticks([])

        ax.set_title(
            "Distribución de centros comerciales por clima",
            fontsize=7
        )

        ax.set_xlabel(
            "% de centros comerciales",
            fontsize=7
        )

        for container in ax.containers:
            ax.bar_label(
                container,
                label_type="center",
                fmt="%.1f%%",
                fontsize=7
            )

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=len(clima_dist),
            fontsize=7
        )

        st.pyplot(fig)

        clima_dist_display = clima_dist.copy()
        clima_dist_display["(%)"] = clima_dist_display["(%)"].map(lambda x: f"{x:.1f}%")

        st.dataframe(
            clima_dist_display,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.warning(
            f"No encontré el archivo de mapeo climático: {climate_mapping_path}"
        )

    # ============================================================
    # Distribución por tipo de centro comercial
    # ============================================================

    st.markdown(
        '<div class="section-title">Distribución por tipo de centro comercial</div>',
        unsafe_allow_html=True
    )

    cc_master_path = DATA_DIR / "profiles" / "cc_master_data.csv"

    if cc_master_path.exists():

        cc_master_df = pd.read_csv(cc_master_path)

        cc_master_df.columns = (
            cc_master_df.columns
            .str.strip()
        )

        tipo_cc_col = "Tipo de Mall"

        if tipo_cc_col in cc_master_df.columns:

            tipo_cc_dist = (
                cc_master_df
                .groupby(tipo_cc_col)
                .size()
                .reset_index(name="Centros comerciales")
                .rename(columns={tipo_cc_col: "Tipo de centro comercial"})
            )

            total_tipo_cc = tipo_cc_dist["Centros comerciales"].sum()

            tipo_cc_dist["(%)"] = (
                tipo_cc_dist["Centros comerciales"] / total_tipo_cc * 100
            ).round(1)

            tipo_cc_dist = tipo_cc_dist.sort_values(
                "(%)",
                ascending=False
            )

            orden_tipo_cc = {
                "Luxury Fashion Mall": 1,
                "Fashion Mall": 2,
                "Regional Mall": 3,
                "Power Center": 4,
                "Strip Mall": 5
            }

            tipo_cc_dist["_orden"] = (
                tipo_cc_dist["Tipo de centro comercial"]
                .map(orden_tipo_cc)
                .fillna(999)
            )

            tipo_cc_dist = (
                tipo_cc_dist
                .sort_values("_orden")
                .drop(columns=["_orden"])
            )

            tipo_cc_bar = pd.DataFrame({
                row["Tipo de centro comercial"]: [row["(%)"]]
                for _, row in tipo_cc_dist.iterrows()
            })

            fig, ax = plt.subplots(figsize=(8, 1.6))

            tipo_cc_bar.plot(
                kind="barh",
                stacked=True,
                ax=ax,
                legend=True
            )

            ax.set_xlim(0, 100)
            ax.set_xlabel(
                "% de centros comerciales",
                fontsize=8
            )
            ax.set_yticks([])
            ax.set_title(
                "Distribución de centros comerciales por tipo",
                fontsize=8
            )

            for container in ax.containers:
                ax.bar_label(
                    container,
                    label_type="center",
                    fmt="%.1f%%",
                    fontsize=8
                )

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.35),
                ncol=4,
                fontsize=7,
                frameon=False
            )

            st.pyplot(fig)

            tipo_cc_display = tipo_cc_dist.copy()
            tipo_cc_display["(%)"] = tipo_cc_display["(%)"].map(lambda x: f"{x:.1f}%")

            st.dataframe(
                tipo_cc_display,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning("No encontré la columna 'Tipo de Mall' en cc_master_data.csv.")

    else:
        st.warning(f"No encontré el archivo maestro de centros comerciales: {cc_master_path}")

with tab_cc:

    st.markdown(
        '<div class="section-title">Análisis por centro comercial</div>',
        unsafe_allow_html=True
    )

    if mall_col:

        mall_options = sorted(
            filtered[mall_col]
            .dropna()
            .unique()
        )

        mall_map = {
            limpiar_nombre_cc(x): x
            for x in mall_options
        }

        selected_cc_display = st.selectbox(
            "Selecciona un centro comercial",
            options=sorted(mall_map.keys())
        )

        selected_cc = mall_map[selected_cc_display]

        cc_parser = filtered[
            filtered[mall_col] == selected_cc
        ].copy()

        general_mall_col = first_existing_column(
            general_data,
            ["NOMBRE DEL CC", "CENTRO COMERCIAL", "CC", "PLAZA", "source_sheet"]
        )

        selected_cc_norm = str(selected_cc).strip().upper()

        general_cc_data = general_data[
            general_data[general_mall_col]
            .astype(str)
            .str.strip()
            .str.upper()
            .apply(lambda x: x in selected_cc_norm or selected_cc_norm in x)
        ].copy()

        def normalize_key_series(s):
            return (
                s.fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)
                .str.replace(".", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("'", "", regex=False)
                .str.replace("&", "Y", regex=False)
                .str.replace("Á", "A", regex=False)
                .str.replace("É", "E", regex=False)
                .str.replace("Í", "I", regex=False)
                .str.replace("Ó", "O", regex=False)
                .str.replace("Ú", "U", regex=False)
                .str.replace("Ñ", "N", regex=False)
                .str.replace("CITIBANAMEX", "BANAMEX", regex=False)
                .str.replace("CAFETERADE", "CAFETERA DE", regex=False)
                .str.replace(" SA DE CV", "", regex=False)
                .str.replace(" S A DE C V", "", regex=False)
                .str.replace(" S DE RL DE CV", "", regex=False)
                .str.replace(" S DE R L DE C V", "", regex=False)
                .str.replace(" SAPI DE CV", "", regex=False)
                .str.replace(" SA", "", regex=False)
                .str.replace(" DE CV", "", regex=False)
                .str.replace(" INST DE BANCA MULT", "", regex=False)
                .str.replace(" INSTITUCION DE BANCA MULTIPLE", "", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

        def normalize_meter_series(s):
            return (
                s.fillna("")
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(" ", "", regex=False)
                .str.replace("-", "", regex=False)
                .str.replace(".", "", regex=False)
            )



        general_meter_col = first_existing_column(
            general_cc_data,
            ["No. De medidor", "No. de medidor", "No de medidor", "MEDIDOR", "Medidor"]
        )

        general_cc_data["_key_medidor"] = normalize_meter_series(general_cc_data[general_meter_col]) if general_meter_col else ""
        general_cc_data["_key_cliente"] = normalize_key_series(general_cc_data["CLIENTE"]) if "CLIENTE" in general_cc_data.columns else ""
        general_cc_data["_key_nombre_comercial"] = normalize_key_series(general_cc_data["NOMBRE COMERCIAL"]) if "NOMBRE COMERCIAL" in general_cc_data.columns else ""
        general_cc_data["_key_tarifa"] = normalize_key_series(general_cc_data["TARIFA"]) if "TARIFA" in general_cc_data.columns else ""

        parser_medidores = set(normalize_meter_series(cc_parser["medidor"]).unique()) if "medidor" in cc_parser.columns else set()
        parser_clientes = set(normalize_key_series(cc_parser["cliente_nombre"]).unique()) if "cliente_nombre" in cc_parser.columns else set()
        parser_nombres = set(normalize_key_series(cc_parser["recibos_subgroup"]).unique()) if "recibos_subgroup" in cc_parser.columns else set()

        parser_medidores.discard("")
        parser_clientes.discard("")
        parser_nombres.discard("")

        cliente_status = general_cc_data["CLIENTE"] if "CLIENTE" in general_cc_data.columns else pd.Series([None] * len(general_cc_data))
        nombre_status = general_cc_data["NOMBRE COMERCIAL"] if "NOMBRE COMERCIAL" in general_cc_data.columns else pd.Series([None] * len(general_cc_data))

        disponible_o_vacio = (
            cliente_status.isna()
            | nombre_status.isna()
            | (cliente_status.astype(str).str.strip() == "")
            | (nombre_status.astype(str).str.strip() == "")
            | (cliente_status.astype(str).str.strip().str.lower() == "disponible")
            | (nombre_status.astype(str).str.strip().str.lower() == "disponible")
        )

        general_ocupados = general_cc_data[~disponible_o_vacio].copy()

        general_ocupados["_match_medidor"] = (
            general_ocupados["_key_medidor"].isin(parser_medidores)
            & (general_ocupados["_key_medidor"] != "")
        )

        general_ocupados["_match_cliente"] = (
            general_ocupados["_key_cliente"].isin(parser_clientes)
            & (general_ocupados["_key_cliente"] != "")
        )

        def has_partial_match(value, candidates):
            if not value:
                return False

            value = str(value).strip()
            value_compact = value.replace(" ", "")

            stopwords = {
                "DE", "DEL", "LA", "LAS", "LOS", "EL", "Y",
                "SA", "CV", "S", "RL", "SAPI", "SOCIEDAD",
                "ANONIMA", "CAPITAL", "VARIABLE", "MEXICO",
                "COMERCIALIZADORA", "GRUPO", "OPERADORA",
                "DISTRIBUIDORA", "TIENDAS", "SERVICIOS"
            }

            value_words = {
                w for w in value.split()
                if len(w) >= 4 and w not in stopwords
            }

            for candidate in candidates:
                if not candidate:
                    continue

                candidate = str(candidate).strip()
                candidate_compact = candidate.replace(" ", "")

                if value in candidate or candidate in value:
                    return True

                if value_compact in candidate_compact or candidate_compact in value_compact:
                    return True

                candidate_words = {
                    w for w in candidate.split()
                    if len(w) >= 4 and w not in stopwords
                }

                if value_words and candidate_words:
                    common_words = value_words & candidate_words

                    if len(common_words) >= 1:
                        return True

            return False

        general_ocupados["_match_nombre_comercial"] = (
            general_ocupados["_key_nombre_comercial"].apply(
                lambda x: has_partial_match(x, parser_nombres)
            )
        )

        general_ocupados["_tiene_recibo"] = (
            general_ocupados["_match_medidor"]
            | general_ocupados["_match_cliente"]
            | general_ocupados["_match_nombre_comercial"]
        )

        # ============================================================
        # Traer tarifa del parser a Datos Generales para análisis
        # ============================================================

        parser_tarifa_lookup = {}

        if "tarifa" in cc_parser.columns:

            if "cliente_nombre" in cc_parser.columns:
                temp = cc_parser.copy()
                temp["_key_cliente"] = normalize_key_series(temp["cliente_nombre"])
                cliente_tarifa = (
                    temp.dropna(subset=["tarifa"])
                    .drop_duplicates(subset=["_key_cliente"])
                    .set_index("_key_cliente")["tarifa"]
                    .to_dict()
                )
                parser_tarifa_lookup.update(cliente_tarifa)

            if "recibos_subgroup" in cc_parser.columns:
                temp = cc_parser.copy()
                temp["_key_nombre_comercial"] = normalize_key_series(temp["recibos_subgroup"])
                nombre_tarifa = (
                    temp.dropna(subset=["tarifa"])
                    .drop_duplicates(subset=["_key_nombre_comercial"])
                    .set_index("_key_nombre_comercial")["tarifa"]
                    .to_dict()
                )
                parser_tarifa_lookup.update(nombre_tarifa)

        def get_tarifa_from_parser(row):
            cliente_key = row.get("_key_cliente")
            nombre_key = row.get("_key_nombre_comercial")

            if cliente_key in parser_tarifa_lookup:
                return parser_tarifa_lookup[cliente_key]

            if nombre_key in parser_tarifa_lookup:
                return parser_tarifa_lookup[nombre_key]

            for parser_key, tarifa_value in parser_tarifa_lookup.items():
                if has_partial_match(cliente_key, [parser_key]):
                    return tarifa_value

                if has_partial_match(nombre_key, [parser_key]):
                    return tarifa_value

            return None

        general_ocupados["TARIFA_ANALISIS"] = general_ocupados.apply(
            get_tarifa_from_parser,
            axis=1
        )

        if "TARIFA" in general_ocupados.columns:
            general_ocupados["TARIFA_ANALISIS"] = general_ocupados["TARIFA_ANALISIS"].fillna(
                general_ocupados["TARIFA"]
            )

        locales_ocupados = len(general_ocupados)
        locales_con_recibo = int(general_ocupados["_tiene_recibo"].sum())
        locales_sin_recibo = int(locales_ocupados - locales_con_recibo)
        cobertura_pct = round((locales_con_recibo / locales_ocupados * 100), 1) if locales_ocupados else 0

        # --------------------------------------------------
        # Forzar uso de la muestra global validada
        # --------------------------------------------------
        # El tab Centro Comercial no debe recalcular cobertura.
        # Debe usar coverage_by_mall, que viene del match global
        # usado por Calidad de Datos y por el Resumen Ejecutivo.

        coverage_cc_global = coverage_by_mall.copy()

        if not coverage_cc_global.empty:
            coverage_cc_global["_cc_key_reporte"] = (
                coverage_cc_global["Centro Comercial"].apply(cc_key)
            )

            selected_cc_key = cc_key(selected_cc)

            coverage_cc_global = coverage_cc_global[
                coverage_cc_global["_cc_key_reporte"].eq(selected_cc_key)
            ].copy()

        if not coverage_cc_global.empty:
            coverage_row_global = coverage_cc_global.iloc[0]

            locales_ocupados = int(
                coverage_row_global.get("Locales ocupados", 0)
            )

            locales_con_recibo = int(
                coverage_row_global.get("Locales ocupados con recibo", 0)
            )

            locales_sin_recibo = int(
                coverage_row_global.get("Locales ocupados sin recibo", 0)
            )

            cobertura_pct = float(
                coverage_row_global.get("Cobertura de muestra (%)", 0)
            )

        col_ocupacion_grafica, col_muestra_grafica = st.columns(2)

        with col_ocupacion_grafica:
            st.markdown(
                '<div class="subsection-title">Ocupación de centro comercial</div>',
                unsafe_allow_html=True
            )

            total_locales = len(general_cc_data)
            disponibles_vacios = int(disponible_o_vacio.sum())

            ocupacion_pie = pd.DataFrame({
                "Estatus": [
                    "Locales ocupados",
                    "Locales vacíos"
                ],
                "Cantidad": [
                    locales_ocupados,
                    disponibles_vacios
                ]
            })

            # Tamaño fijo para que el círculo no cambie según el CC seleccionado.
            # Figura y área del círculo fijas para todos los CC.
            imagen_ocupacion = mostrar_pie_fijo(
                valores=ocupacion_pie["Cantidad"],
                etiqueta_superior="Locales\nvacíos",
                etiqueta_inferior="Locales\nocupados"
            )

            st.image(
                imagen_ocupacion,
                use_container_width=True
            )

        with col_muestra_grafica:
            st.markdown(
                '<div class="subsection-title">Muestra disponible</div>',
                unsafe_allow_html=True
            )

            muestra_pie = pd.DataFrame({
                "Estatus": [
                    "Locales ocupados con recibo",
                    "Locales ocupados sin recibo"
                ],
                "Cantidad": [
                    locales_con_recibo,
                    locales_sin_recibo
                ]
            })

            # Mismo tamaño exacto que la gráfica de ocupación.
            # Figura y área del círculo exactamente iguales a Ocupación.
            imagen_muestra = mostrar_pie_fijo(
                valores=muestra_pie["Cantidad"],
                etiqueta_superior="Locales ocupados\nsin recibo",
                etiqueta_inferior="Locales ocupados\ncon recibo"
            )

            st.image(
                imagen_muestra,
                use_container_width=True
            )

        # --------------------------------------------------------
        # Tablas alineadas en una segunda fila
        # --------------------------------------------------------

        col_ocupacion_tabla, col_muestra_tabla = st.columns(2)

        with col_ocupacion_tabla:
            st.dataframe(
                pd.DataFrame({
                    "Métrica": [
                        "Locales ocupados",
                        "Locales vacíos",
                        "Total de Locales"
                    ],
                    "Cantidad": [
                        locales_ocupados,
                        disponibles_vacios,
                        total_locales
                    ]
                }),
                use_container_width=True,
                hide_index=True
            )

        with col_muestra_tabla:
            st.dataframe(
                pd.DataFrame({
                    "Métrica": [
                        "Locales ocupados con recibo",
                        "Locales ocupados sin recibo",
                        "Total",
                        "Cobertura de muestra (%)"
                    ],
                    "Cantidad": [
                        locales_con_recibo,
                        locales_sin_recibo,
                        locales_ocupados,
                        cobertura_pct
                    ]
                }),
                use_container_width=True,
                hide_index=True
            )

        with st.expander("Diagnóstico de criterios de cruce"):
            st.write("Cruce por medidor:", int(general_ocupados["_match_medidor"].sum()))
            st.write("Cruce por cliente:", int(general_ocupados["_match_cliente"].sum()))
            st.write("Cruce por nombre comercial:", int(general_ocupados["_match_nombre_comercial"].sum()))

        # ============================================================
        # Parser sin match contra Datos Generales
        # ============================================================

        dg_clientes = set(general_ocupados.loc[general_ocupados["_tiene_recibo"], "_key_cliente"])
        dg_nombres = set(general_ocupados.loc[general_ocupados["_tiene_recibo"], "_key_nombre_comercial"])
        dg_medidores = set(general_ocupados.loc[general_ocupados["_tiene_recibo"], "_key_medidor"])

        parser_unmatched = cc_parser.copy()

        parser_unmatched["_cliente_key"] = normalize_key_series(parser_unmatched["cliente_nombre"]) if "cliente_nombre" in parser_unmatched.columns else ""
        parser_unmatched["_nombre_key"] = normalize_key_series(parser_unmatched["recibos_subgroup"]) if "recibos_subgroup" in parser_unmatched.columns else ""
        parser_unmatched["_medidor_key"] = normalize_meter_series(parser_unmatched["medidor"]) if "medidor" in parser_unmatched.columns else ""

        general_clientes = set(general_ocupados["_key_cliente"].dropna().unique())
        general_nombres = set(general_ocupados["_key_nombre_comercial"].dropna().unique())

        general_clientes.discard("")
        general_nombres.discard("")

        parser_unmatched["_match_dg_cliente"] = parser_unmatched["_cliente_key"].apply(
            lambda x: has_partial_match(x, general_clientes)
        )

        parser_unmatched["_match_dg_nombre"] = parser_unmatched["_nombre_key"].apply(
            lambda x: has_partial_match(x, general_nombres)
        )

        parser_unmatched["_match_dg"] = (
            parser_unmatched["_cliente_key"].isin(dg_clientes)
            | parser_unmatched["_nombre_key"].isin(dg_nombres)
            | parser_unmatched["_medidor_key"].isin(dg_medidores)
            | parser_unmatched["_match_dg_cliente"]
            | parser_unmatched["_match_dg_nombre"]
        )

        # --------------------------------------------------
        # Propagar match por no_servicio
        # --------------------------------------------------
        # Si un mismo no_servicio tiene varios medidores, y cualquiera
        # de sus filas sí hizo match contra DG, entonces todo el servicio
        # se considera con match. Esto evita que un medidor histórico
        # salga como "sin match" cuando pertenece al mismo servicio/local.

        if "no_servicio" in parser_unmatched.columns:
            parser_unmatched["_key_no_servicio"] = (
                parser_unmatched["no_servicio"]
                .fillna("")
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.strip()
            )

            servicios_con_match = set(
                parser_unmatched.loc[
                    parser_unmatched["_match_dg"]
                    & parser_unmatched["_key_no_servicio"].ne(""),
                    "_key_no_servicio"
                ]
                .dropna()
                .astype(str)
                .unique()
            )

            parser_unmatched["_match_dg_por_no_servicio"] = (
                parser_unmatched["_key_no_servicio"].isin(servicios_con_match)
                & parser_unmatched["_key_no_servicio"].ne("")
            )

            parser_unmatched["_match_dg"] = (
                parser_unmatched["_match_dg"]
                | parser_unmatched["_match_dg_por_no_servicio"]
            )

        parser_unmatched = parser_unmatched[
            ~parser_unmatched["_match_dg"]
        ].copy()

        st.markdown(
            '<div class="subsection-title">Recibos encontrados en parser sin match en Datos Generales</div>',
            unsafe_allow_html=True
        )

        servicios_parser_sin_match_count = (
            parser_unmatched["no_servicio"].nunique()
            if "no_servicio" in parser_unmatched.columns
            else len(parser_unmatched)
        )

        if servicios_parser_sin_match_count > 0:
            st.write(
                "Servicios únicos en parser sin match:",
                servicios_parser_sin_match_count
            )

        parser_cols = [
            "cliente_nombre",
            "no_servicio",
            "medidor",
            "tarifa",
            "file_path"
        ]

        parser_cols = [
            col for col in parser_cols
            if col in parser_unmatched.columns
        ]

        # --------------------------------------------------
        # Mostrar una fila por no_servicio
        # --------------------------------------------------
        # Un servicio puede tener más de un medidor por cambio durante
        # el año. No debe aparecer duplicado; los medidores se agrupan.
        # También mostramos file_path completo para poder revisar el origen.

        if "no_servicio" in parser_unmatched.columns:

            parser_unmatched_display = parser_unmatched.copy()

            parser_unmatched_display["_key_no_servicio"] = (
                parser_unmatched_display["no_servicio"]
                .fillna("")
                .astype(str)
                .str.replace(r"\.0$", "", regex=True)
                .str.strip()
            )

            def join_unique_display(serie, upper=False):
                vals = []

                for v in serie.dropna().astype(str).tolist():
                    v = v.strip()

                    if re.match(r"^\d+\.0$", v):
                        v = v.replace(".0", "")

                    if upper:
                        v = v.upper()

                    if v in ["", "nan", "None", "NONE", "NAN", "<NA>"]:
                        continue

                    vals.append(v)

                vals = sorted(set(vals))

                return " | ".join(vals) if vals else pd.NA

            agg_dict_display = {}

            if "cliente_nombre" in parser_unmatched_display.columns:
                agg_dict_display["cliente_nombre"] = (
                    "cliente_nombre",
                    lambda x: join_unique_display(x)
                )

            if "no_servicio" in parser_unmatched_display.columns:
                agg_dict_display["no_servicio"] = (
                    "no_servicio",
                    lambda x: join_unique_display(x)
                )

            if "medidor" in parser_unmatched_display.columns:
                agg_dict_display["medidor"] = (
                    "medidor",
                    lambda x: join_unique_display(x, upper=True)
                )

            if "tarifa" in parser_unmatched_display.columns:
                agg_dict_display["tarifa"] = (
                    "tarifa",
                    lambda x: join_unique_display(x, upper=True)
                )

            if "file_path" in parser_unmatched_display.columns:
                agg_dict_display["file_path"] = (
                    "file_path",
                    lambda x: join_unique_display(x)
                )

            parser_unmatched_display = (
                parser_unmatched_display
                .groupby("_key_no_servicio", dropna=False)
                .agg(**agg_dict_display)
                .reset_index(drop=True)
            )

            parser_display_cols = [
                col for col in [
                    "cliente_nombre",
                    "no_servicio",
                    "medidor",
                    "tarifa",
                    "file_path"
                ]
                if col in parser_unmatched_display.columns
            ]

        else:
            parser_unmatched_display = parser_unmatched[parser_cols].drop_duplicates()
            parser_display_cols = parser_cols

        if parser_unmatched_display.empty:
            st.success(
                "No hay servicios únicos en parser sin match para este centro comercial."
            )
        else:
            st.dataframe(
                parser_unmatched_display[parser_display_cols],
                use_container_width=True,
                height=600,
                hide_index=True,
                column_config={
                    "file_path": st.column_config.TextColumn(
                        "file_path",
                        width="large",
                        help="Ruta completa del archivo en el parser"
                    )
                }
            )

        # ============================================================
        # Composición de la muestra
        # ============================================================

        st.markdown(
            """
            <div class="section-title">Composición de la muestra</div>
            """,
            unsafe_allow_html=True
        )

        # ------------------------------------------------------------
        # Base única del CC seleccionado usando la muestra global
        # ------------------------------------------------------------
        # A partir de aquí, el tab CC debe usar esta base,
        # no la muestra local calculada con general_ocupados.

        muestra_cc_global = muestra_con_recibo_global.copy()

        if not muestra_cc_global.empty:
            if "_centro_comercial_limpio" in muestra_cc_global.columns:
                muestra_cc_global["_cc_key_reporte"] = (
                    muestra_cc_global["_centro_comercial_limpio"].apply(cc_key)
                )

            elif "NOMBRE DEL CC" in muestra_cc_global.columns:
                muestra_cc_global["_cc_key_reporte"] = (
                    muestra_cc_global["NOMBRE DEL CC"].apply(cc_key)
                )

            elif "source_sheet" in muestra_cc_global.columns:
                muestra_cc_global["_cc_key_reporte"] = (
                    muestra_cc_global["source_sheet"].apply(cc_key)
                )

            else:
                muestra_cc_global["_cc_key_reporte"] = ""

            muestra_cc_global = muestra_cc_global[
                muestra_cc_global["_cc_key_reporte"].eq(cc_key(selected_cc))
            ].copy()



        # ------------------------------------------------------------
        # Número de usuarios por tarifa
        # ------------------------------------------------------------

        st.markdown(
            '<div class="subsection-title">Número de usuarios por tarifa</div>',
            unsafe_allow_html=True
        )

        if not muestra_cc_global.empty:

            muestra_tarifa = muestra_cc_global.copy()

            # --------------------------------------------------------
            # Tarifa para composición
            # --------------------------------------------------------
            # Esta tarifa ya viene resuelta desde el match global.
            # No se vuelve a decidir aquí.

            if "TARIFA_FINAL" in muestra_tarifa.columns:
                muestra_tarifa["TARIFA_COMPOSICION"] = (
                    muestra_tarifa["TARIFA_FINAL"]
                    .fillna("SIN TARIFA")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .replace({
                        "": "SIN TARIFA",
                        "NAN": "SIN TARIFA",
                        "NONE": "SIN TARIFA",
                        "<NA>": "SIN TARIFA"
                    })
                )
            else:
                muestra_tarifa["TARIFA_COMPOSICION"] = "SIN TARIFA"

            muestra_tarifa["TARIFA_COMPOSICION"] = (
                muestra_tarifa["TARIFA_COMPOSICION"]
                .fillna("SIN TARIFA")
                .astype(str)
                .str.upper()
                .str.strip()
                .replace({
                    "": "SIN TARIFA",
                    "NAN": "SIN TARIFA",
                    "NONE": "SIN TARIFA",
                    "<NA>": "SIN TARIFA"
                })
            )

            tarifa_comp = (
                muestra_tarifa
                .groupby("TARIFA_COMPOSICION", dropna=False)
                .size()
                .reset_index(name="Número de usuarios")
                .rename(columns={"TARIFA_COMPOSICION": "Tarifa"})
                .sort_values("Número de usuarios", ascending=False)
            )

            total_tarifa = len(muestra_tarifa)

            tarifa_comp["(%)"] = (
                tarifa_comp["Número de usuarios"] / total_tarifa * 100
            ).round(1)

            tarifa_total_row = pd.DataFrame({
                "Tarifa": ["Total"],
                "Número de usuarios": [total_tarifa],
                "(%)": [100.0]
            })

            tarifa_comp_display = pd.concat(
                [tarifa_comp, tarifa_total_row],
                ignore_index=True
            )

            tarifa_comp_display["(%)"] = (
                tarifa_comp_display["(%)"]
                .map(lambda x: f"{x:.1f}%")
            )

            col_tarifa_pie, col_tarifa_table = st.columns(
                2,
                vertical_alignment="top"
            )

            with col_tarifa_pie:
                imagen_tarifa = mostrar_pie_composicion_fijo(
                    etiquetas=tarifa_comp["Tarifa"].tolist(),
                    valores=tarifa_comp["Número de usuarios"].tolist()
                )

                st.image(
                    imagen_tarifa,
                    use_container_width=False
                )

            with col_tarifa_table:
                st.dataframe(tarifa_comp_display, use_container_width=True)

        else:
            st.info("No hay información de tarifa para la muestra con recibo.")

        # ------------------------------------------------------------
        # Número de usuarios por giro comercial
        # ------------------------------------------------------------
        st.markdown(
            '<div class="subsection-title">Número de usuarios por giro comercial</div>',
            unsafe_allow_html=True
        )

        if not muestra_cc_global.empty:

            muestra_giro = muestra_cc_global.copy()

            # --------------------------------------------------------
            # Giro normalizado para composición
            # --------------------------------------------------------
            # No eliminamos locales sin giro. Los dejamos como SIN GIRO
            # para que el total de la composición cuadre con la muestra disponible.

            if "SUBGIRO_COMERCIAL" in muestra_giro.columns:
                giro_base = muestra_giro["SUBGIRO_COMERCIAL"]

            elif "GIRO_COMERCIAL" in muestra_giro.columns:
                giro_base = muestra_giro["GIRO_COMERCIAL"]

            elif "GIRO" in muestra_giro.columns:
                giro_base = muestra_giro["GIRO"]

            else:
                giro_base = pd.Series(
                    [pd.NA] * len(muestra_giro),
                    index=muestra_giro.index
                )

            muestra_giro["GIRO_NORMALIZADO"] = (
                giro_base
                .fillna("")
                .astype(str)
                .str.strip()
                .str.title()
                .replace({
                    "": "SIN GIRO",
                    "Nan": "SIN GIRO",
                    "None": "SIN GIRO",
                    "<Na>": "SIN GIRO"
                })
            )

            giro_comp = (
                muestra_giro
                .groupby("GIRO_NORMALIZADO", dropna=False)
                .size()
                .reset_index(name="Número de usuarios")
                .rename(columns={"GIRO_NORMALIZADO": "Giro comercial"})
                .sort_values("Número de usuarios", ascending=False)
            )

            total_giro = len(muestra_giro)

            giro_comp["(%)"] = (
                giro_comp["Número de usuarios"] / total_giro * 100
            ).round(1)

            giro_total_row = pd.DataFrame({
                "Giro comercial": ["Total"],
                "Número de usuarios": [total_giro],
                "(%)": [100.0]
            })

            giro_comp_display = pd.concat(
                [giro_comp, giro_total_row],
                ignore_index=True
            )

            giro_comp_display["(%)"] = (
                giro_comp_display["(%)"]
                .map(lambda x: f"{x:.1f}%")
            )

            col_giro_pie, col_giro_table = st.columns(
                2,
                vertical_alignment="top"
            )

            with col_giro_pie:
                imagen_giro = mostrar_pie_composicion_fijo(
                    etiquetas=giro_comp["Giro comercial"].tolist(),
                    valores=giro_comp["Número de usuarios"].tolist()
                )

                st.image(
                    imagen_giro,
                    use_container_width=False
                )

            with col_giro_table:
                st.dataframe(
                    giro_comp_display,
                    use_container_width=True
                )

        else:
            st.info("No hay información de giro comercial para la muestra con recibo.")

        # ============================================================
        # Demanda Real Vs Demanda Contratada
        # ============================================================

        st.markdown(
            """
            <div class="section-title">Demanda Real Vs Demanda Contratada</div>
            """,
            unsafe_allow_html=True
        )

        # ------------------------------------------------------------
        # Demanda Real Vs Demanda Contratada
        # Base: muestra validada de locales ocupados con recibo
        # ------------------------------------------------------------

        if "muestra_cc_global" not in locals() or muestra_cc_global.empty:
            st.info(
                "No está disponible la muestra validada de locales ocupados con recibo."
            )

        elif "demanda_contratada_kw" not in parsed.columns:
            st.warning(
                "No encontré la columna `demanda_contratada_kw` en el parser. "
                "No se puede construir la comparación contra demanda contratada."
            )

        else:
            demanda_cc_base = enriquecer_muestra_con_demanda(
                muestra_con_recibo=muestra_cc_global,
                parsed=parsed
            )

            # --------------------------------------------------------
            # Filtrar al centro comercial seleccionado
            # --------------------------------------------------------

            if "_centro_comercial_limpio" in demanda_cc_base.columns:
                demanda_cc_base["_cc_key_reporte"] = demanda_cc_base[
                    "_centro_comercial_limpio"
                ].apply(cc_key)

            elif "NOMBRE DEL CC" in demanda_cc_base.columns:
                demanda_cc_base["_cc_key_reporte"] = demanda_cc_base[
                    "NOMBRE DEL CC"
                ].apply(cc_key)

            elif "source_sheet" in demanda_cc_base.columns:
                demanda_cc_base["_cc_key_reporte"] = demanda_cc_base[
                    "source_sheet"
                ].apply(cc_key)

            else:
                demanda_cc_base["_cc_key_reporte"] = ""

            # `selected_cc` normalmente es el centro comercial elegido en el tab.
            # Si en tu código el selector se llama distinto, cambia selected_cc por ese nombre.
            selected_cc_key = cc_key(selected_cc) if "selected_cc" in locals() else None

            if selected_cc_key:
                demanda_cc_base = demanda_cc_base[
                    demanda_cc_base["_cc_key_reporte"] == selected_cc_key
                ].copy()

            # --------------------------------------------------------
            # Agregar demanda contratada desde parsed
            # --------------------------------------------------------

            parsed_demanda_contratada = parsed.copy()

            parsed_demanda_contratada["_cc_key_reporte"] = (
                parsed_demanda_contratada["mall_folder"].apply(cc_key)
                if "mall_folder" in parsed_demanda_contratada.columns
                else ""
            )

            parsed_demanda_contratada["_key_medidor"] = (
                normalize_meter_cc(parsed_demanda_contratada["medidor"])
                if "medidor" in parsed_demanda_contratada.columns
                else ""
            )

            parsed_demanda_contratada["_key_cliente"] = (
                normalize_cc_key(parsed_demanda_contratada["cliente_nombre"])
                if "cliente_nombre" in parsed_demanda_contratada.columns
                else ""
            )

            parsed_demanda_contratada["_key_nombre"] = (
                normalize_cc_key(parsed_demanda_contratada["recibos_subgroup"])
                if "recibos_subgroup" in parsed_demanda_contratada.columns
                else ""
            )

            parsed_demanda_contratada["_demanda_contratada_kw"] = clean_number_series(
                parsed_demanda_contratada["demanda_contratada_kw"]
            )

            parsed_demanda_contratada["_key_no_servicio"] = (
                normalize_service_cc(parsed_demanda_contratada["no_servicio"])
                if "no_servicio" in parsed_demanda_contratada.columns
                else ""
            )

            # Una fila por local/servicio. Usamos la demanda contratada máxima
            # encontrada, por si el mismo servicio aparece en varios recibos.
            demanda_contratada_lookup = (
                parsed_demanda_contratada
                .dropna(subset=["_demanda_contratada_kw"])
                [
                    parsed_demanda_contratada["_key_no_servicio"]
                    .astype(str)
                    .str.strip()
                    .ne("")
                ]
                .groupby(
                    [
                        "_cc_key_reporte",
                        "_key_no_servicio"
                    ],
                    dropna=False
                )
                .agg(
                    demanda_contratada_kw=(
                        "_demanda_contratada_kw",
                        "max"
                    )
                )
                .reset_index()
            )

            # --------------------------------------------------------
            # Llaves en la muestra validada
            # --------------------------------------------------------

            if "_key_medidor" not in demanda_cc_base.columns:
                medidor_col_muestra = first_existing_column(
                    demanda_cc_base,
                    [
                        "No. De medidor",
                        "No. de medidor",
                        "No de medidor",
                        "MEDIDOR",
                        "Medidor"
                    ]
                )

                demanda_cc_base["_key_medidor"] = (
                    normalize_meter_cc(demanda_cc_base[medidor_col_muestra])
                    if medidor_col_muestra
                    else ""
                )

            if "_key_cliente" not in demanda_cc_base.columns:
                demanda_cc_base["_key_cliente"] = (
                    normalize_cc_key(demanda_cc_base["CLIENTE"])
                    if "CLIENTE" in demanda_cc_base.columns
                    else ""
                )

            if "_key_nombre_comercial" not in demanda_cc_base.columns:
                demanda_cc_base["_key_nombre_comercial"] = (
                    normalize_cc_key(demanda_cc_base["NOMBRE COMERCIAL"])
                    if "NOMBRE COMERCIAL" in demanda_cc_base.columns
                    else ""
                )

            demanda_df = demanda_cc_base.copy()

            demanda_df["Demanda contratada kW"] = pd.NA

            # --------------------------------------------------------
            # Lookup principal por no_servicio
            # --------------------------------------------------------
            # El no_servicio es la unidad correcta cuando un local tuvo
            # más de un medidor durante el año.

            if "parser_no_servicio_match" in demanda_df.columns:
                demanda_df["_key_no_servicio_match"] = normalize_service_cc(
                    demanda_df["parser_no_servicio_match"]
                )

            elif "no_servicio" in demanda_df.columns:
                demanda_df["_key_no_servicio_match"] = normalize_service_cc(
                    demanda_df["no_servicio"]
                )

            else:
                demanda_df["_key_no_servicio_match"] = ""

            demanda_contratada_lookup_servicio = (
                parsed_demanda_contratada
                .dropna(subset=["_demanda_contratada_kw"])
                [
                    parsed_demanda_contratada["_key_no_servicio"]
                    .astype(str)
                    .str.strip()
                    .ne("")
                ]
                .groupby(
                    [
                        "_cc_key_reporte",
                        "_key_no_servicio"
                    ],
                    dropna=False
                )
                .agg(
                    demanda_contratada_kw=(
                        "_demanda_contratada_kw",
                        "max"
                    )
                )
                .reset_index()
            )

            demanda_df = demanda_df.merge(
                demanda_contratada_lookup_servicio,
                left_on=[
                    "_cc_key_reporte",
                    "_key_no_servicio_match"
                ],
                right_on=[
                    "_cc_key_reporte",
                    "_key_no_servicio"
                ],
                how="left"
            )

            # ------------------------------------------------------------
            # Asegurar columna demanda_contratada_kw
            # ------------------------------------------------------------

            if "demanda_contratada_kw" not in demanda_df.columns:
                demanda_df["demanda_contratada_kw"] = pd.NA

            demanda_df["demanda_contratada_kw"] = pd.to_numeric(
                demanda_df["demanda_contratada_kw"],
                errors="coerce"
            )

            # ------------------------------------------------------------
            # Regla PDBT: si no trae demanda contratada, usar 25 kW
            # ------------------------------------------------------------

            if "TARIFA_FINAL" in demanda_df.columns:
                tarifa_demanda_df = demanda_df["TARIFA_FINAL"]
            elif "Tarifa" in demanda_df.columns:
                tarifa_demanda_df = demanda_df["Tarifa"]
            elif "tarifa" in demanda_df.columns:
                tarifa_demanda_df = demanda_df["tarifa"]
            else:
                tarifa_demanda_df = pd.Series(
                    "",
                    index=demanda_df.index
                )

            mask_pdbt_demanda = (
                tarifa_demanda_df
                .fillna("")
                .astype(str)
                .str.upper()
                .str.strip()
                .eq("PDBT")
            )

            mask_sin_contratada_demanda = demanda_df["demanda_contratada_kw"].isna()

            demanda_df.loc[
                mask_pdbt_demanda & mask_sin_contratada_demanda,
                "demanda_contratada_kw"
            ] = 25

            demanda_df["Demanda contratada kW"] = demanda_df["demanda_contratada_kw"]

            demanda_df = demanda_df.drop(
                columns=[
                    "demanda_contratada_kw",
                    "_key_no_servicio"
                ],
                errors="ignore"
            )

            # --------------------------------------------------------
            # Lookup fallback por medidor / cliente / nombre
            # --------------------------------------------------------
            # Solo se usa si no se encontró demanda contratada por no_servicio.

            demanda_contratada_lookup_fallback = (
                parsed_demanda_contratada
                .dropna(subset=["_demanda_contratada_kw"])
                .groupby(
                    [
                        "_cc_key_reporte",
                        "_key_medidor",
                        "_key_cliente",
                        "_key_nombre"
                    ],
                    dropna=False
                )
                .agg(
                    demanda_contratada_kw=(
                        "_demanda_contratada_kw",
                        "max"
                    )
                )
                .reset_index()
            )

            # --------------------------------------------------------
            # 1) Fallback por medidor
            # --------------------------------------------------------

            faltantes_contratada = demanda_df["Demanda contratada kW"].isna()

            if faltantes_contratada.any():
                lookup_medidor = (
                    demanda_contratada_lookup_fallback[
                        demanda_contratada_lookup_fallback["_key_medidor"]
                        .astype(str)
                        .str.strip()
                        .ne("")
                    ][
                        [
                            "_cc_key_reporte",
                            "_key_medidor",
                            "demanda_contratada_kw"
                        ]
                    ]
                    .drop_duplicates(
                        subset=[
                            "_cc_key_reporte",
                            "_key_medidor"
                        ],
                        keep="first"
                    )
                )

                temp_medidor = demanda_df.loc[faltantes_contratada].merge(
                    lookup_medidor,
                    on=[
                        "_cc_key_reporte",
                        "_key_medidor"
                    ],
                    how="left"
                )

                demanda_df.loc[
                    faltantes_contratada,
                    "Demanda contratada kW"
                ] = temp_medidor["demanda_contratada_kw"].values

            # --------------------------------------------------------
            # 2) Fallback por cliente
            # --------------------------------------------------------

            faltantes_contratada = demanda_df["Demanda contratada kW"].isna()

            if faltantes_contratada.any():
                lookup_cliente = (
                    demanda_contratada_lookup_fallback[
                        demanda_contratada_lookup_fallback["_key_cliente"]
                        .astype(str)
                        .str.strip()
                        .ne("")
                    ][
                        [
                            "_cc_key_reporte",
                            "_key_cliente",
                            "demanda_contratada_kw"
                        ]
                    ]
                    .drop_duplicates(
                        subset=[
                            "_cc_key_reporte",
                            "_key_cliente"
                        ],
                        keep="first"
                    )
                )

                temp_cliente = demanda_df.loc[faltantes_contratada].merge(
                    lookup_cliente,
                    on=[
                        "_cc_key_reporte",
                        "_key_cliente"
                    ],
                    how="left"
                )

                demanda_df.loc[
                    faltantes_contratada,
                    "Demanda contratada kW"
                ] = temp_cliente["demanda_contratada_kw"].values

            # --------------------------------------------------------
            # 3) Fallback por nombre comercial
            # --------------------------------------------------------

            faltantes_contratada = demanda_df["Demanda contratada kW"].isna()

            if faltantes_contratada.any():
                lookup_nombre = (
                    demanda_contratada_lookup_fallback[
                        demanda_contratada_lookup_fallback["_key_nombre"]
                        .astype(str)
                        .str.strip()
                        .ne("")
                    ][
                        [
                            "_cc_key_reporte",
                            "_key_nombre",
                            "demanda_contratada_kw"
                        ]
                    ]
                    .drop_duplicates(
                        subset=[
                            "_cc_key_reporte",
                            "_key_nombre"
                        ],
                        keep="first"
                    )
                )

                temp_nombre = demanda_df.loc[faltantes_contratada].merge(
                    lookup_nombre,
                    left_on=[
                        "_cc_key_reporte",
                        "_key_nombre_comercial"
                    ],
                    right_on=[
                        "_cc_key_reporte",
                        "_key_nombre"
                    ],
                    how="left"
                )

                demanda_df.loc[
                    faltantes_contratada,
                    "Demanda contratada kW"
                ] = temp_nombre["demanda_contratada_kw"].values

            # --------------------------------------------------------
            # Demanda promedio anual
            # --------------------------------------------------------

            demanda_df["Demanda promedio anual (kW)"] = clean_number_series(
                demanda_df["demanda_promedio_anual_kw"]
            )

            demanda_df["Demanda contratada kW"] = clean_number_series(
                demanda_df["Demanda contratada kW"]
            )

            # --------------------------------------------------------
            # No eliminar locales ocupados con recibo
            # --------------------------------------------------------
            # La sección debe conservar todos los locales de muestra_cc_global.
            # Si falta demanda contratada o demanda real, el local se conserva
            # y se muestra con estatus pendiente.

            demanda_df["Demanda contratada kW"] = pd.to_numeric(
                demanda_df["Demanda contratada kW"],
                errors="coerce"
            )

            demanda_df["Demanda promedio anual (kW)"] = pd.to_numeric(
                demanda_df["Demanda promedio anual (kW)"],
                errors="coerce"
            )

            # No hacemos dropna aquí.
            # No filtramos demanda contratada > 0 aquí.

            demanda_df["Contratada (%)"] = np.where(
                demanda_df["Demanda contratada kW"].notna()
                & (demanda_df["Demanda contratada kW"] > 0),
                100,
                0
            )

            demanda_df["Promedio anual (%)"] = np.where(
                demanda_df["Demanda promedio anual (kW)"].notna()
                & demanda_df["Demanda contratada kW"].notna()
                & (demanda_df["Demanda contratada kW"] > 0),
                demanda_df["Demanda promedio anual (kW)"]
                / demanda_df["Demanda contratada kW"]
                * 100,
                pd.NA
            )

            demanda_df["Nombre Comercial"] = (
                demanda_df["NOMBRE COMERCIAL"]
                if "NOMBRE COMERCIAL" in demanda_df.columns
                else demanda_df["CLIENTE"]
            )

            # --------------------------------------------------------
            # Tarifa visual de la tabla
            # --------------------------------------------------------
            # Debe usar TARIFA_FINAL:
            # parser_tarifa_match primero y DG como respaldo.

            if "TARIFA_FINAL" in demanda_df.columns:
                demanda_df["Tarifa"] = demanda_df["TARIFA_FINAL"]
            elif "parser_tarifa_match" in demanda_df.columns:
                demanda_df["Tarifa"] = demanda_df["parser_tarifa_match"]
            elif "tarifa_norm" in demanda_df.columns:
                demanda_df["Tarifa"] = demanda_df["tarifa_norm"]
            elif "TARIFA_ANALISIS" in demanda_df.columns:
                demanda_df["Tarifa"] = demanda_df["TARIFA_ANALISIS"]
            elif "TARIFA" in demanda_df.columns:
                demanda_df["Tarifa"] = demanda_df["TARIFA"]
            else:
                demanda_df["Tarifa"] = "SIN TARIFA"

            demanda_df["Tarifa"] = (
                normalize_tarifa_series(demanda_df["Tarifa"])
                .fillna("SIN TARIFA")
            )

            if demanda_df.empty:
                st.info(
                    "No hay locales con demanda contratada y demanda promedio anual suficientes "
                    "para este centro comercial."
                )

            else:
                orden_tarifas = {
                    "GDMTH": 1,
                    "GDMTO": 2,
                    "GDBT": 3,
                    "PDBT": 4
                }

                demanda_df["_orden_tarifa"] = (
                    demanda_df["Tarifa"]
                    .astype(str)
                    .str.upper()
                    .map(orden_tarifas)
                    .fillna(999)
                )

                demanda_df = demanda_df.sort_values(
                    [
                        "_orden_tarifa",
                        "Nombre Comercial"
                    ]
                ).reset_index(drop=True)

                st.markdown(
                    '<div class="subsection-title">Resumen de demanda contratada vs demanda promedio anual</div>',
                    unsafe_allow_html=True
                )

                col_dem_1, col_dem_2, col_dem_3, col_dem_4 = st.columns(4)

                col_dem_1.metric(
                    "Locales ocupados con recibo",
                    f"{len(demanda_df):,}"
                )

                st.caption(
                    f"Locales con demanda promedio anual calculada: "
                    f"{demanda_df['Demanda promedio anual (kW)'].notna().sum():,} de {len(demanda_df):,}"
                )

                col_dem_2.metric(
                    "Demanda contratada total kW",
                    format_number(demanda_df["Demanda contratada kW"].sum(), 1)
                )

                col_dem_3.metric(
                    "Demanda promedio anual (kW)",
                    format_number(demanda_df["Demanda promedio anual (kW)"].sum(), 1)
                )

                col_dem_4.metric(
                    "Uso promedio vs contratada",
                    f"{demanda_df['Promedio anual (%)'].mean():,.1f}%"
                )

                st.caption(NOTA_DEMANDA_PROMEDIO_ANUAL)

                # --------------------------------------------------------
                # Base para gráfica
                # --------------------------------------------------------
                # La gráfica debe mostrar todos los locales ocupados con recibo.
                # Si un local todavía no tiene demanda promedio anual calculada,
                # se grafica Promedio anual (%) = 0 temporalmente.

                chart_plot_df = demanda_df.copy()

                chart_plot_df["Promedio anual (%) gráfica"] = (
                    pd.to_numeric(
                        chart_plot_df["Promedio anual (%)"],
                        errors="coerce"
                    )
                    .fillna(0)
                )

                chart_df = chart_plot_df.set_index(
                    "Nombre Comercial"
                )[
                    [
                        "Contratada (%)",
                        "Promedio anual (%) gráfica"
                    ]
                ]

                chart_df = chart_df.rename(
                    columns={
                        "Promedio anual (%) gráfica": "Demanda promedio anual (%)"
                    }
                )

                fig, ax = plt.subplots(
                    figsize=(max(16, len(chart_df) * 0.35), 6)
                )

                chart_df.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_ylim(0, 200)

                promedio_anual_max_grafica = pd.to_numeric(
                    demanda_df["Promedio anual (%)"],
                    errors="coerce"
                ).max()

                if pd.notna(promedio_anual_max_grafica) and promedio_anual_max_grafica > 200:
                    st.caption(
                        f"Nota: existen locales con demanda promedio anual mayor a 200% de la contratada. "
                        f"El valor máximo es {promedio_anual_max_grafica:,.1f}%, por lo que la gráfica se recortó visualmente para mejorar la lectura."
                    )

                grupos_tarifa = (
                    demanda_df
                    .reset_index(drop=True)
                    .groupby("Tarifa", sort=False)
                    .apply(lambda x: (x.index.min(), x.index.max()))
                )

                for tarifa, (inicio, fin) in grupos_tarifa.items():
                    centro = (inicio + fin) / 2

                    # Grupo de tarifa debajo de los nombres de locales
                    # y antes del título del eje X.
                    ax.text(
                        centro,
                        -0.80,
                        tarifa,
                        ha="center",
                        va="top",
                        fontsize=11,
                        fontweight="bold",
                        transform=ax.get_xaxis_transform(),
                        clip_on=False
                    )

                    # Línea tenue que delimita el final de cada grupo tarifario.
                    # Empieza en el eje X y baja hasta la zona del nombre de tarifa.
                    ax.plot(
                        [fin + 0.5, fin + 0.5],
                        [0, -0.82],
                        transform=ax.get_xaxis_transform(),
                        color="#B8B8B8",
                        linewidth=1.0,
                        alpha=0.95,
                        clip_on=False,
                        zorder=5
                    )

                # Espacio para: nombres de locales + tarifa + título del eje X.
                fig.subplots_adjust(bottom=0.50)

                ax.set_ylabel("% de demanda contratada")
                ax.set_xlabel(
                    "Locales ocupados con recibo",
                    labelpad=35
                )
                ax.set_title("Demanda promedio anual como % de la demanda contratada")

                ax.axhline(
                    100,
                    linestyle="--",
                    linewidth=1
                )

                ax.tick_params(
                    axis="x",
                    rotation=90,
                    labelsize=7
                )

                st.pyplot(fig)

                # --------------------------------------------------------
                # Estatus de demanda para tabla de salida
                # --------------------------------------------------------

                demanda_df["estatus_demanda"] = "Calculada"

                demanda_df.loc[
                    demanda_df["Demanda promedio anual (kW)"].isna(),
                    "estatus_demanda"
                ] = "Sin demanda promedio anual calculada"

                if "Tarifa" in demanda_df.columns:
                    demanda_df.loc[
                        demanda_df["Tarifa"]
                        .astype(str)
                        .str.upper()
                        .isin(["GDMTH", "GDMTO", "GDBT"])
                        & demanda_df["Demanda promedio anual (kW)"].notna(),
                        "estatus_demanda"
                    ] = "Calculada con kwmax"

                    demanda_df.loc[
                        demanda_df["Tarifa"]
                        .astype(str)
                        .str.upper()
                        .eq("PDBT")
                        & demanda_df["Demanda promedio anual (kW)"].notna(),
                        "estatus_demanda"
                    ] = "Estimada PDBT con NREL"

                    demanda_df.loc[
                        demanda_df["Tarifa"]
                        .astype(str)
                        .str.upper()
                        .eq("PDBT")
                        & demanda_df["Demanda promedio anual (kW)"].isna(),
                        "estatus_demanda"
                    ] = "Pendiente perfil NREL"

                # --------------------------------------------------------
                # No. servicio para tabla
                # --------------------------------------------------------

                if "parser_no_servicio_match" in demanda_df.columns:
                    demanda_df["no_servicio_tabla"] = demanda_df["parser_no_servicio_match"]
                elif "no_servicio" in demanda_df.columns:
                    demanda_df["no_servicio_tabla"] = demanda_df["no_servicio"]
                else:
                    demanda_df["no_servicio_tabla"] = pd.NA

                demanda_display_cols = [
                    col for col in [
                        "Tarifa",
                        "Nombre Comercial",
                        "no_servicio_tabla",
                        "Demanda contratada kW",
                        "Demanda promedio anual (kW)",
                        "Promedio anual (%)",
                        "meses_con_demanda",
                        "estatus_demanda"
                    ]
                    if col in demanda_df.columns
                ]

                demanda_display = demanda_df[
                    demanda_display_cols
                ].copy()

                demanda_display = demanda_display.rename(
                    columns={
                        "no_servicio_tabla": "No. servicio",
                        "Demanda contratada kW": "Demanda contratada (kW)",
                        "meses_con_demanda": "Meses en muestra"
                    }
                )

                for col in [
                    "Demanda contratada (kW)",
                    "Demanda promedio anual (kW)",
                    "Promedio anual (%)"
                ]:
                    if col in demanda_display.columns:
                        demanda_display[col] = pd.to_numeric(
                            demanda_display[col],
                            errors="coerce"
                        ).round(1)

                if "Meses en muestra" in demanda_display.columns:
                    demanda_display["Meses en muestra"] = (
                        pd.to_numeric(
                            demanda_display["Meses en muestra"],
                            errors="coerce"
                        )
                        .round(0)
                        .astype("Int64")
                    )

                st.caption(
                    "La demanda contratada se toma de `demanda_contratada_kw`. "
                    + NOTA_DEMANDA_PROMEDIO_ANUAL
                )

        # ============================================================
        # Densidad de demanda
        # ============================================================

        st.markdown(
            """
            <div class="section-title">Densidad de demanda</div>
            """,
            unsafe_allow_html=True
        )

        if "benchmark_densidad_base" not in locals() or benchmark_densidad_base.empty:
            st.info(
                "No está disponible la base maestra de densidad de demanda."
            )

        else:
            densidad_df = benchmark_densidad_base.copy()

            # --------------------------------------------------------
            # Filtrar al centro comercial seleccionado
            # --------------------------------------------------------

            selected_cc_key = cc_key(selected_cc) if "selected_cc" in locals() else None

            if selected_cc_key:
                densidad_df = densidad_df[
                    densidad_df["_cc_key_reporte"] == selected_cc_key
                ].copy()

            if densidad_df.empty:
                st.info(
                    "No hay locales ocupados con recibo para el centro comercial seleccionado."
                )

            else:
                giros_disponibles = sorted(
                    densidad_df["Giro comercial densidad"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                )

                if not giros_disponibles:
                    st.info(
                        "No encontré giros comerciales para este centro comercial."
                    )

                else:
                    selected_giro_densidad = st.selectbox(
                        "Selecciona un giro comercial",
                        options=giros_disponibles,
                        key="giro_densidad_selector"
                    )

                    densidad_giro = densidad_df[
                        densidad_df["Giro comercial densidad"]
                        .astype(str)
                        .str.strip()
                        .eq(selected_giro_densidad)
                    ].copy()

                    # Esta debe ser la muestra total del giro:
                    # todos los locales ocupados con recibo.
                    densidad_grafica = densidad_giro.copy()

                    # --------------------------------------------------------
                    # Métricas
                    # --------------------------------------------------------

                    densidad_calculable = densidad_giro[
                        densidad_giro["Densidad de demanda W/m2"].notna()
                    ].copy()

                    col_den_1, col_den_2, col_den_3, col_den_4 = st.columns(4)

                    col_den_1.metric(
                        "Locales ocupados con recibo del giro",
                        f"{len(densidad_giro):,}"
                    )

                    col_den_2.metric(
                        "Locales con densidad calculable",
                        f"{len(densidad_calculable):,}"
                    )

                    col_den_3.metric(
                        "Densidad promedio W/m²",
                        format_number(
                            densidad_calculable["Densidad de demanda W/m2"].mean(),
                            1
                        ) if not densidad_calculable.empty else "—"
                    )

                    col_den_4.metric(
                        "Densidad mediana W/m²",
                        format_number(
                            densidad_calculable["Densidad de demanda W/m2"].median(),
                            1
                        ) if not densidad_calculable.empty else "—"
                    )

                    if densidad_giro.empty:
                        st.info(
                            "No hay locales para este giro comercial."
                        )

                    else:
                        # --------------------------------------------------------
                        # Orden por tarifa y nombre
                        # --------------------------------------------------------

                        orden_tarifas_densidad = {
                            "GDMTH": 1,
                            "GDMTO": 2,
                            "GDBT": 3,
                            "PDBT": 4,
                            "Sin tarifa": 999
                        }

                        densidad_grafica["_orden_tarifa"] = (
                            densidad_grafica["Tarifa"]
                            .astype(str)
                            .str.upper()
                            .map(orden_tarifas_densidad)
                            .fillna(999)
                        )

                        densidad_grafica = densidad_grafica.sort_values(
                            [
                                "_orden_tarifa",
                                "Nombre Comercial"
                            ]
                        ).reset_index(drop=True)

                        densidad_grafica["x"] = range(len(densidad_grafica))

                        promedio = densidad_calculable[
                            "Densidad de demanda W/m2"
                        ].mean()

                        desv = densidad_calculable[
                            "Densidad de demanda W/m2"
                        ].std()

                        if pd.isna(promedio):
                            promedio = 0

                        if pd.isna(desv):
                            desv = 0

                        fig, ax = plt.subplots(
                            figsize=(min(max(14, len(densidad_grafica) * 0.35), 28), 6)
                        )

                        # Puntos con densidad calculable
                        densidad_plot = densidad_grafica[
                            densidad_grafica["Densidad de demanda W/m2"].notna()
                        ].copy()

                        ax.scatter(
                            densidad_plot["x"],
                            densidad_plot["Densidad de demanda W/m2"],
                            label="Densidad de demanda"
                        )

                        # Puntos sin densidad calculable, para que visualmente
                        # aparezcan los 15 locales aunque alguno no tenga valor.
                        densidad_sin_valor = densidad_grafica[
                            densidad_grafica["Densidad de demanda W/m2"].isna()
                        ].copy()

                        if not densidad_sin_valor.empty:
                            ax.scatter(
                                densidad_sin_valor["x"],
                                [0] * len(densidad_sin_valor),
                                marker="x",
                                label="Sin densidad calculable"
                            )

                        if not densidad_calculable.empty:
                            # Promedio: rojo quemado
                            ax.axhline(
                                promedio,
                                color="#C00000",
                                linewidth=1.8,
                                label="Promedio"
                            )

                            # +/- 1 desviación estándar: verde
                            ax.axhline(
                                promedio + desv,
                                color="#009900",
                                linewidth=1.5,
                                label="+1 desv est"
                            )

                            ax.axhline(
                                max(promedio - desv, 0),
                                color="#009900",
                                linewidth=1.5,
                                label="-1 desv est"
                            )

                        ax.set_title(
                            f"Densidad de demanda para {selected_giro_densidad}"
                        )

                        ax.set_ylabel(
                            "Densidad de demanda (W/m²)"
                        )

                        ax.set_xticks(
                            densidad_grafica["x"]
                        )

                        ax.set_xticklabels(
                            densidad_grafica["Nombre Comercial"],
                            rotation=90,
                            fontsize=7
                        )

                        grupos_tarifa = (
                            densidad_grafica
                            .groupby("Tarifa", sort=False)
                            .apply(
                                lambda x: (
                                    x["x"].min(),
                                    x["x"].max()
                                )
                            )
                        )

                        # Línea inicial del primer grupo tarifario.
                        ax.plot(
                            [-0.5, -0.5],
                            [0, -0.60],
                            transform=ax.get_xaxis_transform(),
                            color="#B8B8B8",
                            linewidth=1.0,
                            alpha=0.95,
                            clip_on=False,
                            zorder=5
                        )

                        for tarifa, (inicio, fin) in grupos_tarifa.items():
                            centro = (inicio + fin) / 2

                            # Tarifa: debajo de los nombres de locales,
                            # pero más arriba que antes.
                            ax.text(
                                centro,
                                -0.50,
                                tarifa,
                                ha="center",
                                va="top",
                                fontsize=11,
                                fontweight="bold",
                                transform=ax.get_xaxis_transform(),
                                clip_on=False
                            )

                            # Línea divisoria al final de cada tarifa.
                            ax.plot(
                                [fin + 0.5, fin + 0.5],
                                [0, -0.60],
                                transform=ax.get_xaxis_transform(),
                                color="#B8B8B8",
                                linewidth=1.0,
                                alpha=0.95,
                                clip_on=False,
                                zorder=5
                            )

                        # Recupera más altura útil para el área de puntos,
                        # pero conserva espacio para tarifas, título y leyenda.
                        fig.set_size_inches(
                            min(
                                max(14, len(densidad_grafica) * 0.35),
                                28
                            ),
                            8.0
                        )

                        fig.subplots_adjust(bottom=0.48)

                        # No usamos labelpad porque puede mandar el texto
                        # demasiado abajo o dejarlo fuera de la figura.
                        ax.set_xlabel("")

                        # Título principal del eje X.
                        ax.text(
                            0.5,
                            -0.76,
                            "Locales ocupados con recibo",
                            ha="center",
                            va="top",
                            fontsize=10,
                            transform=ax.transAxes,
                            clip_on=False
                        )

                        # Leyenda debajo del título del eje X.
                        handles, labels = ax.get_legend_handles_labels()

                        fig.legend(
                            handles,
                            labels,
                            loc="lower center",
                            bbox_to_anchor=(0.5, 0.015),
                            ncol=4,
                            frameon=True
                        )

                        st.pyplot(fig)

                        # --------------------------------------------------------
                        # Tabla de detalle limpia
                        # --------------------------------------------------------

                        densidad_display = densidad_giro.copy()

                        # --------------------------------------------------------
                        # Nombre
                        # --------------------------------------------------------

                        if "Nombre Comercial" in densidad_display.columns:
                            densidad_display["Nombre"] = densidad_display["Nombre Comercial"]

                        elif "NOMBRE COMERCIAL" in densidad_display.columns:
                            densidad_display["Nombre"] = densidad_display["NOMBRE COMERCIAL"]

                        elif "Nombre comercial" in densidad_display.columns:
                            densidad_display["Nombre"] = densidad_display["Nombre comercial"]

                        elif "parser_recibos_subgroup_match" in densidad_display.columns:
                            densidad_display["Nombre"] = densidad_display["parser_recibos_subgroup_match"]

                        else:
                            densidad_display["Nombre"] = ""

                        # --------------------------------------------------------
                        # Giro
                        # --------------------------------------------------------

                        if "Giro comercial densidad" in densidad_display.columns:
                            densidad_display["Giro"] = densidad_display["Giro comercial densidad"]

                        elif "SUBGIRO_COMERCIAL" in densidad_display.columns:
                            densidad_display["Giro"] = densidad_display["SUBGIRO_COMERCIAL"]

                        elif "GIRO_COMERCIAL" in densidad_display.columns:
                            densidad_display["Giro"] = densidad_display["GIRO_COMERCIAL"]

                        elif "GIRO" in densidad_display.columns:
                            densidad_display["Giro"] = densidad_display["GIRO"]

                        else:
                            densidad_display["Giro"] = ""

                        # --------------------------------------------------------
                        # Tarifa ya resuelta en la base maestra
                        # --------------------------------------------------------
                        # No volvemos a decidir la tarifa aquí.
                        #
                        # densidad_giro viene de benchmark_densidad_base.
                        # En esa base, "Tarifa" ya se calculó desde TARIFA_FINAL.
                        # TARIFA_FINAL ya prioriza parser y usa DG solo como respaldo.

                        if "Tarifa" in densidad_display.columns:
                            densidad_display["tarifa"] = (
                                densidad_display["Tarifa"]
                                .fillna("SIN TARIFA")
                                .astype(str)
                                .str.upper()
                                .str.strip()
                            )
                        else:
                            densidad_display["tarifa"] = "SIN TARIFA"

                        # --------------------------------------------------------
                        # No servicio
                        # --------------------------------------------------------

                        if "no_servicio" in densidad_display.columns:
                            densidad_display["No servicio"] = densidad_display["no_servicio"]

                        elif "parser_no_servicio_match" in densidad_display.columns:
                            densidad_display["No servicio"] = densidad_display["parser_no_servicio_match"]

                        else:
                            densidad_display["No servicio"] = ""

                        # --------------------------------------------------------
                        # Fuente de cálculo de densidad
                        # --------------------------------------------------------
                        # Esta columna indica si la densidad se calculó con:
                        # - kwmax del recibo
                        # - perfiles NREL
                        # - o si no fue calculable.

                        densidad_display["Fuente densidad"] = "Sin densidad calculable"

                        densidad_display.loc[
                            densidad_display["tarifa"].isin(["GDMTH", "GDMTO", "GDBT"])
                            & densidad_display["Demanda promedio anual (kW)"].notna(),
                            "Fuente densidad"
                        ] = "kwmax del recibo"

                        densidad_display.loc[
                            densidad_display["tarifa"].eq("PDBT")
                            & densidad_display["Demanda promedio anual (kW)"].notna(),
                            "Fuente densidad"
                        ] = "Perfiles NREL"

                        # Si existe estatus_demanda, lo usamos como respaldo
                        if "estatus_demanda" in densidad_display.columns:

                            densidad_display.loc[
                                densidad_display["estatus_demanda"]
                                .fillna("")
                                .astype(str)
                                .str.contains("kwmax", case=False, na=False),
                                "Fuente densidad"
                            ] = "kwmax del recibo"

                            densidad_display.loc[
                                densidad_display["estatus_demanda"]
                                .fillna("")
                                .astype(str)
                                .str.contains("NREL", case=False, na=False),
                                "Fuente densidad"
                            ] = "Perfiles NREL"

                        # --------------------------------------------------------
                        # Columnas finales de la tabla
                        # --------------------------------------------------------

                        densidad_display_cols = [
                            col for col in [
                                "Nombre",
                                "Giro",
                                "tarifa",
                                "No servicio",
                                "Demanda promedio anual (kW)",
                                "Area m2",
                                "Densidad de demanda W/m2",
                                "criterio_union_demanda",
                                "estatus_demanda",
                                "zona_nrel",
                                "Fuente densidad"
                            ]
                            if col in densidad_display.columns
                        ]

                        densidad_display = densidad_display[
                            densidad_display_cols
                        ].copy()

                        for col in [
                            "Demanda promedio anual (kW)",
                            "Area m2",
                            "Densidad de demanda W/m2",
                        ]:
                            if col in densidad_display.columns:
                                densidad_display[col] = pd.to_numeric(
                                    densidad_display[col],
                                    errors="coerce"
                                ).round(2)

                        if "Densidad de demanda W/m2" in densidad_display.columns:
                            densidad_display = densidad_display.sort_values(
                                "Densidad de demanda W/m2",
                                ascending=False,
                                na_position="last"
                            )

                        st.dataframe(
                            densidad_display,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.caption(
                            "Esta sección usa la misma base maestra de densidad del Resumen Ejecutivo, "
                            "filtrada al centro comercial seleccionado y al giro comercial seleccionado. "
                            "La Densidad de demanda (W/m²) se calcula como Demanda promedio anual (kW) "
                            "/ m² × 1,000. "
                            + NOTA_DEMANDA_PROMEDIO_ANUAL
                        )


with tab_sg:

    st.markdown(
        '<div class="section-title">Servicios Generales</div>',
        unsafe_allow_html=True
    )

    cc_master_path = DATA_DIR / "profiles" / "cc_master_data.csv"

    if not cc_master_path.exists():
        st.warning(f"No encontré el archivo maestro: {cc_master_path}")

    else:
        cc_master_df = pd.read_csv(cc_master_path, encoding="latin1")
        cc_master_df.columns = cc_master_df.columns.str.strip()

        # ------------------------------------------------------------
        # Base validada de Servicios Generales
        # ------------------------------------------------------------
        # IMPORTANTE:
        # No usamos el parser crudo para identificar SG.
        # Usamos benchmark_densidad_base, que viene del match global
        # contra Datos Generales. Así el universo debe coincidir con
        # el No. de Servicios Generales del Resumen Ejecutivo.

        if (
            "benchmark_densidad_base" not in globals()
            or benchmark_densidad_base.empty
        ):
            st.warning(
                "No existe benchmark_densidad_base para construir Servicios Generales."
            )

        else:
            sg_df = benchmark_densidad_base.copy()

            giro_col_sg = first_existing_column(
                sg_df,
                [
                    "Giro comercial densidad",
                    "SUBGIRO_COMERCIAL",
                    "SUBGIRO COMERCIAL",
                    "GIRO_COMERCIAL",
                    "GIRO COMERCIAL",
                    "GIRO",
                    "Giro",
                    "Giro comercial"
                ]
            )

            if giro_col_sg is None:
                st.warning(
                    "No encontré columna de giro en benchmark_densidad_base para identificar Servicios Generales."
                )

            else:
                mask_sg = (
                    sg_df[giro_col_sg]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .str.contains(
                        r"SERVICIO[S]?\s+GENERAL(ES)?",
                        regex=True,
                        na=False
                    )
                )

                sg_df = sg_df[mask_sg].copy()

                if sg_df.empty:
                    st.info(
                        "No encontré Servicios Generales en la base validada del benchmark."
                    )

                else:
                    # ------------------------------------------------------------
                    # Columnas base desde la muestra validada
                    # ------------------------------------------------------------

                    cc_col = first_existing_column(
                        sg_df,
                        [
                            "_centro_comercial_limpio",
                            "NOMBRE DEL CC",
                            "CENTRO COMERCIAL",
                            "Centro Comercial",
                            "source_sheet"
                        ]
                    )

                    medidor_col = first_existing_column(
                        sg_df,
                        [
                            "parser_medidor_match",
                            "medidor",
                            "MEDIDOR",
                            "No. De medidor",
                            "No. de medidor"
                        ]
                    )

                    consumo_col = first_existing_column(
                        sg_df,
                        [
                            "kwh_12m",
                            "Consumo anual (kWh)",
                            "kwh_total",
                            "kwh_total_num"
                        ]
                    )

                    demanda_contratada_col = first_existing_column(
                        sg_df,
                        [
                            "demanda_contratada_kw",
                            "Demanda contratada kW",
                            "Demanda Contratada (kW)"
                        ]
                    )

                    demanda_promedio_col = first_existing_column(
                        sg_df,
                        [
                            "demanda_promedio_anual_kw",
                            "Demanda promedio anual (kW)"
                        ]
                    )

                    if cc_col is None:
                        st.warning(
                            "No encontré columna de centro comercial en Servicios Generales."
                        )

                    # Normalizar numéricos
                    sg_df["_consumo_kwh"] = (
                        pd.to_numeric(sg_df[consumo_col], errors="coerce")
                        if consumo_col
                        else np.nan
                    )

                    sg_df["_demanda_contratada_kw"] = (
                        pd.to_numeric(sg_df[demanda_contratada_col], errors="coerce")
                        if demanda_contratada_col
                        else np.nan
                    )

                    sg_df["_demanda_promedio_anual_kw"] = (
                        pd.to_numeric(sg_df[demanda_promedio_col], errors="coerce")
                        if demanda_promedio_col
                        else np.nan
                    )

                    # ------------------------------------------------------------
                    # Tabla de Servicios Generales por local
                    # ------------------------------------------------------------
                    # Cada fila representa un local/registro de Servicios Generales
                    # de la muestra validada. Ya no se suman los locales por CC.

                    nombre_sg_col = first_existing_column(
                        sg_df,
                        [
                            "NOMBRE COMERCIAL",
                            "Nombre Comercial",
                            "Nombre comercial",
                            "parser_recibos_subgroup_match",
                            "parser_recibos_subgroup",
                            "recibos_subgroup",
                            "CLIENTE",
                            "cliente_nombre"
                        ]
                    )

                    servicio_sg_col = first_existing_column(
                        sg_df,
                        [
                            "parser_no_servicio_match",
                            "no_servicio",
                            "No. servicio"
                        ]
                    )

                    # Llave de CC de cada local SG
                    sg_df["_cc_key_sg"] = sg_df[cc_col].apply(cc_key)

                    # ------------------------------------------------------------
                    # Metadatos del centro comercial
                    # ------------------------------------------------------------
                    cc_meta = cc_master_df.copy()

                    cc_meta["_cc_key_sg"] = cc_meta["Nombre Comercial"].apply(
                        cc_key
                    )

                    cc_meta["_cc_nombre_meta"] = cc_meta["Nombre Comercial"].apply(
                        limpiar_nombre_cc
                    )

                    cc_meta["_tipo_cc_meta"] = cc_meta.get(
                        "Tipo de Mall",
                        pd.Series("", index=cc_meta.index)
                    )

                    if "zona_nrel" in cc_meta.columns:
                        zona_meta = cc_meta["zona_nrel"]
                    elif "Zona NREL" in cc_meta.columns:
                        zona_meta = cc_meta["Zona NREL"]
                    else:
                        zona_meta = pd.Series("", index=cc_meta.index)

                    zona_meta = zona_meta.fillna("").astype(str).str.upper()

                    cc_meta["_clima_meta"] = np.select(
                        [
                            zona_meta.str.contains("HOT", na=False),
                            zona_meta.str.contains("MIXED", na=False),
                            zona_meta.str.contains("COLD", na=False)
                        ],
                        [
                            "Cálido",
                            "Templado",
                            "Frío"
                        ],
                        default="Sin clasificar"
                    )

                    abr_meta = cc_meta.get(
                        "Área Bruta Rentable (m²)",
                        pd.Series(np.nan, index=cc_meta.index)
                    )

                    cc_meta["_abr_meta"] = pd.to_numeric(
                        abr_meta.astype(str).str.replace(",", "", regex=False),
                        errors="coerce"
                    )

                    cc_meta = cc_meta[
                        [
                            "_cc_key_sg",
                            "_cc_nombre_meta",
                            "_tipo_cc_meta",
                            "_clima_meta",
                            "_abr_meta"
                        ]
                    ].drop_duplicates(subset=["_cc_key_sg"])

                    # ------------------------------------------------------------
                    # Integrar los metadatos del CC a cada local SG
                    # ------------------------------------------------------------
                    sg_locales_df = sg_df.merge(
                        cc_meta,
                        on="_cc_key_sg",
                        how="left"
                    )

                    sg_locales_df["Centro Comercial"] = (
                        sg_locales_df["_cc_nombre_meta"]
                        .fillna(
                            sg_locales_df[cc_col].apply(limpiar_nombre_cc)
                        )
                    )

                    if nombre_sg_col:
                        sg_locales_df["Local de Servicios Generales"] = (
                            sg_locales_df[nombre_sg_col]
                            .fillna("Sin nombre")
                            .astype(str)
                            .str.strip()
                            .replace(
                                {
                                    "": "Sin nombre",
                                    "nan": "Sin nombre",
                                    "None": "Sin nombre",
                                    "<NA>": "Sin nombre"
                                }
                            )
                        )
                    else:
                        sg_locales_df["Local de Servicios Generales"] = "Sin nombre"

                    if servicio_sg_col:
                        sg_locales_df["No. servicio"] = (
                            sg_locales_df[servicio_sg_col]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                        )
                    else:
                        sg_locales_df["No. servicio"] = ""

                    if medidor_col:
                        sg_locales_df["No. de medidor"] = (
                            sg_locales_df[medidor_col]
                            .fillna("")
                            .astype(str)
                            .str.strip()
                        )
                    else:
                        sg_locales_df["No. de medidor"] = ""

                    sg_locales_df["No. de medidores de Servicios Generales"] = (
                        sg_locales_df["No. de medidor"]
                        .ne("")
                        .astype(int)
                    )

                    # Densidades por local respecto al ABR del CC.
                    sg_locales_df["Densidad demanda (W/m² ABR)"] = np.where(
                        sg_locales_df["_abr_meta"].gt(0),
                        (
                            sg_locales_df["_demanda_promedio_anual_kw"]
                            / sg_locales_df["_abr_meta"]
                            * 1000
                        ),
                        np.nan
                    )

                    sg_locales_df["Densidad de Consumo Anual (kWh/m² ABR)"] = np.where(
                        sg_locales_df["_abr_meta"].gt(0),
                        (
                            sg_locales_df["_consumo_kwh"]
                            / sg_locales_df["_abr_meta"]
                        ),
                        np.nan
                    )

                    sg_resumen_df = pd.DataFrame({
                        "Centro Comercial": sg_locales_df["Centro Comercial"],
                        "Local de Servicios Generales": (
                            sg_locales_df["Local de Servicios Generales"]
                        ),
                        "No. servicio": sg_locales_df["No. servicio"],
                        "Tipo de Centro Comercial": sg_locales_df["_tipo_cc_meta"],
                        "Clima": sg_locales_df["_clima_meta"],
                        "No. de medidores de Servicios Generales": (
                            sg_locales_df[
                                "No. de medidores de Servicios Generales"
                            ]
                        ),
                        "Demanda Contratada (kW)": (
                            sg_locales_df["_demanda_contratada_kw"]
                        ),
                        "Demanda promedio anual (kW)": (
                            sg_locales_df["_demanda_promedio_anual_kw"]
                        ),
                        "Densidad demanda (W/m² ABR)": (
                            sg_locales_df["Densidad demanda (W/m² ABR)"]
                        ),
                        "Consumo Anual (kWh)": (
                            sg_locales_df["_consumo_kwh"]
                        ),
                        "Densidad de Consumo Anual (kWh/m² ABR)": (
                            sg_locales_df[
                                "Densidad de Consumo Anual (kWh/m² ABR)"
                            ]
                        )
                    })

                if sg_resumen_df.empty:
                    st.info(
                        "No se pudo construir la tabla resumen de Servicios Generales."
                    )

                else:
                    sg_resumen_display = sg_resumen_df.copy()

                    for col in [
                        "Demanda Contratada (kW)",
                        "Demanda promedio anual (kW)",
                        "Densidad demanda (W/m² ABR)",
                        "Consumo Anual (kWh)",
                        "Densidad de Consumo Anual (kWh/m² ABR)"
                    ]:
                        if col in sg_resumen_display.columns:
                            sg_resumen_display[col] = pd.to_numeric(
                                sg_resumen_display[col],
                                errors="coerce"
                            ).round(2)

                        sg_resumen_display[col] = sg_resumen_display[col].round(2)

                    # Orden climático
                    orden_clima = {
                        "Cálido": 1,
                        "Templado": 2,
                        "Frío": 3
                    }

                    # Orden de tipo de mall
                    orden_tipo_cc = {
                        "Luxury Fashion Mall": 1,
                        "Fashion Mall": 2,
                        "Regional Mall": 3,
                        "Power Center": 4,
                        "Strip Mall": 5
                    }

                    sg_resumen_display["_orden_clima"] = (
                        sg_resumen_display["Clima"]
                        .map(orden_clima)
                    )

                    sg_resumen_display["_orden_tipo"] = (
                        sg_resumen_display["Tipo de Centro Comercial"]
                        .map(orden_tipo_cc)
                    )

                    sg_resumen_display = (
                        sg_resumen_display
                        .sort_values(
                            [
                                "_orden_clima",
                                "_orden_tipo",
                                "Centro Comercial",
                                "Local de Servicios Generales"
                            ],
                            ascending=[
                                True,
                                True,
                                True,
                                True
                            ]
                        )
                        .drop(
                            columns=[
                                "_orden_clima",
                                "_orden_tipo"
                            ]
                        )
                    )

                    # ------------------------------------------------------------
                    # Fila TOTAL al final
                    # ------------------------------------------------------------
                    # La fila TOTAL siempre se agrega después del ordenamiento,
                    # para que aparezca hasta abajo.
                    #
                    # Se suman:
                    # - No. de medidores de Servicios Generales
                    # - Demanda Contratada (kW)
                    # - Demanda promedio anual (kW)
                    # - Consumo Anual (kWh)
                    #
                    # Las densidades del TOTAL se dejan vacías porque no se deben
                    # sumar ni promediar directamente.

                    total_servicios_generales = len(sg_resumen_df)

                    total_demanda_contratada = sg_resumen_df[
                        "Demanda Contratada (kW)"
                    ].sum(skipna=True)

                    total_demanda_promedio_anual = sg_resumen_df[
                        "Demanda promedio anual (kW)"
                    ].sum(skipna=True)

                    total_consumo_anual = sg_resumen_df[
                        "Consumo Anual (kWh)"
                    ].sum(skipna=True)

                    total_row_sg = {
                        "Centro Comercial": "TOTAL",
                        "Local de Servicios Generales": "",
                        "No. servicio": "",
                        "Tipo de Centro Comercial": "",
                        "Clima": "",
                        "No. de medidores de Servicios Generales": total_servicios_generales,
                        "Demanda Contratada (kW)": total_demanda_contratada,
                        "Demanda promedio anual (kW)": total_demanda_promedio_anual,
                        "Densidad demanda (W/m² ABR)": np.nan,
                        "Consumo Anual (kWh)": total_consumo_anual,
                        "Densidad de Consumo Anual (kWh/m² ABR)": np.nan
                    }

                    total_row_sg_df = pd.DataFrame([total_row_sg])

                    # Asegurar que la fila TOTAL tenga exactamente las mismas columnas
                    # que la tabla visible.
                    for col in sg_resumen_display.columns:
                        if col not in total_row_sg_df.columns:
                            total_row_sg_df[col] = ""

                    total_row_sg_df = total_row_sg_df[
                        sg_resumen_display.columns
                    ]

                    sg_resumen_display = pd.concat(
                        [
                            sg_resumen_display,
                            total_row_sg_df
                        ],
                        ignore_index=True
                    )

                    cols_sg_finales = [
                        "Centro Comercial",
                        "Local de Servicios Generales",
                        "No. servicio",
                        "Tipo de Centro Comercial",
                        "Clima",
                        "No. de medidores de Servicios Generales",
                        "Demanda Contratada (kW)",
                        "Demanda promedio anual (kW)",
                        "Densidad demanda (W/m² ABR)",
                        "Consumo Anual (kWh)",
                        "Densidad de Consumo Anual (kWh/m² ABR)"
                    ]

                    for col in cols_sg_finales:
                        if col not in sg_resumen_display.columns:
                            sg_resumen_display[col] = pd.NA

                    sg_resumen_display = sg_resumen_display[
                        cols_sg_finales
                    ]

                    # ------------------------------------------------------------
                    # Mostrar tabla sin scroll horizontal ni vertical
                    # ------------------------------------------------------------
                    # Usamos HTML en vez de st.dataframe para que la tabla se ajuste
                    # completa al ancho de la página.

                    sg_resumen_html = sg_resumen_display.copy()

                    # Formato visual de números
                    for col in [
                        "No. de medidores de Servicios Generales",
                        "Demanda Contratada (kW)",
                        "Demanda promedio anual (kW)",
                        "Densidad demanda (W/m² ABR)",
                        "Consumo Anual (kWh)",
                        "Densidad de Consumo Anual (kWh/m² ABR)"
                    ]:
                        if col in sg_resumen_html.columns:
                            sg_resumen_html[col] = pd.to_numeric(
                                sg_resumen_html[col],
                                errors="coerce"
                            )

                    if "No. de medidores de Servicios Generales" in sg_resumen_html.columns:
                        sg_resumen_html["No. de medidores de Servicios Generales"] = (
                            sg_resumen_html["No. de medidores de Servicios Generales"]
                            .map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
                        )

                    for col in [
                        "Demanda Contratada (kW)",
                        "Demanda promedio anual (kW)",
                        "Densidad demanda (W/m² ABR)",
                        "Consumo Anual (kWh)",
                        "Densidad de Consumo Anual (kWh/m² ABR)"
                    ]:
                        if col in sg_resumen_html.columns:
                            sg_resumen_html[col] = (
                                sg_resumen_html[col]
                                .map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
                            )

                    st.markdown(
                        """
                        <style>
                        .sg-table-container {
                            width: 100%;
                            overflow-x: visible;
                            overflow-y: visible;
                        }

                        .sg-table-container table {
                            width: 100%;
                            border-collapse: collapse;
                            table-layout: fixed;
                            font-size: 11px;
                        }

                        .sg-table-container th {
                            background-color: #f6f8fb;
                            color: #6b7280;
                            font-weight: 500;
                            border: 1px solid #e5e7eb;
                            padding: 6px 5px;
                            text-align: left;
                            white-space: normal;
                            word-wrap: break-word;
                        }

                        .sg-table-container td {
                            border: 1px solid #e5e7eb;
                            padding: 6px 5px;
                            vertical-align: top;
                            white-space: normal;
                            word-wrap: break-word;
                        }

                        .sg-table-container td:nth-child(6),
                        .sg-table-container td:nth-child(7),
                        .sg-table-container td:nth-child(8),
                        .sg-table-container td:nth-child(9),
                        .sg-table-container td:nth-child(10),
                        .sg-table-container td:nth-child(11) {
                            text-align: right;
                        }

                        .sg-table-container tr:last-child {
                            font-weight: 700;
                            background-color: #f9fafb;
                        }

                        .warning-box {
                            padding: 0.9rem 1rem;
                            border-left: 5px solid #F57C00;
                            background-color: #FFF8E1;
                            border-radius: 0.4rem;
                            margin-bottom: 1rem;
                            color: #333;
                        }

                        /* --------------------------------------------------
                        Tabs de apoyo: Calidad de Datos y Anexo en gris rata
                        -------------------------------------------------- */

                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(5),
                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(6),
                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(5) p,
                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(6) p {
                            color: #747474 !important;
                        }

                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"],
                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(6)[aria-selected="true"],
                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"] p,
                        .stTabs [data-baseweb="tab-list"] button:nth-of-type(6)[aria-selected="true"] p {
                            color: #4A4A4A !important;
                            font-weight: 900 !important;
                        }

                        .stTabs:has([data-baseweb="tab-list"] button:nth-of-type(5)[aria-selected="true"])
                        [data-baseweb="tab-highlight"] {
                            background-color: #747474 !important;
                        }

                        .stTabs:has([data-baseweb="tab-list"] button:nth-of-type(6)[aria-selected="true"])
                        [data-baseweb="tab-highlight"] {
                            background-color: #747474 !important;
                        }

                        </style>
                        
                        </style>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="sg-table-container">'
                        + sg_resumen_html.to_html(
                            index=False,
                            escape=False
                        )
                        + '</div>',
                        unsafe_allow_html=True
                    )

                    st.caption(
                        "La Densidad demanda (W/m² ABR) se calcula como Demanda promedio anual (kW) "
                        "/ ABR × 1,000. "
                        + NOTA_DEMANDA_PROMEDIO_ANUAL
                    )

                # ------------------------------------------------------------
                # Gráfica de densidad de demanda por medidor
                # ------------------------------------------------------------
                st.markdown(
                    '<div class="section-title">Demanda contratada vs demanda promedio anual de Servicios Generales</div>',
                    unsafe_allow_html=True
                )

                # ------------------------------------------------------------
                # Gráfica: demanda contratada vs demanda promedio anual
                # ------------------------------------------------------------
                # Usamos la misma tabla resumen por centro comercial.
                # No recalculamos demanda: usamos "Demanda promedio anual (kW)",
                # que ya viene de la base validada del benchmark.

                plot_df = sg_resumen_display[
                    sg_resumen_display["Centro Comercial"].ne("TOTAL")
                ].copy()

                plot_df["Demanda Contratada (kW)"] = pd.to_numeric(
                    plot_df["Demanda Contratada (kW)"],
                    errors="coerce"
                )

                plot_df["Demanda promedio anual (kW)"] = pd.to_numeric(
                    plot_df["Demanda promedio anual (kW)"],
                    errors="coerce"
                )

                plot_df = plot_df[
                    plot_df["Demanda Contratada (kW)"].notna()
                    & plot_df["Demanda Contratada (kW)"].gt(0)
                    & plot_df["Demanda promedio anual (kW)"].notna()
                ].copy()

                if plot_df.empty:
                    st.info(
                        "No hay información suficiente para graficar demanda contratada vs demanda promedio anual."
                    )

                else:
                    plot_df["Contratada (%)"] = 100

                    plot_df["Promedio anual (%)"] = (
                        plot_df["Demanda promedio anual (kW)"]
                        / plot_df["Demanda Contratada (kW)"]
                        * 100
                    )

                    max_promedio_anual_sg = plot_df["Promedio anual (%)"].max(skipna=True)

                    if pd.notna(max_promedio_anual_sg) and max_promedio_anual_sg > 200:
                        st.caption(
                            f"Nota: existen centros comerciales con demanda promedio anual mayor a 200% de la contratada. "
                            f"El valor máximo es {max_promedio_anual_sg:,.1f}%, por lo que la gráfica se recortó visualmente para mejorar la lectura."
                        )

                    plot_df["Promedio anual visual (%)"] = plot_df["Promedio anual (%)"].clip(
                        upper=200
                    )

                    plot_df = plot_df.sort_values(
                        [
                            "Tipo de Centro Comercial",
                            "Centro Comercial"
                        ]
                    ).reset_index(drop=True)

                plot_df = sg_resumen_display[
                    sg_resumen_display["Centro Comercial"].ne("TOTAL")
                ].copy()

                plot_df["Demanda Contratada (kW)"] = pd.to_numeric(
                    plot_df["Demanda Contratada (kW)"],
                    errors="coerce"
                )

                plot_df["Demanda promedio anual (kW)"] = pd.to_numeric(
                    plot_df["Demanda promedio anual (kW)"],
                    errors="coerce"
                )

                # Conserva locales que tengan al menos una de las dos demandas.
                plot_df = plot_df[
                    plot_df["Demanda Contratada (kW)"].notna()
                    | plot_df["Demanda promedio anual (kW)"].notna()
                ].copy()

                if plot_df.empty:
                    st.info(
                        "No hay información suficiente para graficar los locales de Servicios Generales."
                    )

                else:
                    plot_df = plot_df.sort_values(
                        [
                            "Clima",
                            "Tipo de Centro Comercial",
                            "Centro Comercial",
                            "Local de Servicios Generales"
                        ]
                    ).reset_index(drop=True)

                    plot_df["Etiqueta local"] = (
                        plot_df["Centro Comercial"]
                        .fillna("")
                        .astype(str)
                        .str.strip()
                        + "\n"
                        + plot_df["Local de Servicios Generales"]
                        .fillna("Sin nombre")
                        .astype(str)
                        .str.strip()
                    )

                    # --------------------------------------------------------
                    # Gráfica por local:
                    # Demanda contratada = 100%
                    # Demanda promedio anual = % respecto a la contratada
                    # --------------------------------------------------------

                    # Solo se pueden calcular porcentajes cuando existe
                    # una demanda contratada mayor a cero.
                    plot_df = plot_df[
                        plot_df["Demanda Contratada (kW)"].notna()
                        & plot_df["Demanda Contratada (kW)"].gt(0)
                    ].copy()

                    if plot_df.empty:
                        st.info(
                            "No hay locales de Servicios Generales con demanda "
                            "contratada suficiente para calcular el porcentaje."
                        )

                    else:
                        # La barra azul siempre representa 100% de la demanda contratada.
                        plot_df["Contratada (%)"] = 100.0

                        # La barra naranja representa el uso promedio anual
                        # respecto a la demanda contratada del mismo local.
                        plot_df["Demanda promedio anual (%)"] = (
                            plot_df["Demanda promedio anual (kW)"]
                            / plot_df["Demanda Contratada (kW)"]
                            * 100
                        )

                        # Conservamos los locales sin demanda promedio anual:
                        # aparecen con barra naranja en cero.
                        plot_df["Demanda promedio anual (%)"] = (
                            pd.to_numeric(
                                plot_df["Demanda promedio anual (%)"],
                                errors="coerce"
                            )
                            .fillna(0)
                        )

                        max_promedio_anual_sg = (
                            plot_df["Demanda promedio anual (%)"]
                            .max(skipna=True)
                        )

                        if (
                            pd.notna(max_promedio_anual_sg)
                            and max_promedio_anual_sg > 200
                        ):
                            st.caption(
                                f"Nota: existen locales de Servicios Generales con "
                                f"demanda promedio anual mayor a 200% de la contratada. "
                                f"El valor máximo es "
                                f"{max_promedio_anual_sg:,.1f}%, por lo que la gráfica "
                                f"se recortó visualmente para mejorar la lectura."
                            )

                        # Se recorta solo visualmente; el valor original
                        # se conserva arriba para la nota.
                        plot_df["Demanda promedio anual visual (%)"] = (
                            plot_df["Demanda promedio anual (%)"]
                            .clip(upper=200)
                        )

                        x = np.arange(len(plot_df))
                        width = 0.38

                        fig, ax = plt.subplots(
                            figsize=(max(16, len(plot_df) * 0.65), 6)
                        )

                        ax.bar(
                            x - width / 2,
                            plot_df["Contratada (%)"],
                            width,
                            label="Contratada (%)"
                        )

                        ax.bar(
                            x + width / 2,
                            plot_df["Demanda promedio anual visual (%)"],
                            width,
                            label="Demanda promedio anual (%)"
                        )

                        ax.axhline(
                            100,
                            linestyle="--",
                            linewidth=1
                        )

                        ax.set_ylim(0, 120)

                        ax.set_ylabel(
                            "% de demanda contratada"
                        )

                        ax.set_xlabel(
                            "Locales de Servicios Generales"
                        )

                        ax.set_title(
                            "Demanda promedio anual como % de la demanda contratada"
                        )

                        ax.set_xticks(x)

                        ax.set_xticklabels(
                            plot_df["Etiqueta local"],
                            rotation=90,
                            ha="center",
                            fontsize=7
                        )

                        ax.legend()

                        fig.subplots_adjust(bottom=0.42)

                        st.pyplot(fig)

                        st.caption(NOTA_DEMANDA_PROMEDIO_ANUAL)


# ============================================================
# Data quality
# ============================================================
with tab_calidad:

    st.markdown(
        '<div class="section-title">Diagnóstico de carga de datos</div>',
        unsafe_allow_html=True
    )

    filas_originales = len(parsed_raw) if "parsed_raw" in locals() else None
    filas_preparadas = len(parsed) if "parsed" in locals() else None

    if filas_originales is not None and filas_preparadas is not None:

        filas_descartadas = filas_originales - filas_preparadas

        conservacion_pct = (
            filas_preparadas / filas_originales * 100
            if filas_originales > 0
            else 0
        )

        diagnostico_carga_df = pd.DataFrame({
            "Indicador": [
                "Filas CSV originales",
                "Filas después de preparación",
                "Filas descartadas durante preparación",
                "% conservación de registros"
            ],
            "Valor": [
                filas_originales,
                filas_preparadas,
                filas_descartadas,
                f"{conservacion_pct:.1f}%"
            ]
        })

        st.dataframe(
            diagnostico_carga_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No encontré las variables de filas originales y preparadas.")

    st.markdown(
        '<div class="subsection-title">Diagnóstico de demanda base por tarifa</div>',
        unsafe_allow_html=True
    )

    if "tarifa_norm" in parsed.columns:

        demanda_diag = (
            parsed
            .groupby("tarifa_norm")
            .agg(
                recibos=("tarifa_norm", "size"),
                recibos_con_kwmax=("kwmax_num", lambda x: x.notna().sum()),
                recibos_con_demanda_real=("demanda_real_kw", lambda x: x.notna().sum())
            )
            .reset_index()
            .rename(columns={
                "tarifa_norm": "Tarifa",
                "recibos": "Recibos",
                "recibos_con_kwmax": "Recibos con kwmax",
                "recibos_con_demanda_real": "Recibos con demanda base mensual"
            })
        )

        demanda_diag["% con kwmax"] = (
            demanda_diag["Recibos con kwmax"]
            / demanda_diag["Recibos"]
            * 100
        ).round(1)

        demanda_diag["% con demanda base mensual"] = (
            demanda_diag["Recibos con demanda base mensual"]
            / demanda_diag["Recibos"]
            * 100
        ).round(1)

        st.dataframe(
            demanda_diag,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Para GDMTH, GDMTO y GDBT la demanda base mensual se toma de `kwmax`. "
            "Para PDBT la demanda base mensual se estima con perfiles NREL cuando existe información suficiente. "
            "Esa demanda base mensual alimenta el cálculo de Demanda promedio anual (kW)."
        )

    else:
        st.info("Aún no existe la columna `tarifa_norm` para diagnosticar demanda base mensual.")

    st.markdown(
        '<div class="section-title">Representatividad de la muestra</div>',
        unsafe_allow_html=True
    )

    universo_cc = 119
    muestra_cc = 19
    cobertura_pct = muestra_cc / universo_cc * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Centros Comerciales analizados", f"{muestra_cc}")

    with col2:
        st.metric("Universo total", f"{universo_cc}")

    with col3:
        st.metric("Cobertura", f"{cobertura_pct:.1f}%")

    representatividad_df = pd.DataFrame({
        "Indicador": [
            "Nivel de confianza asumido",
            "Margen de error estimado",
            "Muestra recomendada (±10%)",
            "Muestra recomendada (±5%)"
        ],
        "Valor": [
            "95%",
            "±20%",
            "54 CC",
            "91 CC"
        ]
    })

    st.dataframe(
        representatividad_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "La muestra actual cubre 19 de 119 centros comerciales "
        f"({cobertura_pct:.1f}% del universo). "
        "Bajo un supuesto de 95% de confianza, "
        "el margen de error estimado es de aproximadamente ±20%, "
        "por lo que los resultados deben interpretarse como una muestra exploratoria "
        "y no como una representación estadística completa del universo."
    )

    quality_cols = []

    for col in [
        "cliente_nombre",
        "no_servicio",
        "cuenta",
        "rmu",
        "tarifa",
        "medidor",
        "periodo_inicio",
        "periodo_fin",
        "kwh_total",
        "importe_total",
    ]:
        if col in filtered.columns:
            non_null = filtered[col].notna().sum()
            coverage = 100 * non_null / len(filtered) if len(filtered) else 0
            quality_cols.append({
                "campo": col,
                "valores_no_vacios": non_null,
                "total_recibos": len(filtered),
                "cobertura_%": round(coverage, 2),
            })

    quality_df = pd.DataFrame(quality_cols)

    # ------------------------------------------------------------
    # Cobertura global confirmada
    # ------------------------------------------------------------
    # IMPORTANTE:
    # No recalculamos la cobertura dentro del tab Calidad de Datos.
    # Mostramos los resultados ya generados por construir_muestra_con_recibo_global(),
    # donde un local solo cuenta como "con recibo" si logró copiar datos reales
    # del parser: no_servicio, medidor, tarifa o file_path.

    st.markdown(
        '<div class="section-title">Cobertura de muestra por centro comercial</div>',
        unsafe_allow_html=True
    )

    if "coverage_by_mall" in globals() and not coverage_by_mall.empty:
        st.dataframe(
            coverage_by_mall,
            use_container_width=True,
            hide_index=True,
            height=min(900, (len(coverage_by_mall) + 2) * 35)
        )
    else:
        st.info("No se encontraron centros comerciales con cobertura calculable.")

    # ------------------------------------------------------------
    # Diagnóstico de match por criterio global
    # ------------------------------------------------------------

    st.markdown(
        '<div class="subsection-title">Diagnóstico de match por criterio</div>',
        unsafe_allow_html=True
    )

    if "match_summary_df" in globals() and not match_summary_df.empty:
        st.dataframe(
            match_summary_df.sort_values("Centro Comercial").reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=min(900, (len(match_summary_df) + 2) * 35)
        )
    else:
        st.info("No se pudo construir el diagnóstico de match por criterio.")


    # ------------------------------------------------------------
    # Matches amplios no confirmados
    # ------------------------------------------------------------

    st.markdown(
        '<div class="subsection-title">Diagnóstico de matches amplios no confirmados</div>',
        unsafe_allow_html=True
    )

    if "debug_sin_tarifa_df" in globals() and not debug_sin_tarifa_df.empty:
        st.warning(
            f"{len(debug_sin_tarifa_df)} locales tuvieron coincidencia amplia por cliente o nombre comercial, "
            "pero no se confirmaron porque no se copió no_servicio, medidor, tarifa ni file_path del parser. "
            "Estos locales NO cuentan como ocupados con recibo."
        )

        st.dataframe(
            debug_sin_tarifa_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )
    else:
        st.success(
            "No hay matches amplios no confirmados."
        )

    # ------------------------------------------------------------
    # Locales ocupados con recibo con información incompleta de demanda
    # ------------------------------------------------------------

    st.markdown(
        '<div class="subsection-title">Locales ocupados con recibo con información incompleta de demanda</div>',
        unsafe_allow_html=True
    )

    if "benchmark_densidad_base" in globals() and not benchmark_densidad_base.empty:

        demanda_incompleta_base = benchmark_densidad_base.copy()

        # --------------------------------------------------------
        # Centro comercial
        # --------------------------------------------------------

        if "_cc_key_reporte" in demanda_incompleta_base.columns:
            demanda_incompleta_base["Centro Comercial"] = demanda_incompleta_base["_cc_key_reporte"]

        elif "_centro_comercial_limpio" in demanda_incompleta_base.columns:
            demanda_incompleta_base["Centro Comercial"] = (
                demanda_incompleta_base["_centro_comercial_limpio"].apply(cc_key)
            )

        elif "NOMBRE DEL CC" in demanda_incompleta_base.columns:
            demanda_incompleta_base["Centro Comercial"] = (
                demanda_incompleta_base["NOMBRE DEL CC"].apply(cc_key)
            )

        elif "source_sheet" in demanda_incompleta_base.columns:
            demanda_incompleta_base["Centro Comercial"] = (
                demanda_incompleta_base["source_sheet"].apply(cc_key)
            )

        else:
            demanda_incompleta_base["Centro Comercial"] = ""

        # --------------------------------------------------------
        # Nombre comercial
        # --------------------------------------------------------

        if "Nombre Comercial" not in demanda_incompleta_base.columns:

            if "NOMBRE COMERCIAL" in demanda_incompleta_base.columns:
                demanda_incompleta_base["Nombre Comercial"] = demanda_incompleta_base["NOMBRE COMERCIAL"]

            elif "CLIENTE" in demanda_incompleta_base.columns:
                demanda_incompleta_base["Nombre Comercial"] = demanda_incompleta_base["CLIENTE"]

            elif "cliente_nombre" in demanda_incompleta_base.columns:
                demanda_incompleta_base["Nombre Comercial"] = demanda_incompleta_base["cliente_nombre"]

            else:
                demanda_incompleta_base["Nombre Comercial"] = ""

        # --------------------------------------------------------
        # No. servicio
        # --------------------------------------------------------

        if "no_servicio" not in demanda_incompleta_base.columns:
            demanda_incompleta_base["no_servicio"] = pd.NA

        if "parser_no_servicio_match" in demanda_incompleta_base.columns:

            servicio_faltante = (
                demanda_incompleta_base["no_servicio"].isna()
                | demanda_incompleta_base["no_servicio"]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>"])
            )

            servicio_parser_valido = (
                demanda_incompleta_base["parser_no_servicio_match"].notna()
                & ~demanda_incompleta_base["parser_no_servicio_match"]
                .astype(str)
                .str.upper()
                .isin(["", "NAN", "NONE", "<NA>"])
            )

            demanda_incompleta_base.loc[
                servicio_faltante & servicio_parser_valido,
                "no_servicio"
            ] = demanda_incompleta_base.loc[
                servicio_faltante & servicio_parser_valido,
                "parser_no_servicio_match"
            ]

        # --------------------------------------------------------
        # Tarifa
        # --------------------------------------------------------

        if "Tarifa" not in demanda_incompleta_base.columns:

            if "TARIFA_FINAL" in demanda_incompleta_base.columns:
                demanda_incompleta_base["Tarifa"] = demanda_incompleta_base["TARIFA_FINAL"]

            elif "parser_tarifa_match" in demanda_incompleta_base.columns:
                demanda_incompleta_base["Tarifa"] = demanda_incompleta_base["parser_tarifa_match"]

            elif "tarifa_norm" in demanda_incompleta_base.columns:
                demanda_incompleta_base["Tarifa"] = demanda_incompleta_base["tarifa_norm"]

            else:
                demanda_incompleta_base["Tarifa"] = "SIN TARIFA"

        demanda_incompleta_base["Tarifa"] = normalize_tarifa_series(
            demanda_incompleta_base["Tarifa"]
        ).fillna("SIN TARIFA")

        # --------------------------------------------------------
        # Área
        # --------------------------------------------------------

        if "Area m2" not in demanda_incompleta_base.columns:
            demanda_incompleta_base["Area m2"] = pd.NA

        demanda_incompleta_base["Area m2"] = pd.to_numeric(
            demanda_incompleta_base["Area m2"],
            errors="coerce"
        )

        # --------------------------------------------------------
        # Preparar parser para diagnóstico por recibo / file_path
        # --------------------------------------------------------

        parsed_demanda_diag = parsed.copy()

        if mall_col and mall_col in parsed_demanda_diag.columns:
            parsed_demanda_diag["Centro Comercial"] = (
                parsed_demanda_diag[mall_col].apply(cc_key)
            )
        else:
            parsed_demanda_diag["Centro Comercial"] = ""

        parsed_demanda_diag["_key_no_servicio_debug"] = (
            normalize_service_cc(parsed_demanda_diag["no_servicio"])
            if "no_servicio" in parsed_demanda_diag.columns
            else ""
        )

        parsed_demanda_diag["_key_no_servicio_debug_sin_ceros"] = (
            parsed_demanda_diag["_key_no_servicio_debug"]
            .fillna("")
            .astype(str)
            .str.lstrip("0")
        )

        if "file_path" not in parsed_demanda_diag.columns:
            parsed_demanda_diag["file_path"] = ""

        if "file_name" not in parsed_demanda_diag.columns:
            parsed_demanda_diag["file_name"] = ""

        parsed_demanda_diag["_kwh_total_num_diag"] = (
            clean_number_series(parsed_demanda_diag["kwh_total"])
            if "kwh_total" in parsed_demanda_diag.columns
            else pd.NA
        )

        parsed_demanda_diag["_kwmax_num_diag"] = (
            clean_number_series(parsed_demanda_diag["kwmax"])
            if "kwmax" in parsed_demanda_diag.columns
            else pd.NA
        )

        parsed_demanda_diag["_demanda_contratada_parser_num"] = (
            clean_number_series(parsed_demanda_diag["demanda_contratada_kw"])
            if "demanda_contratada_kw" in parsed_demanda_diag.columns
            else pd.NA
        )

        parsed_demanda_diag["_demanda_real_fila_kw"] = (
            pd.to_numeric(
                parsed_demanda_diag["demanda_maxima_mensual_kw"],
                errors="coerce"
            )
            if "demanda_maxima_mensual_kw" in parsed_demanda_diag.columns
            else pd.NA
        )

        # --------------------------------------------------------
        # Diagnóstico por servicio:
        # detectar valores sospechosos repetidos en todas las filas
        # --------------------------------------------------------
        # Para PDBT revisamos kWh_total.
        # Para GDMTH/GDMTO/GDBT revisamos kwmax.
        #
        # Si un mismo servicio tiene el mismo número en todos sus recibos,
        # lo marcamos como sospechoso porque probablemente fue error de lectura.

        parsed_demanda_diag["_n_recibos_servicio_diag"] = 0
        parsed_demanda_diag["_kwmax_unicos_servicio_diag"] = pd.NA
        parsed_demanda_diag["_kwmax_constante_servicio_diag"] = False
        parsed_demanda_diag["_kwmax_constante_valor_diag"] = pd.NA
        parsed_demanda_diag["_kwh_unicos_servicio_diag"] = pd.NA
        parsed_demanda_diag["_kwh_constante_servicio_diag"] = False
        parsed_demanda_diag["_kwh_constante_valor_diag"] = pd.NA

        if (
            "Centro Comercial" in parsed_demanda_diag.columns
            and "_key_no_servicio_debug" in parsed_demanda_diag.columns
        ):

            group_cols_demanda_diag = [
                "Centro Comercial",
                "_key_no_servicio_debug"
            ]

            servicio_stats_diag = (
                parsed_demanda_diag
                .groupby(group_cols_demanda_diag, dropna=False)
                .agg(
                    _n_recibos_servicio_diag=(
                        "file_path",
                        "nunique"
                    ),
                    _kwmax_unicos_servicio_diag=(
                        "_kwmax_num_diag",
                        lambda x: x.dropna().nunique()
                    ),
                    _kwmax_constante_valor_diag=(
                        "_kwmax_num_diag",
                        lambda x: (
                            x.dropna().iloc[0]
                            if len(x.dropna()) > 0
                            else pd.NA
                        )
                    ),
                    _kwh_unicos_servicio_diag=(
                        "_kwh_total_num_diag",
                        lambda x: x.dropna().nunique()
                    ),
                    _kwh_constante_valor_diag=(
                        "_kwh_total_num_diag",
                        lambda x: (
                            x.dropna().iloc[0]
                            if len(x.dropna()) > 0
                            else pd.NA
                        )
                    )
                )
                .reset_index()
            )

            servicio_stats_diag["_kwmax_constante_servicio_diag"] = (
                servicio_stats_diag["_n_recibos_servicio_diag"].ge(3)
                & servicio_stats_diag["_kwmax_unicos_servicio_diag"].eq(1)
                & pd.to_numeric(
                    servicio_stats_diag["_kwmax_constante_valor_diag"],
                    errors="coerce"
                ).notna()
            )

            servicio_stats_diag["_kwh_constante_servicio_diag"] = (
                servicio_stats_diag["_n_recibos_servicio_diag"].ge(3)
                & servicio_stats_diag["_kwh_unicos_servicio_diag"].eq(1)
                & pd.to_numeric(
                    servicio_stats_diag["_kwh_constante_valor_diag"],
                    errors="coerce"
                ).notna()
            )

            parsed_demanda_diag = parsed_demanda_diag.merge(
                servicio_stats_diag,
                on=group_cols_demanda_diag,
                how="left",
                suffixes=("", "_servicio")
            )

        if "tarifa_norm" in parsed_demanda_diag.columns:
            parsed_demanda_diag["_tarifa_diag"] = normalize_tarifa_series(
                parsed_demanda_diag["tarifa_norm"]
            )
        elif "tarifa" in parsed_demanda_diag.columns:
            parsed_demanda_diag["_tarifa_diag"] = normalize_tarifa_series(
                parsed_demanda_diag["tarifa"]
            )
        else:
            parsed_demanda_diag["_tarifa_diag"] = pd.NA

        # --------------------------------------------------------
        # Función auxiliar para separar no_servicio cuando venga con |
        # --------------------------------------------------------

        def split_servicios_demanda_incompleta(value):
            if pd.isna(value):
                return []

            servicios = []

            for raw in str(value).split("|"):
                raw = raw.strip()

                if raw.upper() in ["", "NAN", "NONE", "<NA>"]:
                    continue

                servicio_key = normalize_service_cc(
                    pd.Series([raw])
                ).iloc[0]

                if servicio_key:
                    servicios.append(servicio_key)

            return sorted(set(servicios))

        # --------------------------------------------------------
        # Construir tabla: una fila por recibo / file_path con problema
        # --------------------------------------------------------

        filas_demanda_incompleta = []

        for _, row_local in demanda_incompleta_base.iterrows():

            servicios_local = split_servicios_demanda_incompleta(
                row_local.get("no_servicio", "")
            )

            if not servicios_local:
                servicios_local = split_servicios_demanda_incompleta(
                    row_local.get("parser_no_servicio_match", "")
                )

            servicios_local_sin_ceros = [
                str(s).lstrip("0")
                for s in servicios_local
                if str(s).strip() != ""
            ]

            base_row = {
                "Centro Comercial": row_local.get("Centro Comercial", ""),
                "Nombre Comercial": row_local.get("Nombre Comercial", ""),
                "no_servicio": row_local.get("no_servicio", ""),
                "Tarifa": row_local.get("Tarifa", ""),
                "Demanda contratada kW": pd.NA,
                "Demanda base mensual kW": pd.NA,
                "Area m2": row_local.get("Area m2", pd.NA),
                "file_path": "",
                "file_name": "",
                "Motivo": ""
            }

            if not servicios_local:
                continue

            mask_parser_local = (
                parsed_demanda_diag["Centro Comercial"]
                .astype(str)
                .eq(str(row_local.get("Centro Comercial", "")))
                & (
                    parsed_demanda_diag["_key_no_servicio_debug"]
                    .isin(servicios_local)
                    |
                    parsed_demanda_diag["_key_no_servicio_debug_sin_ceros"]
                    .isin(servicios_local_sin_ceros)
                )
            )

            parser_files_local = parsed_demanda_diag[
                mask_parser_local
            ].copy()

            if parser_files_local.empty:
                continue

            parser_files_local = (
                parser_files_local
                .drop_duplicates(subset=["file_path"])
                .sort_values("file_path")
            )

            for _, row_file in parser_files_local.iterrows():

                tarifa_archivo = row_file.get(
                    "_tarifa_diag",
                    row_local.get("Tarifa", "")
                )

                tarifa_archivo = normalize_tarifa_value(tarifa_archivo)

                demanda_real_fila_kw = row_file.get(
                    "_demanda_real_fila_kw",
                    pd.NA
                )

                demanda_contratada_parser = row_file.get(
                    "_demanda_contratada_parser_num",
                    pd.NA
                )

                kwh_total_fila = row_file.get(
                    "_kwh_total_num_diag",
                    pd.NA
                )

                kwmax_fila = row_file.get(
                    "_kwmax_num_diag",
                    pd.NA
                )

                motivos = []

                # ------------------------------------------------
                # 1) Falta demanda real del recibo
                # ------------------------------------------------

                demanda_real_incompleta = (
                    pd.isna(demanda_real_fila_kw)
                    or demanda_real_fila_kw <= 0
                )

                if demanda_real_incompleta:

                    if str(tarifa_archivo).upper() == "PDBT":

                        if pd.isna(kwh_total_fila) or kwh_total_fila <= 0:
                            motivos.append(
                                "Sin demanda base mensual: PDBT sin kWh_total válido en parser"
                            )
                        else:
                            motivos.append(
                                "Sin demanda base mensual: PDBT sin estimación NREL"
                            )

                    elif str(tarifa_archivo).upper() in ["GDMTH", "GDMTO", "GDBT"]:

                        if pd.isna(kwmax_fila) or kwmax_fila <= 0:
                            motivos.append(
                                "Sin demanda base mensual: tarifa con demanda medida sin kwmax válido en parser"
                            )
                        else:
                            motivos.append(
                                "Sin demanda base mensual: no se calculó demanda mensual"
                            )

                    else:
                        motivos.append(
                            "Sin demanda base mensual: tarifa no identificada o sin dato de demanda"
                        )

                # ------------------------------------------------
                # 1B) Valor sospechoso repetido en todas las filas
                # ------------------------------------------------
                # Esto detecta errores de lectura como:
                # - kwmax = 1 en todos los recibos GDMTO/GDMTH/GDBT
                # - kWh_total idéntico en todos los recibos PDBT

                n_recibos_servicio = row_file.get(
                    "_n_recibos_servicio_diag",
                    pd.NA
                )

                kwmax_constante_servicio = bool(
                    row_file.get(
                        "_kwmax_constante_servicio_diag",
                        False
                    )
                )

                kwmax_constante_valor = row_file.get(
                    "_kwmax_constante_valor_diag",
                    pd.NA
                )

                kwh_constante_servicio = bool(
                    row_file.get(
                        "_kwh_constante_servicio_diag",
                        False
                    )
                )

                kwh_constante_valor = row_file.get(
                    "_kwh_constante_valor_diag",
                    pd.NA
                )

                if (
                    str(tarifa_archivo).upper() == "PDBT"
                    and kwh_constante_servicio
                    and pd.notna(kwh_constante_valor)
                    and n_recibos_servicio >= 3
                ):
                    motivos.append(
                        f"Valor sospechoso: PDBT con kWh_total repetido en todos los recibos del servicio ({kwh_constante_valor})"
                    )

                if (
                    str(tarifa_archivo).upper() in ["GDMTH", "GDMTO", "GDBT"]
                    and kwmax_constante_servicio
                    and pd.notna(kwmax_constante_valor)
                    and n_recibos_servicio >= 3
                ):
                    motivos.append(
                        f"Valor sospechoso: tarifa con demanda medida con kwmax repetido en todos los recibos del servicio ({kwmax_constante_valor})"
                    )

                # ------------------------------------------------
                # 2) Falta demanda contratada en parser
                # ------------------------------------------------
                # Excepción: PDBT sin demanda contratada usa default 25 kW.

                if str(tarifa_archivo).upper() == "PDBT":

                    demanda_contratada_mostrar = (
                        demanda_contratada_parser
                        if pd.notna(demanda_contratada_parser)
                        else 25
                    )

                else:

                    demanda_contratada_mostrar = demanda_contratada_parser

                    if pd.isna(demanda_contratada_parser) or demanda_contratada_parser <= 0:
                        motivos.append(
                            "Sin demanda contratada registrada en parser"
                        )

                # Si el recibo no tiene problema, no entra a la tabla.
                if not motivos:
                    continue

                fila = base_row.copy()

                fila["Tarifa"] = tarifa_archivo
                fila["Demanda contratada kW"] = demanda_contratada_mostrar
                fila["Demanda base mensual kW"] = demanda_real_fila_kw
                fila["file_path"] = row_file.get("file_path", "")
                fila["file_name"] = row_file.get("file_name", "")
                fila["Motivo"] = " | ".join(motivos)

                filas_demanda_incompleta.append(fila)

        demanda_incompleta_display = pd.DataFrame(filas_demanda_incompleta)

        if not demanda_incompleta_display.empty:

            demanda_incompleta_display = (
                demanda_incompleta_display
                .sort_values(
                    [
                        "Centro Comercial",
                        "Nombre Comercial",
                        "file_path"
                    ],
                    na_position="last"
                )
                .reset_index(drop=True)
            )

            for col_num in [
                "Demanda contratada kW",
                "Demanda base mensual kW",
                "Area m2"
            ]:
                if col_num in demanda_incompleta_display.columns:
                    demanda_incompleta_display[col_num] = pd.to_numeric(
                        demanda_incompleta_display[col_num],
                        errors="coerce"
                    ).round(2)

            # --------------------------------------------------------
            # Conteo de locales únicos con información incompleta
            # --------------------------------------------------------

            local_key_cols_demanda = [
                "Centro Comercial",
                "Nombre Comercial",
                "no_servicio"
            ]

            locales_incompletos_unicos = (
                demanda_incompleta_display[
                    local_key_cols_demanda
                ]
                .drop_duplicates()
                .shape[0]
            )

            archivos_incompletos_unicos = (
                demanda_incompleta_display["file_path"]
                .replace("", pd.NA)
                .dropna()
                .nunique()
                if "file_path" in demanda_incompleta_display.columns
                else 0
            )

            # --------------------------------------------------------
            # Desglose por tipo de problema
            # --------------------------------------------------------
            # La tabla tiene una fila por file_path.
            # Para el warning agrupamos por local único y juntamos todos
            # los motivos encontrados en sus recibos.

            motivos_por_local = (
                demanda_incompleta_display
                .groupby(
                    local_key_cols_demanda,
                    dropna=False
                )["Motivo"]
                .apply(
                    lambda s: " | ".join(
                        sorted(
                            set(
                                s.dropna()
                                .astype(str)
                                .str.strip()
                            )
                        )
                    )
                )
                .reset_index()
            )

            motivos_upper = (
                motivos_por_local["Motivo"]
                .fillna("")
                .astype(str)
                .str.upper()
            )

            # Sin kWh_total o kwmax válido:
            # - PDBT sin kWh_total válido
            # - GDMTH/GDMTO/GDBT sin kwmax válido
            mask_sin_kwh_o_kwmax = (
                motivos_upper.str.contains(
                    "PDBT SIN KWH_TOTAL",
                    na=False
                )
                |
                motivos_upper.str.contains(
                    "SIN KWMAX",
                    na=False
                )
            )

            # Sin demanda contratada en parser
            mask_sin_demanda_contratada = (
                motivos_upper.str.contains(
                    "SIN DEMANDA CONTRATADA",
                    na=False
                )
            )

            # Tiene ambas condiciones
            mask_sin_ambas = (
                mask_sin_kwh_o_kwmax
                & mask_sin_demanda_contratada
            )

            # Valor sospechoso repetido en todas las filas
            # Esto no necesariamente es "vacío", pero sí requiere reparseo.
            mask_valor_sospechoso = (
                motivos_upper.str.contains(
                    "VALOR SOSPECHOSO",
                    na=False
                )
            )

            locales_sin_kwh_o_kwmax = int(
                mask_sin_kwh_o_kwmax.sum()
            )

            locales_sin_demanda_contratada = int(
                mask_sin_demanda_contratada.sum()
            )

            locales_sin_ambas = int(
                mask_sin_ambas.sum()
            )

            locales_valor_sospechoso = int(
                mask_valor_sospechoso.sum()
            )

            st.warning(
                f"{locales_incompletos_unicos} locales ocupados con recibo tienen información incompleta o sospechosa de demanda. "
                f"De esos, {locales_sin_kwh_o_kwmax} no tienen kWh_total o kwmax válido según su tarifa; "
                f"{locales_sin_demanda_contratada} no tienen demanda contratada en parser; "
                f"y {locales_sin_ambas} tienen ambas condiciones. "
                f"Además, {locales_valor_sospechoso} locales tienen valores sospechosos repetidos en todos sus recibos. "
                f"La tabla muestra {len(demanda_incompleta_display)} filas, una por cada file_path real del parser con información incompleta o sospechosa. "
                f"Archivos únicos encontrados: {archivos_incompletos_unicos}."
            )

            st.dataframe(
                demanda_incompleta_display,
                use_container_width=True,
                height=600,
                hide_index=True,
                column_config={
                    "file_path": st.column_config.TextColumn(
                        "file_path",
                        width="large"
                    ),
                    "file_name": st.column_config.TextColumn(
                        "file_name",
                        width="medium"
                    )
                }
            )

            # --------------------------------------------------------
            # Tabla adicional:
            # locales NO PDBT sin kWh_total válido en ningún recibo
            # --------------------------------------------------------
            # Esta tabla parte de la tabla superior de información incompleta.
            # Agrupa por local/servicio y muestra solo locales con tarifa
            # distinta de PDBT donde ningún recibo/fila del parser tiene
            # kWh_total válido.

            st.markdown(
                '<div class="subsection-title">Locales no PDBT sin kWh_total en ningún recibo</div>',
                unsafe_allow_html=True
            )

            locales_sin_kwh_no_pdbt_rows = []

            locales_revision_kwh = (
                demanda_incompleta_display[
                    [
                        "Centro Comercial",
                        "Nombre Comercial",
                        "no_servicio",
                        "Tarifa",
                        "Demanda contratada kW",
                        "Area m2"
                    ]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            for _, row_local in locales_revision_kwh.iterrows():

                servicios_local = split_servicios_demanda_incompleta(
                    row_local.get("no_servicio", "")
                )

                servicios_local_sin_ceros = [
                    str(s).lstrip("0")
                    for s in servicios_local
                    if str(s).strip() != ""
                ]

                if not servicios_local:
                    continue

                mask_parser_local = (
                    parsed_demanda_diag["Centro Comercial"]
                    .astype(str)
                    .eq(str(row_local.get("Centro Comercial", "")))
                    & (
                        parsed_demanda_diag["_key_no_servicio_debug"]
                        .isin(servicios_local)
                        |
                        parsed_demanda_diag["_key_no_servicio_debug_sin_ceros"]
                        .isin(servicios_local_sin_ceros)
                    )
                )

                parser_local = parsed_demanda_diag[
                    mask_parser_local
                ].copy()

                if parser_local.empty:
                    continue

                parser_local = parser_local.drop_duplicates(
                    subset=["file_path"]
                )

                # Tarifa detectada en las filas reales del parser.
                if "_tarifa_diag" in parser_local.columns:
                    parser_local["_tarifa_local_norm"] = normalize_tarifa_series(
                        parser_local["_tarifa_diag"]
                    )
                else:
                    parser_local["_tarifa_local_norm"] = normalize_tarifa_value(
                        row_local.get("Tarifa", "")
                    )

                tarifas_local = (
                    parser_local["_tarifa_local_norm"]
                    .dropna()
                    .astype(str)
                    .str.upper()
                    .str.strip()
                    .replace("", pd.NA)
                    .dropna()
                    .unique()
                    .tolist()
                )

                # Excluir locales que sean únicamente PDBT.
                tarifas_no_pdbt = [
                    t for t in tarifas_local
                    if t != "PDBT"
                ]

                if not tarifas_no_pdbt:
                    continue

                parser_local["_kwh_total_valido"] = (
                    pd.to_numeric(
                        parser_local["_kwh_total_num_diag"],
                        errors="coerce"
                    )
                    .gt(0)
                )

                recibos_parser = len(parser_local)

                recibos_con_kwh = int(
                    parser_local["_kwh_total_valido"].sum()
                )

                recibos_sin_kwh = int(
                    recibos_parser - recibos_con_kwh
                )

                # Condición clave:
                # solo mostrar si ningún recibo tiene kWh_total válido.
                if recibos_con_kwh > 0:
                    continue

                file_paths_local = (
                    parser_local["file_path"]
                    .replace("", pd.NA)
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                    if "file_path" in parser_local.columns
                    else []
                )

                locales_sin_kwh_no_pdbt_rows.append({
                    "Centro Comercial": row_local.get("Centro Comercial", ""),
                    "Nombre Comercial": row_local.get("Nombre Comercial", ""),
                    "no_servicio": row_local.get("no_servicio", ""),
                    "Tarifa": (
                        " | ".join(sorted(set(tarifas_no_pdbt)))
                        if tarifas_no_pdbt
                        else row_local.get("Tarifa", "")
                    ),
                    "Recibos / filas parser": recibos_parser,
                    "Recibos con kWh_total válido": recibos_con_kwh,
                    "Recibos sin kWh_total válido": recibos_sin_kwh,
                    "Demanda contratada kW": row_local.get(
                        "Demanda contratada kW",
                        pd.NA
                    ),
                    "Area m2": row_local.get("Area m2", pd.NA),
                    "File paths únicos": len(file_paths_local),
                    "file_path ejemplo": (
                        file_paths_local[0]
                        if file_paths_local
                        else ""
                    )
                })

            locales_sin_kwh_no_pdbt_df = pd.DataFrame(
                locales_sin_kwh_no_pdbt_rows
            )

            if locales_sin_kwh_no_pdbt_df.empty:
                st.success(
                    "No hay locales no PDBT donde todos sus recibos estén sin kWh_total válido."
                )

            else:
                locales_sin_kwh_no_pdbt_df = (
                    locales_sin_kwh_no_pdbt_df
                    .sort_values(
                        [
                            "Centro Comercial",
                            "Nombre Comercial",
                            "no_servicio"
                        ],
                        na_position="last"
                    )
                    .reset_index(drop=True)
                )

                for col_num in [
                    "Recibos / filas parser",
                    "Recibos con kWh_total válido",
                    "Recibos sin kWh_total válido",
                    "Demanda contratada kW",
                    "Area m2",
                    "File paths únicos"
                ]:
                    if col_num in locales_sin_kwh_no_pdbt_df.columns:
                        locales_sin_kwh_no_pdbt_df[col_num] = pd.to_numeric(
                            locales_sin_kwh_no_pdbt_df[col_num],
                            errors="coerce"
                        )

                st.warning(
                    f"{len(locales_sin_kwh_no_pdbt_df):,} locales ocupados con recibo "
                    "con tarifa distinta de PDBT no tienen kWh_total válido en ninguno de sus recibos."
                )

                st.dataframe(
                    locales_sin_kwh_no_pdbt_df,
                    use_container_width=True,
                    hide_index=True,
                    height=500,
                    column_config={
                        "file_path ejemplo": st.column_config.TextColumn(
                            "file_path ejemplo",
                            width="large"
                        )
                    }
                )

    else:
        st.info(
            "No está disponible benchmark_densidad_base para diagnosticar demanda incompleta."
        )

    # ------------------------------------------------------------
    # Tabla provisional: locales ocupados con recibo confirmado
    # ------------------------------------------------------------

    st.markdown(
        '<div class="subsection-title">Locales ocupados con recibo confirmado por centro comercial</div>',
        unsafe_allow_html=True
    )

    if "muestra_con_recibo_global" in globals() and not muestra_con_recibo_global.empty:

        muestra_confirmada_debug = muestra_con_recibo_global.copy()

        # --------------------------------------------------------
        # Centro comercial
        # --------------------------------------------------------

        if "_centro_comercial_limpio" in muestra_confirmada_debug.columns:
            muestra_confirmada_debug["Centro Comercial"] = (
                muestra_confirmada_debug["_centro_comercial_limpio"].apply(cc_key)
            )

        elif "NOMBRE DEL CC" in muestra_confirmada_debug.columns:
            muestra_confirmada_debug["Centro Comercial"] = (
                muestra_confirmada_debug["NOMBRE DEL CC"].apply(cc_key)
            )

        elif "source_sheet" in muestra_confirmada_debug.columns:
            muestra_confirmada_debug["Centro Comercial"] = (
                muestra_confirmada_debug["source_sheet"].apply(cc_key)
            )

        else:
            muestra_confirmada_debug["Centro Comercial"] = ""

        centros_confirmados = sorted(
            muestra_confirmada_debug["Centro Comercial"]
            .dropna()
            .astype(str)
            .unique()
        )

        cc_debug_selected = st.selectbox(
            "Selecciona centro comercial para ver locales ocupados con recibo",
            options=centros_confirmados,
            key="cc_debug_locales_con_recibo"
        )

        tabla_confirmados_cc = muestra_confirmada_debug[
            muestra_confirmada_debug["Centro Comercial"]
            .astype(str)
            .eq(str(cc_debug_selected))
        ].copy()

        # --------------------------------------------------------
        # Cliente
        # --------------------------------------------------------

        if "CLIENTE" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Cliente"] = tabla_confirmados_cc["CLIENTE"]

        elif "cliente_nombre" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Cliente"] = tabla_confirmados_cc["cliente_nombre"]

        elif "parser_cliente_match" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Cliente"] = tabla_confirmados_cc["parser_cliente_match"]

        else:
            tabla_confirmados_cc["Cliente"] = ""

        # --------------------------------------------------------
        # Nombre comercial
        # --------------------------------------------------------

        if "NOMBRE COMERCIAL" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Nombre comercial"] = tabla_confirmados_cc["NOMBRE COMERCIAL"]

        elif "parser_recibos_subgroup_match" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Nombre comercial"] = tabla_confirmados_cc["parser_recibos_subgroup_match"]

        elif "recibos_subgroup" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Nombre comercial"] = tabla_confirmados_cc["recibos_subgroup"]

        else:
            tabla_confirmados_cc["Nombre comercial"] = ""

        # --------------------------------------------------------
        # No. servicio
        # --------------------------------------------------------

        if "parser_no_servicio_match" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["no_servicio"] = tabla_confirmados_cc["parser_no_servicio_match"]

        elif "no_servicio" not in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["no_servicio"] = ""

        # --------------------------------------------------------
        # Tarifa
        # --------------------------------------------------------

        if "TARIFA_FINAL" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Tarifa"] = tabla_confirmados_cc["TARIFA_FINAL"]

        elif "parser_tarifa_match" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Tarifa"] = tabla_confirmados_cc["parser_tarifa_match"]

        elif "tarifa_norm" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Tarifa"] = tabla_confirmados_cc["tarifa_norm"]

        else:
            tabla_confirmados_cc["Tarifa"] = "SIN TARIFA"

        # --------------------------------------------------------
        # Medidor
        # --------------------------------------------------------

        if "parser_medidor_match" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Medidor"] = tabla_confirmados_cc["parser_medidor_match"]

        elif "medidor" in tabla_confirmados_cc.columns:
            tabla_confirmados_cc["Medidor"] = tabla_confirmados_cc["medidor"]

        else:
            tabla_confirmados_cc["Medidor"] = ""

        # --------------------------------------------------------
        # File path real desde parsed: un archivo por fila
        # --------------------------------------------------------
        # Esta tabla NO usa parser_file_match para listar archivos,
        # porque parser_file_match ya viene resumido.
        #
        # En su lugar:
        # 1. Toma el no_servicio confirmado del local.
        # 2. Busca en parsed todas las filas de ese no_servicio.
        # 3. Muestra un renglón por cada file_path real del parser.

        def split_servicios_debug(value):
            if pd.isna(value):
                return []

            servicios = []

            for raw in str(value).split("|"):
                raw = raw.strip()

                if raw.upper() in ["", "NAN", "NONE", "<NA>"]:
                    continue

                servicio_key = normalize_service_cc(
                    pd.Series([raw])
                ).iloc[0]

                if servicio_key:
                    servicios.append(servicio_key)

            return sorted(set(servicios))

        # Llaves en parsed para buscar los archivos reales
        parsed_debug_files = parsed.copy()

        if mall_col and mall_col in parsed_debug_files.columns:
            parsed_debug_files["Centro Comercial"] = (
                parsed_debug_files[mall_col].apply(cc_key)
            )
        else:
            parsed_debug_files["Centro Comercial"] = ""

        parsed_debug_files["_key_no_servicio_debug"] = (
            normalize_service_cc(parsed_debug_files["no_servicio"])
            if "no_servicio" in parsed_debug_files.columns
            else ""
        )

        # Versión sin ceros iniciales.
        # Esto permite que:
        # 056201051440 del parser
        # haga match con:
        # 56201051440 de la muestra confirmada.
        parsed_debug_files["_key_no_servicio_debug_sin_ceros"] = (
            parsed_debug_files["_key_no_servicio_debug"]
            .fillna("")
            .astype(str)
            .str.lstrip("0")
        )

        if "file_path" not in parsed_debug_files.columns:
            parsed_debug_files["file_path"] = ""

        if "file_name" not in parsed_debug_files.columns:
            parsed_debug_files["file_name"] = ""

        filas_archivos_confirmados = []

        for _, row_local in tabla_confirmados_cc.iterrows():

            servicios_local = split_servicios_debug(
                row_local.get("no_servicio", "")
            )

            if not servicios_local:
                servicios_local = split_servicios_debug(
                    row_local.get("parser_no_servicio_match", "")
                )

            # Si no hay no_servicio, dejamos una fila del local sin file_path.
            if not servicios_local:
                filas_archivos_confirmados.append({
                    "Centro Comercial": row_local.get("Centro Comercial", ""),
                    "Cliente": row_local.get("Cliente", ""),
                    "Nombre comercial": row_local.get("Nombre comercial", ""),
                    "no_servicio": row_local.get("no_servicio", ""),
                    "Tarifa": row_local.get("Tarifa", ""),
                    "Medidor": row_local.get("Medidor", ""),
                    "file_path": "",
                    "file_name": "",
                    "parser_criterio_match": row_local.get("parser_criterio_match", "")
                })
                continue

            servicios_local_sin_ceros = [
                str(s).lstrip("0")
                for s in servicios_local
                if str(s).strip() != ""
            ]

            mask_parser_local = (
                parsed_debug_files["Centro Comercial"]
                .astype(str)
                .eq(str(cc_debug_selected))
                & (
                    parsed_debug_files["_key_no_servicio_debug"]
                    .isin(servicios_local)
                    |
                    parsed_debug_files["_key_no_servicio_debug_sin_ceros"]
                    .isin(servicios_local_sin_ceros)
                )
            )

            parser_files_local = parsed_debug_files[
                mask_parser_local
            ].copy()

            if parser_files_local.empty:
                filas_archivos_confirmados.append({
                    "Centro Comercial": row_local.get("Centro Comercial", ""),
                    "Cliente": row_local.get("Cliente", ""),
                    "Nombre comercial": row_local.get("Nombre comercial", ""),
                    "no_servicio": row_local.get("no_servicio", ""),
                    "Tarifa": row_local.get("Tarifa", ""),
                    "Medidor": row_local.get("Medidor", ""),
                    "file_path": "",
                    "file_name": "",
                    "parser_criterio_match": row_local.get("parser_criterio_match", "")
                })
                continue

            # Un renglón por cada archivo real del parser.
            # Drop duplicates evita repetir el mismo PDF si el parser lo tiene duplicado.
            parser_files_local = (
                parser_files_local
                .drop_duplicates(subset=["file_path"])
                .sort_values("file_path")
            )

            for _, row_file in parser_files_local.iterrows():

                filas_archivos_confirmados.append({
                    "Centro Comercial": row_local.get("Centro Comercial", ""),
                    "Cliente": row_local.get("Cliente", ""),
                    "Nombre comercial": row_local.get("Nombre comercial", ""),
                    "no_servicio": row_local.get("no_servicio", ""),
                    "Tarifa": row_local.get("Tarifa", ""),
                    "Medidor": row_local.get("Medidor", ""),
                    "file_path": row_file.get("file_path", ""),
                    "file_name": row_file.get("file_name", ""),
                    "parser_criterio_match": row_local.get("parser_criterio_match", "")
                })

        tabla_confirmados_display = pd.DataFrame(
            filas_archivos_confirmados
        )

        if not tabla_confirmados_display.empty:
            tabla_confirmados_display = (
                tabla_confirmados_display
                .sort_values(
                    ["Nombre comercial", "Cliente", "file_path"],
                    na_position="last"
                )
                .reset_index(drop=True)
            )

        locales_unicos_confirmados = (
            tabla_confirmados_cc[
                ["Cliente", "Nombre comercial", "no_servicio"]
            ]
            .drop_duplicates()
            .shape[0]
        )

        archivos_unicos_confirmados = (
            tabla_confirmados_display["file_path"]
            .replace("", pd.NA)
            .dropna()
            .nunique()
            if not tabla_confirmados_display.empty
            and "file_path" in tabla_confirmados_display.columns
            else 0
        )

        st.caption(
            f"{locales_unicos_confirmados} locales ocupados con recibo confirmado para {cc_debug_selected}. "
            f"La tabla muestra {len(tabla_confirmados_display)} filas, una por cada file_path real del parser. "
            f"Archivos únicos encontrados: {archivos_unicos_confirmados}."
        )

        st.dataframe(
            tabla_confirmados_display,
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "file_path": st.column_config.TextColumn(
                    "file_path",
                    width="large"
                )
            }
        )

    else:
        st.info(
            "No está disponible muestra_con_recibo_global para mostrar locales ocupados con recibo confirmado."
        )

    # ------------------------------------------------------------
    # Locales ocupados sin recibo confirmado
    # ------------------------------------------------------------

    st.markdown(
        '<div class="subsection-title">Locales ocupados sin recibo confirmado</div>',
        unsafe_allow_html=True
    )

    if "sin_match_df" in globals() and not sin_match_df.empty:
        st.dataframe(
            sin_match_df.sort_values(
                ["Centro Comercial", "Nombre comercial", "Cliente"],
                na_position="last"
            ).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=500
        )
    else:
        st.success("No hay locales ocupados sin recibo confirmado.")

    # ------------------------------------------------------------
    # Recibos encontrados en parser sin match en Datos Generales
    # ------------------------------------------------------------

    st.markdown(
        '<div class="subsection-title">Recibos encontrados en parser sin match en Datos Generales</div>',
        unsafe_allow_html=True
    )

    if "parser_sin_match_df" in globals() and not parser_sin_match_df.empty:
        st.dataframe(
            parser_sin_match_df.sort_values(
                ["Centro Comercial", "Cliente parser", "Subgrupo parser"],
                na_position="last"
            ).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=500
        )

        st.caption(
            "Esta tabla muestra recibos o servicios que sí están en el parser, "
            "pero que no lograron asociarse a un local ocupado en Datos Generales usando el match global confirmado."
        )
    else:
        st.success("Todos los recibos del parser hicieron match contra Datos Generales.")

    # ------------------------------------------------------------
    # Cruce de faltantes: DG sin recibo vs parser sin match
    # ------------------------------------------------------------

    st.markdown(
        '<div class="subsection-title">Candidatos para recuperar match en centros comerciales con faltantes</div>',
        unsafe_allow_html=True
    )

    if (
        "sin_match_df" in globals()
        and not sin_match_df.empty
        and "parser_sin_match_df" in globals()
        and not parser_sin_match_df.empty
    ):

        def key_debug_match(value):
            if pd.isna(value):
                return ""

            value = normalizar_texto_simple(value)
            value = re.sub(r"[^A-Z0-9\s]", " ", value)
            value = re.sub(r"\s+", " ", value).strip()

            for term in [
                "SA DE CV",
                "S A DE C V",
                "SAPI DE CV",
                "S DE RL DE CV",
                "DE CV",
                "SA",
                "CV",
                "SAPI",
                "S DE RL",
                "GRUPO",
                "OPERADORA",
                "COMERCIAL",
                "COMERCIALIZADORA",
                "SERVICIOS",
                "TIENDAS"
            ]:
                value = value.replace(term, " ")

            value = re.sub(r"\s+", " ", value).strip()

            return value

        def compact_debug_match(value):
            return re.sub(
                r"[^A-Z0-9]",
                "",
                key_debug_match(value)
            )

        def palabras_fuertes(value):
            stop_words = {
                "DE", "DEL", "LA", "EL", "LOS", "LAS", "Y", "EN",
                "SA", "CV", "SAPI", "GRUPO", "MEXICO", "COMERCIAL",
                "OPERADORA", "SERVICIOS", "TIENDAS", "SUCURSAL"
            }

            return {
                w for w in key_debug_match(value).split()
                if len(w) >= 4 and w not in stop_words
            }

        def score_candidato_match(row_dg, row_parser):
            score = 0
            motivos = []

            dg_nombre = row_dg.get("Nombre comercial", "")
            dg_cliente = row_dg.get("Cliente", "")
            dg_local = row_dg.get("No de local", "")

            parser_nombre = row_parser.get("Subgrupo parser", "")
            parser_cliente = row_parser.get("Cliente parser", "")
            parser_file = row_parser.get("file_path", "")
            parser_servicio = row_parser.get("no_servicio", "")
            parser_medidor = row_parser.get("medidor", "")

            dg_nombre_key = key_debug_match(dg_nombre)
            parser_nombre_key = key_debug_match(parser_nombre)

            dg_nombre_compact = compact_debug_match(dg_nombre)
            parser_text_compact = compact_debug_match(
                str(parser_nombre)
                + " "
                + str(parser_cliente)
                + " "
                + str(parser_file)
            )

            # Nombre comercial exacto
            if dg_nombre_key and parser_nombre_key and dg_nombre_key == parser_nombre_key:
                score += 100
                motivos.append("nombre comercial exacto")

            # Nombre comercial contenido en file_path/subgrupo
            elif dg_nombre_compact and len(dg_nombre_compact) >= 5 and dg_nombre_compact in parser_text_compact:
                score += 80
                motivos.append("nombre comercial aparece en parser/file_path")

            # Palabras fuertes compartidas
            palabras_dg = palabras_fuertes(dg_nombre)
            palabras_parser = palabras_fuertes(
                str(parser_nombre) + " " + str(parser_file)
            )

            palabras_comunes = palabras_dg & palabras_parser

            if len(palabras_comunes) >= 2:
                score += 50
                motivos.append(
                    "palabras comunes: " + ", ".join(sorted(palabras_comunes))
                )

            # Cliente exacto, con bajo peso porque puede agrupar varias marcas
            if key_debug_match(dg_cliente) and key_debug_match(dg_cliente) == key_debug_match(parser_cliente):
                score += 15
                motivos.append("cliente exacto")

            # No. local aparece en file_path
            if pd.notna(dg_local) and str(dg_local).strip() != "":
                variantes_local = generar_variantes_no_local(dg_local)

                parser_file_key = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    normalizar_texto_simple(parser_file)
                )

                if variantes_local and any(v in parser_file_key for v in variantes_local):
                    score += 120
                    motivos.append("no. local aparece en file_path")

            return score, " | ".join(motivos)

        candidatos_rows = []

        cc_con_faltantes = set(
            sin_match_df["Centro Comercial"]
            .dropna()
            .astype(str)
            .unique()
        )

        for cc_debug in sorted(cc_con_faltantes):

            dg_cc = sin_match_df[
                sin_match_df["Centro Comercial"].astype(str).eq(cc_debug)
            ].copy()

            parser_cc_debug = parser_sin_match_df[
                parser_sin_match_df["Centro Comercial"].astype(str).eq(cc_debug)
            ].copy()

            if dg_cc.empty or parser_cc_debug.empty:
                continue

            for _, row_dg in dg_cc.iterrows():

                candidatos_local = []

                for _, row_parser in parser_cc_debug.iterrows():

                    score, motivo = score_candidato_match(
                        row_dg,
                        row_parser
                    )

                    if score <= 0:
                        continue

                    candidatos_local.append({
                        "Centro Comercial": cc_debug,
                        "DG Nombre comercial": row_dg.get("Nombre comercial", ""),
                        "DG Cliente": row_dg.get("Cliente", ""),
                        "DG No de local": row_dg.get("No de local", ""),
                        "Parser Subgrupo": row_parser.get("Subgrupo parser", ""),
                        "Parser Cliente": row_parser.get("Cliente parser", ""),
                        "Parser no_servicio": row_parser.get("no_servicio", ""),
                        "Parser medidor": row_parser.get("medidor", ""),
                        "Parser tarifa": row_parser.get("tarifa", ""),
                        "Parser file_path": row_parser.get("file_path", ""),
                        "Score": score,
                        "Motivo": motivo
                    })

                candidatos_local = sorted(
                    candidatos_local,
                    key=lambda x: x["Score"],
                    reverse=True
                )

                # Mostramos máximo 3 candidatos por local faltante
                candidatos_rows.extend(candidatos_local[:3])

        candidatos_match_df = pd.DataFrame(candidatos_rows)

        if not candidatos_match_df.empty:

            st.warning(
                f"Se encontraron {len(candidatos_match_df)} candidatos posibles. "
                "Revisa primero los de mayor score; esto todavía NO cambia la muestra."
            )

            st.dataframe(
                candidatos_match_df.sort_values(
                    ["Centro Comercial", "Score"],
                    ascending=[True, False]
                ),
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    "Parser file_path": st.column_config.TextColumn(
                        "Parser file_path",
                        width="large"
                    )
                }
            )

        else:
            st.info(
                "No se encontraron candidatos claros entre locales sin recibo y parser sin match."
            )

    else:
        st.info(
            "No hay suficientes datos para comparar locales sin recibo contra parser sin match."
        )

    st.markdown(
        '<div class="subsection-title">Diagnóstico de diferencia parser vs Datos Generales</div>',
        unsafe_allow_html=True
    )

    servicios_parser = (
        parsed["no_servicio"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    medidores_parser = (
        parsed["medidor"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    st.write("Servicios únicos en parser:", servicios_parser.nunique())
    st.write("Medidores únicos en parser:", medidores_parser.nunique())
    st.write("Filas totales en parser:", len(parsed))

    total_ocupados_datos_generales = coverage_by_mall["Locales ocupados"].iloc[:-1].sum()
    total_con_match = coverage_by_mall["Locales ocupados con recibo"].iloc[:-1].sum()
    total_sin_match = coverage_by_mall["Locales ocupados sin recibo"].iloc[:-1].sum()

    st.write("Locales ocupados en Datos Generales:", total_ocupados_datos_generales)
    st.write("Locales ocupados con match a parser:", total_con_match)
    st.write("Locales ocupados sin match a parser:", total_sin_match)

    st.markdown(
        '<div class="section-title">Comprobación de muestra</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="Recibos sin número de servicio pero con medidor</div>',
        unsafe_allow_html=True
    )

    missing_service_with_meter = parsed[
        parsed["no_servicio"].isna()
        & parsed["medidor"].notna()
    ]

    st.write(
        "Recibos sin número de servicio pero con medidor:",
        len(missing_service_with_meter)
    )

    if not missing_service_with_meter.empty:
        st.dataframe(
            missing_service_with_meter[
                ["mall_folder", "cliente_nombre", "no_servicio", "medidor", "cuenta", "rmu"]
            ].head(50),
            use_container_width=True
        )


    st.markdown(
        '<div class="subsection-title">Servicios con más de un medidor</div>',
        unsafe_allow_html=True
    )

    servicios_con_varios_medidores = (
        parsed
        .dropna(subset=["no_servicio", "medidor"])
        .groupby("no_servicio")
        .agg(
            cliente=("cliente_nombre", lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))),
            medidores=("medidor", lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))),
            cantidad_medidores=("medidor", lambda x: x.dropna().astype(str).nunique()),
            centros_comerciales=(mall_col, lambda x: " | ".join(sorted(x.dropna().astype(str).unique()))),
        )
        .reset_index()
    )

    servicios_con_varios_medidores = servicios_con_varios_medidores[
        servicios_con_varios_medidores["cantidad_medidores"] > 1
    ].sort_values("cantidad_medidores", ascending=False)

    st.write(
        "Servicios con más de un medidor:",
        len(servicios_con_varios_medidores)
    )

    st.dataframe(
        servicios_con_varios_medidores,
        use_container_width=True
    )

    st.markdown(
        '<div class="subsection-title">Resumen por centro comercial</div>',
        unsafe_allow_html=True
    )

    if mall_col:
        mall_quality = (
            parsed
            .groupby(mall_col)
            .agg(
                servicios=("no_servicio", "nunique"),
                rmu=("rmu", "nunique"),
                medidores=("medidor", "nunique"),
            )
            .reset_index()
            .rename(columns={mall_col: "Centro Comercial"})
            .sort_values("Centro Comercial", ascending=True)
        )

        st.dataframe(
            mall_quality,
            use_container_width=True
        )

    st.markdown(
        '<div class="subsection-title">Cobertura de campos del parser</div>',
        unsafe_allow_html=True
    )

    if not quality_df.empty:
        st.dataframe(
            quality_df,
            use_container_width=True
        )

        chart_df = quality_df.set_index("campo")[["cobertura_%"]]
        st.bar_chart(chart_df)
    else:
        st.info("No hay columnas suficientes para calcular cobertura de campos.")



with tab_anexo:

    st.markdown(
        '<div class="section-title">Anexo metodológico</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subsection-title">Metodología de estimación de Demanda promedio anual (kW)</div>',
        unsafe_allow_html=True
    )

    metodologia_df = pd.DataFrame({
        "Tarifa": ["GDMTH", "GDMTO", "GDBT", "PDBT"],
        "Criterio de Demanda promedio anual (kW)": [
            "Promedio de kwmax de los recibos más recientes disponibles",
            "Promedio de kwmax de los recibos más recientes disponibles",
            "Promedio de kwmax de los recibos más recientes disponibles",
            "Promedio de demanda estimada con perfil NREL calibrado con factor Allux por giro"
        ],
        "Ventana ideal": [
            "Hasta 6 recibos bimestrales o 12 mensuales",
            "Hasta 6 recibos bimestrales o 12 mensuales",
            "Hasta 6 recibos bimestrales o 12 mensuales",
            "Hasta 6 recibos bimestrales o 12 mensuales"
        ],
        "Tipo de dato": [
            "Medido en recibo",
            "Medido en recibo",
            "Medido en recibo",
            "Estimado híbrido"
        ]
    })

    st.dataframe(metodologia_df, use_container_width=True)
    st.caption(NOTA_DEMANDA_PROMEDIO_ANUAL)

    st.markdown(
        '<div class="subsection-title">Calibración híbrida PDBT: NREL + factor Allux</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Para usuarios PDBT, los recibos no reportan demanda máxima medida. 
        Por ello, la app estima primero una demanda base con perfiles horarios NREL:

        **Demanda NREL PDBT = kWh promedio diario × peso máximo horario del perfil NREL**

        Los perfiles NREL incorporan una forma horaria diferenciada por zona climática y tipo de local. 
        Sin embargo, al representar una operación típica teórica, pueden no capturar completamente la 
        operación real observada en los centros comerciales de Allux.

        Para calibrar la estimación, se usan los locales del mismo portafolio que sí tienen demanda medida 
        en recibo (**GDMTH, GDMTO y GDBT**). Para esos locales se calcula:

        **Ratio Allux medido = demanda promedio anual medida / kWh promedio diario**

        Para los PDBT estimados con NREL se calcula:

        **Ratio NREL PDBT = demanda PDBT estimada con NREL / kWh promedio diario**

        El factor de ajuste compara ambos ratios:

        **Factor Allux = mediana(Ratio Allux medido) / mediana(Ratio NREL PDBT)**

        El modelo aplica el factor al nivel más específico con muestra suficiente:

        1. **Giro comercial + clima**: se usa si existen al menos 5 locales medidos y 3 locales PDBT 
           del mismo giro y clima.
        2. **Giro comercial**: si no alcanza la muestra por giro + clima, se usa el factor del giro 
           considerando todos los climas, siempre que existan al menos 5 locales medidos y 3 locales PDBT.
        3. **Clima**: si no alcanza la muestra por giro, se usa el factor climático considerando todos 
           los giros, siempre que existan al menos 10 locales medidos y 5 locales PDBT.
        4. **General portafolio**: si no hay muestra suficiente en los niveles anteriores, se usa el factor 
           general del portafolio.

        Para reducir el efecto de outliers, los factores se calculan con la mediana de los ratios y se acotan 
        dentro del rango definido en el modelo.
        """
    )

    st.markdown(
        '<div class="subsection-title">Factores de ajuste Allux aplicados a PDBT</div>',
        unsafe_allow_html=True
    )

    if (
        "factores_ajuste_allux_pdbt_df" in globals()
        and isinstance(factores_ajuste_allux_pdbt_df, pd.DataFrame)
        and not factores_ajuste_allux_pdbt_df.empty
    ):
        st.dataframe(
            factores_ajuste_allux_pdbt_df,
            use_container_width=True,
            hide_index=True,
            height=500
        )

        st.caption(
            "La tabla muestra los factores evaluados por nivel metodológico. "
            "La columna 'Muestra suficiente' indica si ese nivel cumple los mínimos definidos. "
            "El factor aplicado a cada PDBT sigue la jerarquía: giro + clima, giro, clima y general portafolio."
        )

    else:
        st.info(
            "No hay suficientes datos para construir la tabla de factores de ajuste Allux PDBT."
        )

    st.markdown(
        '<div class="subsection-title">Asignación de perfiles por giro comercial</div>',
        unsafe_allow_html=True
    )

    perfil_mapping_df = pd.DataFrame({
        "Giro comercial": [
            "Alimentos y Bebidas",
            "Alimentos y Bebidas",
            "Tiendas Departamentales",
            "Moda",
            "Tecnología / Electrónicos",
            "Wellness",
            "Servicios Generales",
            "Bienes de Consumo",
            "Entretenimiento",
            "Otros"
        ],
        "Tipo de local": [
            "Food Court",
            "Cualquier otro",
            "Cualquiera",
            "Cualquiera",
            "Cualquiera",
            "Cualquiera",
            "Cualquiera",
            "Cualquiera",
            "Cualquiera",
            "Cualquiera"
        ],
        "Perfil NREL": [
            "Quick Service Restaurant",
            "Full Service Restaurant",
            "Retail Standalone",
            "Retail Strip Mall",
            "Retail Strip Mall",
            "Retail Strip Mall",
            "Retail Strip Mall",
            "Retail Strip Mall",
            "Retail Strip Mall",
            "Retail Strip Mall"
        ]
    })

    st.dataframe(
        perfil_mapping_df,
        use_container_width=True
    )

    st.markdown(
        '<div class="section-title">Uso mensual de perfiles NREL</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    La estimación de demanda para usuarios PDBT utilizará perfiles horarios
    de NREL/DOE diferenciados por:

    - Zona climática
    - Giro
    """)

    st.markdown(
        '<div class="subsection-title">Perfil horario NREL utilizado para estimación PDBT</div>',
        unsafe_allow_html=True
    )

    zona_nrel = st.selectbox(
        "Zona climática",
        [
            "Cálido húmedo (Hot-Humid)",
            "Templado seco (Mixed-Dry)",
            "Templado húmedo (Mixed-Humid)",
            "Cálido seco (Hot-Dry)",
            "Frío (Cold)"
        ],
        key="zona_nrel_anexo"
    )

    tipo_perfil = st.selectbox(
        "Tipo de perfil",
        [
            "Local comercial (Retail Strip Mall)",
            "Tienda departamental (Retail Standalone)",
            "Restaurante de comida rápida (Quick Service Restaurant)",
            "Restaurante de servicio completo (Full Service Restaurant)"
        ],
        key="tipo_perfil_anexo"
    )

    zona_file_map = {
        "Cálido húmedo (Hot-Humid)": "hot-humid",
        "Templado seco (Mixed-Dry)": "mixed-dry",
        "Templado húmedo (Mixed-Humid)": "mixed-humid",
        "Cálido seco (Hot-Dry)": "hot-dry",
        "Frío (Cold)": "cold"
    }

    perfil_file_map = {
        "Local comercial (Retail Strip Mall)": "retailstripmall",
        "Tienda departamental (Retail Standalone)": "retailstandalone",
        "Restaurante de comida rápida (Quick Service Restaurant)": "quickservicerestaurant",
        "Restaurante de servicio completo (Full Service Restaurant)": "fullservicerestaurant"
    }

    perfil_filename = (
        f"up0-"
        f"{zona_file_map[zona_nrel]}-"
        f"{perfil_file_map[tipo_perfil]}.csv"
    )

    perfil_path = (
        DATA_DIR
        / "profiles"
        / perfil_filename
    )

    st.caption(f"Archivo seleccionado: `{perfil_filename}`")


    if perfil_path.exists():

        perfil_df = construir_perfil_mensual_nrel(
            perfil_path
        )

        meses_map = {
            "Enero": 1,
            "Febrero": 2,
            "Marzo": 3,
            "Abril": 4,
            "Mayo": 5,
            "Junio": 6,
            "Julio": 7,
            "Agosto": 8,
            "Septiembre": 9,
            "Octubre": 10,
            "Noviembre": 11,
            "Diciembre": 12
        }

        selected_mes_label = st.selectbox(
            "Selecciona mes",
            options=list(meses_map.keys()),
            key="mes_perfil_nrel"
        )

        selected_mes = meses_map[selected_mes_label]

        perfil_mes = perfil_df[
            perfil_df["mes"] == selected_mes
        ].copy()

        st.dataframe(
            perfil_mes,
            use_container_width=True
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        ax.plot(
            perfil_mes["hora"],
            perfil_mes["peso_normalizado"]
        )

        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Peso normalizado")

        ax.set_title(
            f"Perfil NREL - {selected_mes_label}"
        )

        st.pyplot(fig)

    else:

        st.warning(
            f"No encontré el archivo: {perfil_path}"
        )
  
    st.write(
        "Carga máxima relativa:",
        perfil_mes["peso_normalizado"].max()
    )

    st.info(
        "Este valor representa la hora pico de un día promedio del mes. "
        "Para estimar demanda PDBT se usará: "
        "(kWh mensual / días del mes) × carga máxima relativa."
    )

    st.markdown(
        '<div class="subsection-title">Mapeo de centros comerciales a zona climática NREL</div>',
        unsafe_allow_html=True
    )

    climate_mapping_path = DATA_DIR / "profiles" / "cc_master_data.csv"

    if climate_mapping_path.exists():
        climate_mapping_df = pd.read_csv(climate_mapping_path)

        st.dataframe(
            climate_mapping_df,
            use_container_width=True
        )
    else:
        st.warning(
            f"No encontré el archivo de mapeo climático: {climate_mapping_path}"
        )

    st.markdown(
        '<div class="subsection-title">Archivos de perfiles NREL disponibles</div>',
        unsafe_allow_html=True
    )
    profiles_dir = DATA_DIR / "profiles"

    expected_profiles = []

    zonas_file = {
        "hot-humid": "Cálido húmedo (Hot-Humid)",
        "mixed-dry": "Templado seco (Mixed-Dry)",
        "mixed-humid": "Templado húmedo (Mixed-Humid)",
        "hot-dry": "Cálido seco (Hot-Dry)",
        "cold": "Frío (Cold)"
    }

    tipos_file = {
        "retailstripmall": "Local comercial (Retail Strip Mall)",
        "retailstandalone": "Tienda departamental (Retail Standalone)",
        "quickservicerestaurant": "Comida rápida (Quick Service Restaurant)",
        "fullservicerestaurant": "Restaurante (Full Service Restaurant)"
    }

    for zona_code, zona_nombre in zonas_file.items():

        for tipo_code, tipo_nombre in tipos_file.items():

            filename = f"up0-{zona_code}-{tipo_code}.csv"

            expected_profiles.append({
                "Zona climática": zona_nombre,
                "Tipo de perfil": tipo_nombre,
                "Disponible": "Sí" if (profiles_dir / filename).exists() else "No"
            })

    expected_profiles_df = pd.DataFrame(expected_profiles)

    expected_profiles_df = (
        expected_profiles_df
        .sort_values(
            ["Zona climática", "Tipo de perfil"]
        )
        .reset_index(drop=True)
    )

    st.dataframe(
        expected_profiles_df,
        use_container_width=True
    )

    st.markdown(
        '<div class="subsection-title">Vista preliminar de asignación de perfiles por local</div>',
        unsafe_allow_html=True
    )

    climate_mapping_path = DATA_DIR / "profiles" / "cc_climate_zone_mapping.csv"

    if climate_mapping_path.exists() and "muestra_con_recibo" in locals():

        climate_mapping_df = pd.read_csv(
            climate_mapping_path,
            encoding="latin-1"
        )

        demanda_template_df = crear_demanda_real_anual_template(
            muestra_con_recibo,
            climate_mapping_df
        )

        st.dataframe(
            demanda_template_df[
                [
                    "centro_comercial",
                    "nombre_comercial",
                    "subgiro_comercial",
                    "tipo_local",
                    "tarifa",
                    "zona_nrel",
                    "perfil_nrel_nombre",
                    "demanda_real_anual_kw",
                    "criterio_demanda"
                ]
            ],
            use_container_width=True
        )

    else:
        st.info(
            "La vista preliminar de asignación de perfiles se mostrará cuando exista muestra con recibo y el archivo de zonas climáticas."
        )

    st.markdown(
        '<div class="subsection-title">Verificación de archivos NREL requeridos</div>',
        unsafe_allow_html=True
    )
    profiles_dir = DATA_DIR / "profiles"

    zonas_requeridas = {
        "hot-humid": "Cálido húmedo (Hot-Humid)",
        "mixed-dry": "Templado seco (Mixed-Dry)",
        "mixed-humid": "Templado húmedo (Mixed-Humid)",
        "hot-dry": "Cálido seco (Hot-Dry)",
        "cold": "Frío (Cold)"
    }

    tipos_requeridos = {
        "retailstripmall": "Local comercial (Retail Strip Mall)",
        "retailstandalone": "Tienda departamental (Retail Standalone)",
        "quickservicerestaurant": "Comida rápida (Quick Service Restaurant)",
        "fullservicerestaurant": "Restaurante (Full Service Restaurant)"
    }

    verificacion_rows = []

    for zona_code, zona_nombre in zonas_requeridas.items():
        for tipo_code, tipo_nombre in tipos_requeridos.items():

            filename = f"up0-{zona_code}-{tipo_code}.csv"
            file_path = profiles_dir / filename

            verificacion_rows.append({
                "Zona climática": zona_nombre,
                "Tipo de perfil": tipo_nombre,
                "Archivo": filename,
                "Disponible": "Sí" if file_path.exists() else "No"
            })

    verificacion_df = pd.DataFrame(verificacion_rows)

    st.dataframe(
        verificacion_df,
        use_container_width=True
    )

    faltantes_df = verificacion_df[
        verificacion_df["Disponible"] == "No"
    ]

    if faltantes_df.empty:
        st.success("Todos los archivos NREL requeridos están disponibles.")
    else:
        st.warning("Faltan archivos NREL requeridos.")
        st.dataframe(
            faltantes_df,
            use_container_width=True
        )

    st.markdown(
        '<div class="subsection-title">Referencias</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
**Fuente base del perfil:**  
NREL / DOE — End-Use Load Profiles for the U.S. Building Stock.  
Dataset público de perfiles de carga a resolución de 15 minutos para edificios comerciales y residenciales, desarrollado con modelos ResStock/ComStock calibrados con datos medidos.

**Referencia:**  
https://data.openei.org/submissions/4520

**Nota metodológica:**  
El archivo `perfil_pdbt_retail_nrel.csv` se usa como perfil horario comercial normalizado para estimar demanda máxima en usuarios PDBT. En esta versión puede contener un perfil temporal; cuando se sustituya por un perfil descargado de NREL/DOE, el código no necesita cambiar.
""")

