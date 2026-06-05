from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import re


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"

DEFAULT_PARSED_CSV = OUTPUT_DIR / "bills_parsed_v2_RUN_9933.csv"
DEFAULT_HISTORICO_CSV = OUTPUT_DIR / "bills_historico_v2_RUN_9933.csv"
DEFAULT_GENERAL_DATA = DATA_DIR / "260414_Datos Generales_19 CC.xlsx"


st.set_page_config(
    page_title="Allux Live Energy Report",
    layout="wide"
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1B5E20;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #2E7D32;
        margin-top: 1.4rem;
        margin-bottom: 0.6rem;
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


def clean_date_series(s: pd.Series) -> pd.Series:
    """
    Converts a date-like series into datetime.
    """
    if s is None:
        return pd.Series(dtype="datetime64[ns]")

    return pd.to_datetime(s, errors="coerce", dayfirst=False)


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

def construir_perfil_mensual_nrel(profile_path):

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

# ============================================================
# Header
# ============================================================

st.markdown('<div class="main-title">Allux Live Energy Report</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Reporte vivo de consumo eléctrico, facturación, histórico y datos generales por centro comercial y locatario.</div>',
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

parsed_raw = read_csv_safe(parsed_path)
historico_raw = read_csv_safe(historico_path)
general_raw = read_general_excel_all_sheets(general_data_path)

parsed = prepare_parsed_data(parsed_raw)
st.write("Filas CSV originales:", len(parsed_raw))
st.write("Filas después de preparación:", len(parsed))

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


tab_resumen, tab_calidad, tab_general, tab_cc, tab_sg, tab_anexo = st.tabs([
    "Resumen Ejecutivo",
    "Calidad de Datos",
    "Portafolio",
    "Centro Comercial",
    "Servicios Generales",
    "Anexo"
])

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

with tab_resumen:
    # ============================================================
    # Executive summary
    # ============================================================

    st.markdown('<div class="section-title">1. Resumen ejecutivo</div>', unsafe_allow_html=True)

    total_bills = len(parsed)

    unique_services = parsed[service_col].nunique() if service_col else None

    st.write("Columna usada para servicios únicos:", service_col)

    if service_col:
        st.write("Servicios únicos sin vacíos:", parsed[service_col].nunique())
        st.write("Servicios únicos incluyendo vacíos:", parsed[service_col].fillna("VACÍO").nunique())


    unique_tenants = parsed[tenant_col].nunique() if tenant_col else None

    total_kwh = filtered["kwh_total_num"].sum(skipna=True) if "kwh_total_num" in filtered.columns else pd.NA
    total_amount = filtered["importe_total_num"].sum(skipna=True) if "importe_total_num" in filtered.columns else pd.NA

    avg_mxn_kwh = pd.NA
    if "mxn_per_kwh" in filtered.columns:
        avg_mxn_kwh = filtered["mxn_per_kwh"].dropna().mean()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Recibos", f"{total_bills:,}")
    col2.metric("Servicios únicos", "—" if unique_services is None else f"{unique_services:,}")
    col3.metric("Locatarios/subgrupos", "—" if unique_tenants is None else f"{unique_tenants:,}")
    col4.metric("kWh total", format_number(total_kwh, 0))
    col5.metric("Importe total", format_money_compact(total_amount))

    col6, col7, col8 = st.columns(3)
    col6.metric("Costo promedio MXN/kWh", format_mxn_per_kwh(avg_mxn_kwh))
    col7.metric("Archivo recibos", parsed_path.name)
    st.write(parsed_path)
    col8.metric("Archivo histórico", historico_path.name if historico_path.exists() else "No disponible")
    
    st.markdown(
        """
        <div class="note-box">
        Este resumen se calcula a partir de los recibos extraídos por el parser local.
        Los filtros laterales modifican todas las métricas y gráficas de esta página.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ============================================================
    # Benchmark de densidad de demanda
    # ============================================================

    st.markdown("### Benchmark de densidad de demanda")

    st.caption(
        "Esta sección mostrará la densidad de demanda por giro comercial "
        "filtrando por clima y tipo de centro comercial. "
        "Se activará cuando exista la columna de densidad calculada."
    )

    cc_master_path = DATA_DIR / "profiles" / "cc_master_data.csv"

    if cc_master_path.exists():

        cc_master_df = pd.read_csv(cc_master_path)
        cc_master_df.columns = cc_master_df.columns.str.strip()

        clima_selector = st.selectbox(
            "Selecciona clima",
            ["Cálido", "Templado", "Frío"],
            key="benchmark_clima"
        )

        tipo_cc_selector = st.selectbox(
            "Selecciona tipo de centro comercial",
            [
                "Luxury Fashion Mall",
                "Fashion Mall",
                "Regional Mall",
                "Power Center",
                "Strip Mall"
            ],
            key="benchmark_tipo_cc"
        )

        # Placeholder hasta que exista la densidad real
        densidad_col = first_existing_column(
            locals().get("demanda_real_anual_df", pd.DataFrame()),
            [
                "densidad_demanda_w_m2",
                "Densidad de demanda W/m2",
                "densidad_demanda_kw_m2",
                "Densidad de demanda kW/m2"
            ]
        )

        if "demanda_real_anual_df" in locals() and densidad_col:

            benchmark_df = demanda_real_anual_df.copy()

            benchmark_filtrado = benchmark_df[
                (benchmark_df["macro_clima"] == clima_selector)
                & (benchmark_df["tipo_cc"] == tipo_cc_selector)
            ].copy()

            if not benchmark_filtrado.empty:

                resumen_benchmark = (
                    benchmark_filtrado
                    .groupby("subgiro_comercial")
                    .agg(
                        densidad_promedio=(densidad_col, "mean"),
                        densidad_std=(densidad_col, "std"),
                        no_locales=("subgiro_comercial", "size")
                    )
                    .reset_index()
                    .rename(columns={
                        "subgiro_comercial": "Giro comercial",
                        "densidad_promedio": "Densidad de demanda promedio",
                        "no_locales": "No. de locales en la muestra"
                    })
                )

                resumen_benchmark["CV (%)"] = (
                    resumen_benchmark["densidad_std"]
                    / resumen_benchmark["Densidad de demanda promedio"]
                    * 100
                ).round(1)

                def clasificar_confiabilidad(row):
                    n = row["No. de locales en la muestra"]
                    cv = row["CV (%)"]

                    if n < 10:
                        return "🔴 Baja"

                    if pd.isna(cv):
                        return "🔴 Baja"

                    if cv < 5:
                        return "🟢 Alta"

                    if cv <= 10:
                        return "🟡 Media"

                    return "🔴 Baja"

                resumen_benchmark["Confiabilidad"] = resumen_benchmark.apply(
                    clasificar_confiabilidad,
                    axis=1
                )

                resumen_benchmark = resumen_benchmark[
                    [
                        "Giro comercial",
                        "Densidad de demanda promedio",
                        "No. de locales en la muestra",
                        "CV (%)",
                        "Confiabilidad"
                    ]
                ]

                st.dataframe(
                    resumen_benchmark,
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "Confiabilidad: 🟢 Alta si CV < 5% y muestra ≥ 10 locales; "
                    "🟡 Media si CV entre 5% y 10% y muestra ≥ 10 locales; "
                    "🔴 Baja si CV > 10% o muestra < 10 locales."
                )

            else:
                st.info("No hay datos para la combinación seleccionada.")

        else:
            st.info(
                "La tabla se llenará cuando el modelo genere "
                "`demanda_real_anual_df` con densidad de demanda por local."
            )

    else:
        st.warning(f"No encontré el archivo maestro de centros comerciales: {cc_master_path}")    



# ============================================================
# Data quality
# ============================================================
with tab_calidad:
    st.markdown('<div class="section-title">2. Calidad de datos</div>', unsafe_allow_html=True)

    st.markdown("### Representatividad de la muestra")

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

    st.markdown("### Cobertura de muestra por centro comercial")

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

    def has_partial_match_cc(value, candidates):
        if not value:
            return False

        value = str(value).strip()
        value_compact = value.replace(" ", "")

        stopwords = {
            "DE", "DEL", "LA", "LAS", "LOS", "EL", "Y",
            "SA", "CV", "S", "RL", "SAPI", "MEXICO",
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

            if value_words and candidate_words and (value_words & candidate_words):
                return True

        return False

    general_mall_col = first_existing_column(
        general_data,
        ["NOMBRE DEL CC", "CENTRO COMERCIAL", "CC", "PLAZA", "source_sheet"]
    )

    coverage_rows = []

    if mall_col and general_mall_col and not general_data.empty:

        for mall_name in sorted(parsed[mall_col].dropna().unique()):

            parser_cc = parsed[
                parsed[mall_col] == mall_name
            ].copy()

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
                ["No. De medidor", "No. de medidor", "No de medidor", "MEDIDOR", "Medidor"]
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
                set(normalize_meter_cc(parser_cc["medidor"]).unique())
                if "medidor" in parser_cc.columns
                else set()
            )

            parser_clientes = (
                set(normalize_cc_key(parser_cc["cliente_nombre"]).unique())
                if "cliente_nombre" in parser_cc.columns
                else set()
            )

            parser_nombres = (
                set(normalize_cc_key(parser_cc["recibos_subgroup"]).unique())
                if "recibos_subgroup" in parser_cc.columns
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
                general_ocupados_cc["_key_cliente"].isin(parser_clientes)
                & (general_ocupados_cc["_key_cliente"] != "")
            )

            general_ocupados_cc["_match_nombre_comercial"] = (
                general_ocupados_cc["_key_nombre_comercial"]
                .apply(lambda x: has_partial_match_cc(x, parser_nombres))
            )

            general_ocupados_cc["_tiene_recibo"] = (
                general_ocupados_cc["_match_medidor"]
                | general_ocupados_cc["_match_cliente"]
                | general_ocupados_cc["_match_nombre_comercial"]
            )

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

        coverage_by_mall = pd.DataFrame(coverage_rows)

        if not coverage_by_mall.empty:
            st.dataframe(
                coverage_by_mall.sort_values("Centro Comercial"),
                use_container_width=True
            )
        else:
            st.info("No se encontraron centros comerciales con cobertura calculable.")

    else:
        st.warning(
            "No encontré columnas suficientes para calcular cobertura por centro comercial."
        )

    st.markdown("### Recibos sin número de servicio pero con medidor")

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

    st.markdown("### Servicios con más de un medidor")

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

    st.markdown("### Resumen por centro comercial")

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

    st.markdown("### Cobertura de campos del parser")

    if not quality_df.empty:
        st.dataframe(
            quality_df,
            use_container_width=True
        )

        chart_df = quality_df.set_index("campo")[["cobertura_%"]]
        st.bar_chart(chart_df)
    else:
        st.info("No hay columnas suficientes para calcular cobertura de campos.")


with tab_general:
    st.markdown("### Distribución MT vs BT")

    universo_usuarios = muestra_con_recibo.copy() if "muestra_con_recibo" in locals() else general_data.copy()

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
        st.markdown("#### Usuarios por nivel de tensión")

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
        st.markdown("#### Demanda real por nivel de tensión")

        demanda_col = first_existing_column(
            universo_usuarios,
            [
                "demanda_real_anual_kw",
                "Demanda real kW",
                "demanda_maxima_kw",
                "Demanda Contratada (kW)_num",
                "Demanda Contratada (kW)"
            ]
        )

        if demanda_col:

            universo_usuarios["Demanda real análisis kW"] = clean_number_series(
                universo_usuarios[demanda_col]
            )

            demanda_mt_bt = (
                universo_usuarios[
                    universo_usuarios["Nivel de tensión"].isin(["MT", "BT"])
                ]
                .groupby("Nivel de tensión")["Demanda real análisis kW"]
                .sum()
                .reset_index(name="Demanda real total kW")
            )

            total_demanda_mt_bt = demanda_mt_bt["Demanda real total kW"].sum()

            demanda_mt_bt["(%)"] = (
                demanda_mt_bt["Demanda real total kW"] / total_demanda_mt_bt * 100
            ).round(1)

            demanda_mt_bt_display = demanda_mt_bt.copy()
            demanda_mt_bt_display["Demanda real total kW"] = demanda_mt_bt_display[
                "Demanda real total kW"
            ].round(1)
            demanda_mt_bt_display["(%)"] = demanda_mt_bt_display["(%)"].map(lambda x: f"{x:.1f}%")

            fig_demanda = demanda_mt_bt.set_index("Nivel de tensión").plot.pie(
                y="Demanda real total kW",
                autopct="%1.1f%%",
                figsize=(4, 4),
                legend=False
            ).figure

            st.pyplot(fig_demanda)

            st.dataframe(
                demanda_mt_bt_display,
                use_container_width=True
            )

            st.caption(f"Columna usada provisionalmente para demanda: `{demanda_col}`")

        else:
            st.info(
                "Aún no encontré una columna de demanda real. "
                "Esta gráfica se activará cuando el parser genere la demanda real anual."
            )

    # ============================================================
    # Distribución por Clima
    # ============================================================

    st.markdown("### Distribución por clima")

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
        ax.set_xlabel("% de centros comerciales")
        ax.set_yticks([])
        ax.set_title("Distribución de centros comerciales por clima")

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
            ncol=len(clima_dist)
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

    st.markdown("### Distribución por tipo de centro comercial")

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
            ax.set_xlabel("% de centros comerciales")
            ax.set_yticks([])
            ax.set_title(
                "Distribución de centros comerciales por tipo",
                fontsize=12
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
                bbox_to_anchor=(0.5, -0.25),
                ncol=4,
                fontsize=8,
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

    # ============================================================
    # Consumption and billing distribution
    # ============================================================

    st.markdown('<div class="section-title">3. Distribución de consumo y facturación</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribución de kWh por recibo")
        if "kwh_total_num" in filtered.columns:
            kwh_data = filtered[["kwh_total_num"]].dropna()
            if not kwh_data.empty:
                st.bar_chart(kwh_data.reset_index(drop=True))
            else:
                st.info("No hay valores de kWh para graficar.")
        else:
            st.info("No existe columna `kwh_total`.")

    with col_b:
        st.subheader("Distribución de importe por recibo")
        if "importe_total_num" in filtered.columns:
            amount_data = filtered[["importe_total_num"]].dropna()
            if not amount_data.empty:
                st.bar_chart(amount_data.reset_index(drop=True))
            else:
                st.info("No hay valores de importe para graficar.")
        else:
            st.info("No existe columna `importe_total`.")


    # ============================================================
    # Rankings
    # ============================================================

    st.markdown('<div class="section-title">4. Rankings de consumo y facturación</div>', unsafe_allow_html=True)

    ranking_group_col = tenant_col or service_col or mall_col

    if ranking_group_col:
        ranking_df = filtered.copy()

        aggregations = {}

        if "kwh_total_num" in ranking_df.columns:
            aggregations["kwh_total_num"] = "sum"

        if "importe_total_num" in ranking_df.columns:
            aggregations["importe_total_num"] = "sum"

        if aggregations:
            grouped = (
                ranking_df
                .groupby(ranking_group_col, dropna=False)
                .agg(aggregations)
                .reset_index()
            )

            if "kwh_total_num" in grouped.columns and "importe_total_num" in grouped.columns:
                grouped["mxn_per_kwh"] = grouped["importe_total_num"] / grouped["kwh_total_num"]
                grouped.loc[grouped["kwh_total_num"] <= 0, "mxn_per_kwh"] = pd.NA

            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                st.subheader("Top 15 por kWh")
                if "kwh_total_num" in grouped.columns:
                    top_kwh = grouped.sort_values("kwh_total_num", ascending=False).head(15)
                    st.dataframe(top_kwh, use_container_width=True)
                    st.bar_chart(top_kwh.set_index(ranking_group_col)[["kwh_total_num"]])

            with col_r2:
                st.subheader("Top 15 por importe")
                if "importe_total_num" in grouped.columns:
                    top_amount = grouped.sort_values("importe_total_num", ascending=False).head(15)
                    st.dataframe(top_amount, use_container_width=True)
                    st.bar_chart(top_amount.set_index(ranking_group_col)[["importe_total_num"]])

            with col_r3:
                st.subheader("Top 15 por MXN/kWh")
                if "mxn_per_kwh" in grouped.columns:
                    top_cost = (
                        grouped
                        .dropna(subset=["mxn_per_kwh"])
                        .sort_values("mxn_per_kwh", ascending=False)
                        .head(15)
                    )
                    st.dataframe(top_cost, use_container_width=True)
                    if not top_cost.empty:
                        st.bar_chart(top_cost.set_index(ranking_group_col)[["mxn_per_kwh"]])
        else:
            st.info("No hay columnas numéricas suficientes para rankings.")
    else:
        st.info("No encontré columna de locatario, servicio o centro comercial para agrupar.")


    # ============================================================
    # Monthly evolution from parsed bills
    # ============================================================

    st.markdown('<div class="section-title">5. Evolución mensual desde recibos parseados</div>', unsafe_allow_html=True)

    date_col = None
    for candidate in ["periodo_inicio_dt", "periodo_fin_dt", "limite_pago_dt", "parsed_at_dt"]:
        if candidate in filtered.columns and filtered[candidate].notna().any():
            date_col = candidate
            break

    if date_col:
        time_df = filtered.copy()
        time_df["month"] = time_df[date_col].dt.to_period("M").astype(str)

        monthly_aggs = {}

        if "kwh_total_num" in time_df.columns:
            monthly_aggs["kwh_total_num"] = "sum"

        if "importe_total_num" in time_df.columns:
            monthly_aggs["importe_total_num"] = "sum"

        if monthly_aggs:
            monthly = (
                time_df
                .groupby("month")
                .agg(monthly_aggs)
                .reset_index()
                .sort_values("month")
            )

            if "kwh_total_num" in monthly.columns and "importe_total_num" in monthly.columns:
                monthly["mxn_per_kwh"] = monthly["importe_total_num"] / monthly["kwh_total_num"]
                monthly.loc[monthly["kwh_total_num"] <= 0, "mxn_per_kwh"] = pd.NA

            st.dataframe(monthly, use_container_width=True)

            col_m1, col_m2, col_m3 = st.columns(3)

            with col_m1:
                if "kwh_total_num" in monthly.columns:
                    st.subheader("kWh mensual")
                    st.line_chart(monthly.set_index("month")[["kwh_total_num"]])

            with col_m2:
                if "importe_total_num" in monthly.columns:
                    st.subheader("Importe mensual")
                    st.line_chart(monthly.set_index("month")[["importe_total_num"]])

            with col_m3:
                if "mxn_per_kwh" in monthly.columns:
                    st.subheader("MXN/kWh mensual")
                    st.line_chart(monthly.set_index("month")[["mxn_per_kwh"]])
        else:
            st.info("No hay columnas numéricas suficientes para evolución mensual.")
    else:
        st.info(
            "No encontré fechas utilizables para construir evolución mensual "
            "desde los recibos parseados."
        )


    # ============================================================
    # Historical table exploration
    # ============================================================

    st.markdown('<div class="section-title">6. Exploración del histórico de consumo</div>', unsafe_allow_html=True)

    if historico.empty:
        st.info("No se encontró archivo histórico o no contiene filas.")
    else:
        st.write(f"Filas históricas disponibles: **{len(historico):,}**")
        st.dataframe(historico.head(100), use_container_width=True)

        hist_numeric_cols = [c for c in historico.columns if c.endswith("_num")]
        hist_date_cols = [c for c in historico.columns if c.endswith("_dt")]

        if hist_numeric_cols:
            selected_hist_metric = st.selectbox(
                "Métrica histórica para explorar",
                options=hist_numeric_cols
            )

            usable_date_col = None
            for c in hist_date_cols:
                if historico[c].notna().any():
                    usable_date_col = c
                    break

            if usable_date_col:
                hdf = historico.copy()
                hdf["month"] = hdf[usable_date_col].dt.to_period("M").astype(str)
                hmonthly = (
                    hdf
                    .groupby("month")[selected_hist_metric]
                    .sum()
                    .reset_index()
                    .sort_values("month")
                )

                st.subheader("Serie histórica agregada")
                st.dataframe(hmonthly, use_container_width=True)
                st.line_chart(hmonthly.set_index("month")[[selected_hist_metric]])
            else:
                st.subheader("Distribución de métrica histórica")
                st.bar_chart(historico[[selected_hist_metric]].dropna().reset_index(drop=True))
        else:
            st.info(
                "El histórico fue cargado, pero todavía no identifiqué una columna numérica estándar. "
                "Podemos adaptar esta sección al formato exacto de `bills_historico_v2.csv`."
            )


    # ============================================================
    # Outlier candidates
    # ============================================================

    st.markdown('<div class="section-title">7. Candidatos a revisión</div>', unsafe_allow_html=True)

    review_cols = []

    for col in [
        mall_col,
        tenant_col,
        service_col,
        "cliente_nombre",
        "tarifa",
        "periodo_inicio",
        "periodo_fin",
        "kwh_total",
        "importe_total",
        "mxn_per_kwh",
    ]:
        if col and col in filtered.columns and col not in review_cols:
            review_cols.append(col)

    candidates = filtered.copy()

    if "mxn_per_kwh" in candidates.columns and candidates["mxn_per_kwh"].notna().any():
        q95 = candidates["mxn_per_kwh"].dropna().quantile(0.95)
        out_cost = candidates[candidates["mxn_per_kwh"] >= q95].copy()
    else:
        out_cost = pd.DataFrame()

    if "kwh_total_num" in candidates.columns and candidates["kwh_total_num"].notna().any():
        q95_kwh = candidates["kwh_total_num"].dropna().quantile(0.95)
        out_kwh = candidates[candidates["kwh_total_num"] >= q95_kwh].copy()
    else:
        out_kwh = pd.DataFrame()

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        st.subheader("Costo MXN/kWh alto")
        if not out_cost.empty and review_cols:
            st.dataframe(out_cost[review_cols].head(50), use_container_width=True)
        else:
            st.info("No hay suficientes datos para detectar costo alto.")

    with col_o2:
        st.subheader("Consumo kWh alto")
        if not out_kwh.empty and review_cols:
            st.dataframe(out_kwh[review_cols].head(50), use_container_width=True)
        else:
            st.info("No hay suficientes datos para detectar consumo alto.")




    # ============================================================
    # Energy intensity by area
    # ============================================================

    st.markdown('<div class="section-title">9. Intensidad energética por superficie</div>', unsafe_allow_html=True)

    if general_data.empty:
        st.info("Para calcular kWh/m² y MXN/m² se necesita cargar el archivo de datos generales.")
    elif not use_general_merge:
        st.info("Activa el cruce con datos generales desde la barra lateral.")
    elif not receipt_key_col or not general_key_col or not area_col:
        st.info("Selecciona llaves de cruce y columna de superficie en la barra lateral.")
    elif "kwh_total_num" not in filtered.columns or "importe_total_num" not in filtered.columns:
        st.info("Se requieren columnas kWh e importe numéricas para calcular intensidad.")
    else:
        st.markdown(
            """
            <div class="note-box">
            Esta sección cruza los recibos parseados con el archivo de datos generales.
            Como el archivo de datos generales aún no está completo, el cruce puede ser parcial.
            Los registros sin coincidencia o sin superficie se reportan explícitamente.
            </div>
            """,
            unsafe_allow_html=True
        )

        receipts_for_merge = filtered.copy()
        general_for_merge = general_data.copy()

        receipts_for_merge["_merge_key"] = normalize_text_key(receipts_for_merge[receipt_key_col])
        general_for_merge["_merge_key"] = normalize_text_key(general_for_merge[general_key_col])

        area_num_col = area_col + "_num"
        if area_num_col not in general_for_merge.columns:
            general_for_merge[area_num_col] = clean_number_series(general_for_merge[area_col])

        # Aggregate receipts by merge key before merging.
        receipt_aggs = {
            "kwh_total_num": "sum",
            "importe_total_num": "sum",
        }

        if mall_col:
            receipt_aggs[mall_col] = "first"
        if tenant_col:
            receipt_aggs[tenant_col] = "first"
        if service_col:
            receipt_aggs[service_col] = "first"
        if "cliente_nombre" in receipts_for_merge.columns:
            receipt_aggs["cliente_nombre"] = "first"
        if "medidor" in receipts_for_merge.columns:
            receipt_aggs["medidor"] = "first"

        receipts_grouped = (
            receipts_for_merge
            .dropna(subset=["_merge_key"])
            .groupby("_merge_key", dropna=False)
            .agg(receipt_aggs)
            .reset_index()
        )

        # Deduplicate general data by merge key.
        # If repeated keys exist, sum area and preserve first descriptive values.
        general_aggs = {
            area_num_col: "sum",
        }

        for c in [
            "source_sheet",
            "NOMBRE DEL CC",
            "CLIENTE",
            "NOMBRE COMERCIAL",
            "SUBGIRO_COMERCIAL",
            "AREA",
            "TIPO LOCAL",
            "No de Local",
            "NIVEL",
            "TARIFA",
            "No. De medidor",
            "Carga Conectada (kW)",
            "Demanda Contratada (kW)",
        ]:
            if c in general_for_merge.columns:
                general_aggs[c] = "first"

        general_grouped = (
            general_for_merge
            .dropna(subset=["_merge_key"])
            .groupby("_merge_key", dropna=False)
            .agg(general_aggs)
            .reset_index()
        )

        intensity = receipts_grouped.merge(
            general_grouped,
            on="_merge_key",
            how="left",
            indicator=True
        )

        intensity["merge_status"] = intensity["_merge"].map({
            "both": "matched",
            "left_only": "no_general_data",
            "right_only": "only_general_data",
        })

        intensity["area_m2"] = intensity[area_num_col]

        intensity["kwh_per_m2"] = intensity["kwh_total_num"] / intensity["area_m2"]
        intensity["mxn_per_m2"] = intensity["importe_total_num"] / intensity["area_m2"]

        intensity.loc[intensity["area_m2"] <= 0, ["kwh_per_m2", "mxn_per_m2"]] = pd.NA

        matched_count = (intensity["merge_status"] == "matched").sum()
        missing_general_count = (intensity["merge_status"] == "no_general_data").sum()
        area_available_count = intensity["area_m2"].notna().sum()

        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        col_i1.metric("Registros cruzados", f"{matched_count:,}")
        col_i2.metric("Sin datos generales", f"{missing_general_count:,}")
        col_i3.metric("Con superficie m²", f"{area_available_count:,}")
        col_i4.metric("Llave usada", f"{receipt_key_col} ↔ {general_key_col}")

        intensity_display_cols = []

        for c in [
            mall_col,
            tenant_col,
            service_col,
            "cliente_nombre",
            "medidor",
            "source_sheet",
            "NOMBRE COMERCIAL",
            "CLIENTE",
            "SUBGIRO_COMERCIAL",
            "AREA",
            "TIPO LOCAL",
            "No de Local",
            "TARIFA",
            "area_m2",
            "kwh_total_num",
            "importe_total_num",
            "kwh_per_m2",
            "mxn_per_m2",
            "merge_status",
        ]:
            if c and c in intensity.columns and c not in intensity_display_cols:
                intensity_display_cols.append(c)

        col_t1, col_t2 = st.columns(2)

        with col_t1:
            st.subheader("Top 15 por kWh/m²")
            top_kwh_m2 = (
                intensity
                .dropna(subset=["kwh_per_m2"])
                .sort_values("kwh_per_m2", ascending=False)
                .head(15)
            )

            if not top_kwh_m2.empty:
                st.dataframe(top_kwh_m2[intensity_display_cols], use_container_width=True)

                label_col = "NOMBRE COMERCIAL" if "NOMBRE COMERCIAL" in top_kwh_m2.columns else "_merge_key"
                st.bar_chart(top_kwh_m2.set_index(label_col)[["kwh_per_m2"]])
            else:
                st.info("No hay datos suficientes para kWh/m².")

        with col_t2:
            st.subheader("Top 15 por MXN/m²")
            top_mxn_m2 = (
                intensity
                .dropna(subset=["mxn_per_m2"])
                .sort_values("mxn_per_m2", ascending=False)
                .head(15)
            )

            if not top_mxn_m2.empty:
                st.dataframe(top_mxn_m2[intensity_display_cols], use_container_width=True)

                label_col = "NOMBRE COMERCIAL" if "NOMBRE COMERCIAL" in top_mxn_m2.columns else "_merge_key"
                st.bar_chart(top_mxn_m2.set_index(label_col)[["mxn_per_m2"]])
            else:
                st.info("No hay datos suficientes para MXN/m².")

        if "SUBGIRO_COMERCIAL" in intensity.columns:
            st.subheader("Promedios por subgiro comercial")

            subgiro_summary = (
                intensity
                .dropna(subset=["SUBGIRO_COMERCIAL"])
                .groupby("SUBGIRO_COMERCIAL")
                .agg(
                    registros=("SUBGIRO_COMERCIAL", "size"),
                    area_m2=("area_m2", "sum"),
                    kwh_total=("kwh_total_num", "sum"),
                    importe_total=("importe_total_num", "sum"),
                    kwh_per_m2_prom=("kwh_per_m2", "mean"),
                    mxn_per_m2_prom=("mxn_per_m2", "mean"),
                )
                .reset_index()
                .sort_values("kwh_per_m2_prom", ascending=False)
            )

            st.dataframe(subgiro_summary.head(50), use_container_width=True)

        with st.expander("Ver tabla completa de cruce e intensidad"):
            st.dataframe(intensity[intensity_display_cols].head(500), use_container_width=True)

        intensity_csv = intensity.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Descargar cruce con intensidad energética CSV",
            data=intensity_csv,
            file_name="allux_energy_intensity_by_area.csv",
            mime="text/csv"
        )


    # ============================================================
    # Raw filtered data download
    # ============================================================

    st.markdown('<div class="section-title">10. Datos filtrados</div>', unsafe_allow_html=True)

    st.write("Vista previa de los datos filtrados usados en este reporte.")

    st.dataframe(filtered.head(200), use_container_width=True)

    csv_data = filtered.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Descargar datos filtrados CSV",
        data=csv_data,
        file_name="allux_live_report_filtered_data.csv",
        mime="text/csv"
    )

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

        st.markdown(
            '<div class="section-title">Calidad de muestra</div>',
            unsafe_allow_html=True
        )

        col_ocupacion, col_muestra = st.columns(2)

        with col_ocupacion:
            st.markdown(
                """
                <h4 style='margin-bottom:10px;'>
                    Ocupación del centro comercial
                </h4>
                """,
                unsafe_allow_html=True
            )

            total_locales = len(general_cc_data)
            disponibles_vacios = int(disponible_o_vacio.sum())

            ocupacion_pie = pd.DataFrame({
                "Estatus": ["Ocupados", "Disponibles o vacíos"],
                "Cantidad": [locales_ocupados, disponibles_vacios]
            })

            fig_ocupacion = ocupacion_pie.set_index("Estatus").plot.pie(
                y="Cantidad",
                autopct="%1.1f%%",
                figsize=(5, 5),
                legend=False
            ).figure

            st.pyplot(fig_ocupacion)

            st.dataframe(
                pd.DataFrame({
                    "Métrica": [
                        "Total de filas en datos generales",
                        "Total de locales ocupados",
                        "Filas vacías o disponibles"
                    ],
                    "Cantidad": [
                        total_locales,
                        locales_ocupados,
                        disponibles_vacios
                    ]
                }),
                use_container_width=True
            )

        with col_muestra:
            st.markdown(
                """
                <h4 style='margin-bottom:10px;'>
                    Muestra disponible
                </h4>
                """,
                unsafe_allow_html=True
            )

            muestra_pie = pd.DataFrame({
                "Estatus": ["Con recibo en parser", "Sin recibo en parser"],
                "Cantidad": [locales_con_recibo, locales_sin_recibo]
            })

            fig_muestra = muestra_pie.set_index("Estatus").plot.pie(
                y="Cantidad",
                autopct="%1.1f%%",
                figsize=(5, 5),
                legend=False
            ).figure

            st.pyplot(fig_muestra)

            st.dataframe(
                pd.DataFrame({
                    "Métrica": [
                        "Locales ocupados",
                        "Locales ocupados con recibo",
                        "Locales ocupados sin recibo",
                        "Cobertura de muestra (%)"
                    ],
                    "Cantidad": [
                        locales_ocupados,
                        locales_con_recibo,
                        locales_sin_recibo,
                        cobertura_pct
                    ]
                }),
                use_container_width=True
            )

        with st.expander("Diagnóstico de criterios de cruce"):
            st.write("Cruce por medidor:", int(general_ocupados["_match_medidor"].sum()))
            st.write("Cruce por cliente:", int(general_ocupados["_match_cliente"].sum()))
            st.write("Cruce por nombre comercial:", int(general_ocupados["_match_nombre_comercial"].sum()))

        # ============================================================
        # Parser sin match contra Datos Generales
        # ============================================================

        matched_clientes = set(general_ocupados.loc[general_ocupados["_tiene_recibo"], "_key_cliente"])
        matched_nombres = set(general_ocupados.loc[general_ocupados["_tiene_recibo"], "_key_nombre_comercial"])
        matched_medidores = set(general_ocupados.loc[general_ocupados["_tiene_recibo"], "_key_medidor"])

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
            parser_unmatched["_cliente_key"].isin(matched_clientes)
            | parser_unmatched["_nombre_key"].isin(matched_nombres)
            | parser_unmatched["_medidor_key"].isin(matched_medidores)
            | parser_unmatched["_match_dg_cliente"]
            | parser_unmatched["_match_dg_nombre"]
        )

        parser_unmatched = parser_unmatched[
            ~parser_unmatched["_match_dg"]
        ]

        st.markdown(
            """
            <h4 style='margin-top:20px; margin-bottom:10px;'>
                Recibos encontrados en parser sin match en Datos Generales
            </h4>
            """,
            unsafe_allow_html=True
        )

        st.write(
            "Servicios únicos en parser sin match:",
            parser_unmatched["no_servicio"].nunique()
            if "no_servicio" in parser_unmatched.columns
            else len(parser_unmatched)
        )

        parser_cols = [
            "cliente_nombre",
            "recibos_subgroup",
            "no_servicio",
            "medidor",
            "tarifa"
        ]

        parser_cols = [col for col in parser_cols if col in parser_unmatched.columns]

        st.dataframe(
            parser_unmatched[parser_cols].drop_duplicates(),
            use_container_width=True,
            height=600
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

        muestra_con_recibo = general_ocupados[
            general_ocupados["_tiene_recibo"]
        ].copy()

        # ============================================================
        # DIAGNÓSTICO TEMPORAL DE TARIFAS
        # ============================================================


        if muestra_con_recibo["TARIFA_ANALISIS"].isna().sum() > 0:

            cols_debug = [
                c for c in [
                    "CLIENTE",
                    "NOMBRE COMERCIAL",
                    "No de Local",
                    "TARIFA",
                    "TARIFA_ANALISIS",
                    "_tiene_recibo",
                    "Criterio de cruce"
                ]
                if c in muestra_con_recibo.columns
            ]

            st.dataframe(
                muestra_con_recibo[
                    muestra_con_recibo["TARIFA_ANALISIS"].isna()
                ][cols_debug],
                use_container_width=True
            )

            st.write("Ejemplos parser:")

            cols_debug = [
                c for c in [
                    "cliente_nombre",
                    "recibos_subgroup",
                    "tarifa",
                    "no_servicio"
                ]
                if c in cc_parser.columns
            ]

            st.dataframe(
                cc_parser[cols_debug],
                use_container_width=True,
                height=500
            )

        # ------------------------------------------------------------
        # Número de usuarios por tarifa
        # ------------------------------------------------------------

        st.markdown(
            """
            <h5 style='margin-top:15px; margin-bottom:10px;'>
                Número de usuarios por tarifa
            </h5>
            """,
            unsafe_allow_html=True
        )

        if "TARIFA_ANALISIS" in muestra_con_recibo.columns and not muestra_con_recibo.empty:

            tarifa_comp = (
                muestra_con_recibo
                .dropna(subset=["TARIFA_ANALISIS"])
                .groupby("TARIFA_ANALISIS")
                .size()
                .reset_index(name="Número de usuarios")
                .rename(columns={"TARIFA_ANALISIS": "Tarifa"})
                .sort_values("Número de usuarios", ascending=False)
            )

            total_tarifa = tarifa_comp["Número de usuarios"].sum()

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

            col_tarifa_pie, col_tarifa_table = st.columns(2)

            with col_tarifa_pie:
                fig_tarifa = tarifa_comp.set_index("Tarifa").plot.pie(
                    y="Número de usuarios",
                    autopct="%1.1f%%",
                    figsize=(5, 5),
                    legend=False
                ).figure

                st.pyplot(fig_tarifa)

            with col_tarifa_table:
                st.dataframe(tarifa_comp_display, use_container_width=True)

        else:
            st.info("No hay información de tarifa para la muestra con recibo.")

        # ------------------------------------------------------------
        # Número de usuarios por giro comercial
        # ------------------------------------------------------------

        st.markdown(
            """
            <h5 style='margin-top:20px; margin-bottom:10px;'>
                Número de usuarios por giro comercial
            </h5>
            """,
            unsafe_allow_html=True
        )

        if "SUBGIRO_COMERCIAL" in muestra_con_recibo.columns and not muestra_con_recibo.empty:

            muestra_con_recibo["GIRO_NORMALIZADO"] = (
                muestra_con_recibo["SUBGIRO_COMERCIAL"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.title()
            )

            giro_comp = (
                muestra_con_recibo
                .dropna(subset=["GIRO_NORMALIZADO"])
                .groupby("GIRO_NORMALIZADO")
                .size()
                .reset_index(name="Número de usuarios")
                .rename(columns={"GIRO_NORMALIZADO": "Giro comercial"})
                .sort_values("Número de usuarios", ascending=False)
            )

            giro_comp = giro_comp[
                giro_comp["Giro comercial"].astype(str).str.strip() != ""
            ]

            total_giro = giro_comp["Número de usuarios"].sum()

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
            col_giro_pie, col_giro_table = st.columns(2)

            with col_giro_pie:
                fig_giro = giro_comp.set_index("Giro comercial").plot.pie(
                    y="Número de usuarios",
                    autopct="%1.1f%%",
                    figsize=(5, 5),
                    legend=False
                ).figure

                st.pyplot(fig_giro)

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

        if (
            "demanda_contratada_kw" in cc_parser.columns
            and "carga_conectada_kw" in cc_parser.columns
        ):

            parser_demanda = cc_parser.copy()

            parser_demanda["_key_cliente"] = (
                normalize_key_series(parser_demanda["cliente_nombre"])
                if "cliente_nombre" in parser_demanda.columns
                else ""
            )

            parser_demanda["_key_nombre_comercial"] = (
                normalize_key_series(parser_demanda["recibos_subgroup"])
                if "recibos_subgroup" in parser_demanda.columns
                else ""
            )

            parser_demanda["_demanda_contratada"] = clean_number_series(
                parser_demanda["demanda_contratada_kw"]
            )

            parser_demanda["_demanda_real"] = clean_number_series(
                parser_demanda["carga_conectada_kw"]
            )

            parser_demanda_lookup = {}

            for _, r in parser_demanda.iterrows():

                if pd.notna(r["_demanda_contratada"]) and pd.notna(r["_demanda_real"]):

                    demanda_values = {
                        "contratada": r["_demanda_contratada"],
                        "real": r["_demanda_real"]
                    }

                    if r.get("_key_cliente"):
                        parser_demanda_lookup[r["_key_cliente"]] = demanda_values

                    if r.get("_key_nombre_comercial"):
                        parser_demanda_lookup[r["_key_nombre_comercial"]] = demanda_values


            def get_demanda_from_parser(row):

                cliente_key = row.get("_key_cliente")
                nombre_key = row.get("_key_nombre_comercial")

                if cliente_key in parser_demanda_lookup:
                    return parser_demanda_lookup[cliente_key]

                if nombre_key in parser_demanda_lookup:
                    return parser_demanda_lookup[nombre_key]

                for parser_key, demanda_values in parser_demanda_lookup.items():

                    if has_partial_match(cliente_key, [parser_key]):
                        return demanda_values

                    if has_partial_match(nombre_key, [parser_key]):
                        return demanda_values

                return None


            demanda_df = muestra_con_recibo.copy()

            demanda_df["_demanda_values"] = demanda_df.apply(
                get_demanda_from_parser,
                axis=1
            )

            demanda_df["Demanda contratada kW"] = (
                demanda_df["_demanda_values"]
                .apply(lambda x: x["contratada"] if isinstance(x, dict) else None)
            )

            demanda_df["Demanda real kW"] = (
                demanda_df["_demanda_values"]
                .apply(lambda x: x["real"] if isinstance(x, dict) else None)
            )

            demanda_df = demanda_df.dropna(
                subset=[
                    "Demanda contratada kW",
                    "Demanda real kW"
                ]
            )

            demanda_df = demanda_df[
                demanda_df["Demanda contratada kW"] > 0
            ]

            demanda_df["Contratada (%)"] = 100

            demanda_df["Real (%)"] = (
                demanda_df["Demanda real kW"]
                / demanda_df["Demanda contratada kW"]
                * 100
            )

            demanda_df["Nombre Comercial"] = (
                demanda_df["NOMBRE COMERCIAL"]
            )

            demanda_df["Tarifa"] = (
                demanda_df["TARIFA_ANALISIS"]
            )

            demanda_df = demanda_df.sort_values(
                ["Tarifa", "Nombre Comercial"]
            )

            if not demanda_df.empty:

                orden_tarifas = {
                    "GDMTH": 1,
                    "GDMTO": 2,
                    "PDBT": 3,
                    "GDBT": 4
                }

                demanda_df["_orden_tarifa"] = (
                    demanda_df["Tarifa"]
                    .map(orden_tarifas)
                    .fillna(999)
                )

                demanda_df = demanda_df.sort_values(
                    ["_orden_tarifa", "Nombre Comercial"]
                )

                chart_df = demanda_df.set_index(
                    "Nombre Comercial"
                )[
                    [
                        "Contratada (%)",
                        "Real (%)"
                    ]
                ]

                fig, ax = plt.subplots(
                    figsize=(max(16, len(chart_df) * 0.35), 6)
                )

                chart_df.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_ylim(0, 200)

                # Segundo nivel de etiquetas: tarifa debajo de los nombres
                tarifa_por_local = demanda_df.set_index("Nombre Comercial")["Tarifa"]

                grupos_tarifa = (
                    demanda_df
                    .reset_index(drop=True)
                    .groupby("Tarifa", sort=False)
                    .apply(lambda x: (x.index.min(), x.index.max()))
                )

                for tarifa, (inicio, fin) in grupos_tarifa.items():
                    centro = (inicio + fin) / 2

                    ax.text(
                        centro,
                        -80,
                        tarifa,
                        ha="center",
                        va="top",
                        fontsize=11,
                        fontweight="bold",
                        transform=ax.transData
                    )

                    ax.axvline(
                        fin + 0.5,
                        color="lightgray",
                        linewidth=0.8
                    )

                fig.subplots_adjust(bottom=0.50)

                ax.set_ylim(0, 200)

                ax.set_ylabel(
                    "% de demanda contratada"
                )

                ax.set_xlabel(
                    "Locales ocupados con recibo"
                )

                ax.set_title(
                    "Demanda máxima como % de la demanda contratada"
                )

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

                st.dataframe(
                    demanda_df[
                        [
                            "Tarifa",
                            "Nombre Comercial",
                            "Demanda contratada kW",
                            "Demanda real kW",
                            "Real (%)"
                        ]
                    ],
                    use_container_width=True
                )

            else:

                st.info(
                    "No encontré registros con demanda contratada y demanda máxima."
                )

        else:

            st.warning(
                "No encontré las columnas demanda_contratada_kw o demanda_maxima_kw en el parser."
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

        if "demanda_df" in locals() and not demanda_df.empty:

            densidad_df = muestra_con_recibo.copy()

            densidad_df["Nombre Comercial"] = (
                densidad_df["NOMBRE COMERCIAL"]
                if "NOMBRE COMERCIAL" in densidad_df.columns
                else densidad_df["CLIENTE"]
            )

            densidad_df["Tarifa"] = densidad_df["TARIFA_ANALISIS"]

            if "demanda_maxima_kw" in cc_parser.columns:
                demanda_real_parser_col = "demanda_maxima_kw"
            else:
                demanda_real_parser_col = None

            area_col_cc = first_existing_column(
                densidad_df,
                ["MTS2_num", "MTS2", "M2", "m2", "MTS 2", "SUPERFICIE", "SUPERFICIE M2"]
            )

            if area_col_cc:

                if area_col_cc.endswith("_num"):
                    densidad_df["Area m2"] = densidad_df[area_col_cc]
                else:
                    densidad_df["Area m2"] = clean_number_series(densidad_df[area_col_cc])

                if demanda_real_parser_col:
                    parser_demanda_real = cc_parser.copy()
            
            area_col_cc = first_existing_column(
                densidad_df,
                ["MTS2_num", "MTS2", "M2", "m2", "MTS 2", "SUPERFICIE", "SUPERFICIE M2"]
            )

            if area_col_cc:



                if demanda_real_parser_col:

                    parser_demanda_real = cc_parser.copy()

                    parser_demanda_real["_key_cliente"] = (
                        normalize_key_series(parser_demanda_real["cliente_nombre"])
                        if "cliente_nombre" in parser_demanda_real.columns
                        else ""
                    )

                    parser_demanda_real["_key_nombre_comercial"] = (
                        normalize_key_series(parser_demanda_real["recibos_subgroup"])
                        if "recibos_subgroup" in parser_demanda_real.columns
                        else ""
                    )

                    parser_demanda_real["_demanda_real_kw"] = clean_number_series(
                        parser_demanda_real[demanda_real_parser_col]
                    )

                    demanda_real_lookup = {}

                    for _, r in parser_demanda_real.dropna(subset=["_demanda_real_kw"]).iterrows():

                        if r.get("_key_cliente"):
                            demanda_real_lookup[r["_key_cliente"]] = r["_demanda_real_kw"]

                        if r.get("_key_nombre_comercial"):
                            demanda_real_lookup[r["_key_nombre_comercial"]] = r["_demanda_real_kw"]

                    def get_demanda_real_from_parser(row):
                        cliente_key = row.get("_key_cliente")
                        nombre_key = row.get("_key_nombre_comercial")

                        if cliente_key in demanda_real_lookup:
                            return demanda_real_lookup[cliente_key]

                        if nombre_key in demanda_real_lookup:
                            return demanda_real_lookup[nombre_key]

                        for parser_key, demanda_value in demanda_real_lookup.items():
                            if has_partial_match(cliente_key, [parser_key]):
                                return demanda_value

                            if has_partial_match(nombre_key, [parser_key]):
                                return demanda_value

                        return None

                    densidad_df["Demanda real kW"] = densidad_df.apply(
                        get_demanda_real_from_parser,
                        axis=1
                    )

                else:
                    densidad_df["Demanda real kW"] = pd.NA

                    st.info(
                        "Todavía no encontré `demanda_maxima_kw` en el parser. "
                        "La sección queda lista y se calculará cuando vuelvas a correr el parser con esa columna."
                    )

                densidad_df = densidad_df.dropna(
                    subset=["SUBGIRO_COMERCIAL", "Tarifa", "Nombre Comercial"]
                ).copy()

                densidad_df["Densidad de demanda kW/m2"] = pd.NA

                mask_densidad_valida = (
                    densidad_df["Demanda real kW"].notna()
                    & densidad_df["Area m2"].notna()
                    & (densidad_df["Area m2"] > 0)
                )

                densidad_df.loc[mask_densidad_valida, "Densidad de demanda kW/m2"] = (
                    densidad_df.loc[mask_densidad_valida, "Demanda real kW"]
                    / densidad_df.loc[mask_densidad_valida, "Area m2"]
                )

                giros_disponibles = sorted(
                    densidad_df["SUBGIRO_COMERCIAL"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .unique()
                )

                selected_giro_densidad = st.selectbox(
                    "Selecciona un giro comercial",
                    options=giros_disponibles,
                    key="giro_densidad_selector"
                )

                densidad_giro = densidad_df[
                    densidad_df["SUBGIRO_COMERCIAL"].astype(str).str.strip() == selected_giro_densidad
                ].copy()

                st.write(
                    "Usuarios del giro en composición:",
                    len(
                        muestra_con_recibo[
                            muestra_con_recibo["SUBGIRO_COMERCIAL"]
                            .astype(str)
                            .str.strip()
                            == selected_giro_densidad
                        ]
                    )
                )

                st.write(
                    "Usuarios del giro en densidad:",
                    len(densidad_giro)
                )

                st.write(
                    "Usuarios con densidad calculable:",
                    densidad_giro["Densidad de demanda kW/m2"].notna().sum()
                )

                # TODOS los usuarios del giro (para la tabla)
                densidad_grafica = densidad_giro.dropna(
                    subset=["Densidad de demanda kW/m2"]
                ).copy()

                orden_tarifas_densidad = {
                    "GDMTH": 1,
                    "GDMTO": 2,
                    "GDBT": 3,
                    "PDBT": 4
                }

                densidad_giro["_orden_tarifa"] = (
                    densidad_giro["Tarifa"]
                    .map(orden_tarifas_densidad)
                    .fillna(999)
                )

                densidad_grafica["_orden_tarifa"] = (
                    densidad_grafica["Tarifa"]
                    .map(orden_tarifas_densidad)
                    .fillna(999)
                )

                densidad_grafica = densidad_grafica.sort_values(
                    ["_orden_tarifa", "Nombre Comercial"]
                ).reset_index(drop=True)

                if not densidad_grafica.empty:

                    promedio = densidad_grafica["Densidad de demanda kW/m2"].mean()

                    desv = densidad_grafica["Densidad de demanda kW/m2"].std()

                    densidad_grafica["x"] = range(len(densidad_grafica))

                    fig, ax = plt.subplots(
                        figsize=(max(14, len(densidad_grafica) * 0.35), 6)
                    )

                    ax.scatter(
                        densidad_grafica["x"],
                        densidad_grafica["Densidad de demanda kW/m2"],
                        label="Densidad de demanda"
                    )

                    ax.axhline(promedio, linewidth=1.5, label="Promedio")
                    ax.axhline(promedio + desv, linewidth=1.5, label="+1 desv est")
                    ax.axhline(max(promedio - desv, 0), linewidth=1.5, label="-1 desv est")

                    ax.set_title(f"Densidad de demanda para {selected_giro_densidad}")
                    ax.set_ylabel("Densidad de demanda (kW/m2)")
                    ax.set_xlabel("Locales ocupados con recibo")

                    ax.set_xticks(densidad_grafica["x"])

                    ax.set_xticklabels(
                        densidad_grafica["Nombre Comercial"],
                        rotation=90,
                        fontsize=7
                    )

                    grupos_tarifa = (
                        densidad_grafica
                        .groupby("Tarifa", sort=False)
                        .apply(lambda x: (x["x"].min(), x["x"].max()))
                    )

                    for tarifa, (inicio, fin) in grupos_tarifa.items():
                        centro = (inicio + fin) / 2

                        ax.text(
                            centro,
                            -0.22,
                            tarifa,
                            ha="center",
                            va="top",
                            fontsize=11,
                            fontweight="bold",
                            transform=ax.get_xaxis_transform()
                        )

                        ax.axvline(
                            fin + 0.5,
                            color="lightgray",
                            linewidth=0.8
                        )

                    fig.subplots_adjust(bottom=0.45)

                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.32),
                        ncol=4
                    )

                    st.pyplot(fig)

                    st.dataframe(
                        densidad_giro[
                            [
                                "Tarifa",
                                "Nombre Comercial",
                                "SUBGIRO_COMERCIAL",
                                "Demanda real kW",
                                "Area m2",
                                "Densidad de demanda kW/m2"
                            ]
                        ],
                        use_container_width=True
                    )

                else:
                    st.info("No hay locales con datos suficientes para este giro comercial.")

            else:
                st.warning("No encontré columna de superficie/m2 para calcular densidad de demanda.")

        else:
            st.warning("Primero debe construirse la tabla de demanda para calcular densidad.")

    else:
        st.warning("No encontré columna de centro comercial.")

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

        parsed_sg = parsed.copy()

        # ------------------------------------------------------------
        # Columnas base
        # ------------------------------------------------------------

        cc_col = first_existing_column(
            parsed_sg,
            ["mall_folder", "centro_comercial", "NOMBRE DEL CC", "Centro Comercial"]
        )

        cliente_col = first_existing_column(
            parsed_sg,
            ["cliente_nombre", "CLIENTE", "cliente"]
        )

        subgroup_col = first_existing_column(
            parsed_sg,
            ["recibos_subgroup", "NOMBRE COMERCIAL", "nombre_comercial"]
        )

        medidor_col = first_existing_column(
            parsed_sg,
            ["medidor", "MEDIDOR", "No. De medidor", "No de medidor"]
        )

        servicio_col = first_existing_column(
            parsed_sg,
            ["no_servicio", "servicio", "No. Servicio"]
        )

        consumo_col = first_existing_column(
            parsed_sg,
            ["kwh_total", "consumo_kwh", "Consumo anual (kWh)"]
        )

        demanda_contratada_col = first_existing_column(
            parsed_sg,
            [
                "demanda_contratada_kw",
                "Demanda Contratada (kW)",
                "Demanda Contratada (kW)_num"
            ]
        )

        demanda_real_col = first_existing_column(
            parsed_sg,
            [
                "demanda_real_kw",
                "demanda_real_anual_kw",
                "demanda_maxima_kw",
                "Demanda real (kW)",
                "Demanda real kW"
            ]
        )

        # ------------------------------------------------------------
        # Identificar Servicios Generales
        # ------------------------------------------------------------

        search_cols = [
            col for col in [cliente_col, subgroup_col]
            if col is not None
        ]

        if not search_cols or cc_col is None:
            st.warning(
                "No encontré columnas suficientes para identificar Servicios Generales."
            )

        else:
            sg_mask = False

            for col in search_cols:
                sg_mask = sg_mask | (
                    parsed_sg[col]
                    .astype(str)
                    .str.upper()
                    .str.contains(
                        "SERVICIOS GENERALES|PARKS|MANTENIMIENTO|AREAS COMUNES|ÁREAS COMUNES|ADMINISTRACION|ADMINISTRACIÓN",
                        na=False,
                        regex=True
                    )
                )

            sg_df = parsed_sg[sg_mask].copy()

            if sg_df.empty:
                st.info(
                    "No encontré registros asociados a Servicios Generales en el parser."
                )

            else:

                # ------------------------------------------------------------
                # Normalizar columnas numéricas
                # ------------------------------------------------------------

                if consumo_col:
                    sg_df["_consumo_kwh"] = clean_number_series(sg_df[consumo_col])
                else:
                    sg_df["_consumo_kwh"] = np.nan

                if demanda_contratada_col:
                    sg_df["_demanda_contratada_kw"] = clean_number_series(
                        sg_df[demanda_contratada_col]
                    )
                else:
                    sg_df["_demanda_contratada_kw"] = np.nan

                if demanda_real_col:
                    sg_df["_demanda_real_kw"] = clean_number_series(
                        sg_df[demanda_real_col]
                    )
                else:
                    sg_df["_demanda_real_kw"] = np.nan

                # ------------------------------------------------------------
                # Tabla resumen por centro comercial
                # ------------------------------------------------------------

                rows = []

                for _, mall in cc_master_df.iterrows():

                    nombre_cc = limpiar_nombre_cc(mall.get("Nombre Comercial", ""))
                    tipo_cc = mall.get("Tipo de Mall", "")
                    zona_nrel = mall.get("zona_nrel", mall.get("Zona NREL", ""))

                    zona_upper = str(zona_nrel).upper()

                    if "HOT" in zona_upper:
                        clima = "Cálido"
                    elif "MIXED" in zona_upper:
                        clima = "Templado"
                    elif "COLD" in zona_upper:
                        clima = "Frío"
                    else:
                        clima = "Sin clasificar"

                    abr = mall.get("Área Bruta Rentable (m²)", np.nan)
                    abr = pd.to_numeric(
                        str(abr).replace(",", ""),
                        errors="coerce"
                    )

                    nombre_cc_key = normalizar_texto_simple(nombre_cc)

                    cc_sg = sg_df[
                        sg_df[cc_col]
                        .astype(str)
                        .apply(
                            lambda x: (
                                nombre_cc_key in normalizar_texto_simple(x)
                                or normalizar_texto_simple(x) in nombre_cc_key
                            )
                        )
                    ].copy()

                
                    if cc_sg.empty:
                        continue

                    medidores_sg = (
                        cc_sg[medidor_col].nunique()
                        if medidor_col
                        else len(cc_sg)
                    )

                    demanda_contratada = cc_sg["_demanda_contratada_kw"].sum()
                    demanda_real = cc_sg["_demanda_real_kw"].sum()
                    consumo_anual = cc_sg["_consumo_kwh"].sum()

                    factor_carga = np.nan

                    if pd.notna(demanda_real) and demanda_real > 0:
                        factor_carga = consumo_anual / (demanda_real * 8760)

                    densidad_demanda = np.nan
                    densidad_consumo = np.nan

                    if pd.notna(abr) and abr > 0:
                        densidad_demanda = demanda_real / abr
                        densidad_consumo = consumo_anual / abr

                    rows.append({
                        "Centro Comercial": nombre_cc,
                        "Tipo de CC": tipo_cc,
                        "Clima": clima,
                        "Medidores SG": medidores_sg,
                        "Demanda contratada (kW)": demanda_contratada,
                        "Demanda real (kW)": demanda_real,
                        "Factor de carga": factor_carga,
                        "Densidad demanda (kW/m² ABR)": densidad_demanda,
                        "Consumo anual (kWh)": consumo_anual,
                        "Densidad consumo (kWh/m² ABR)": densidad_consumo
                    })

                sg_resumen_df = pd.DataFrame(rows)

                if sg_resumen_df.empty:
                    st.info(
                        "No se pudo construir la tabla resumen de Servicios Generales."
                    )

                else:
                    sg_resumen_display = sg_resumen_df.copy()

                    for col in [
                        "Demanda contratada (kW)",
                        "Demanda real (kW)",
                        "Densidad demanda (kW/m² ABR)",
                        "Consumo anual (kWh)",
                        "Densidad consumo (kWh/m² ABR)"
                    ]:
                        sg_resumen_display[col] = sg_resumen_display[col].round(2)

                    sg_resumen_display["Factor de carga"] = (
                        sg_resumen_display["Factor de carga"] * 100
                    ).round(1).map(lambda x: f"{x}%" if pd.notna(x) else "")

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
                        sg_resumen_display["Tipo de CC"]
                        .map(orden_tipo_cc)
                    )

                    sg_resumen_display = (
                        sg_resumen_display
                        .sort_values(
                            [
                                "_orden_clima",
                                "_orden_tipo",
                                "Densidad demanda (kW/m² ABR)",
                                "Centro Comercial"
                            ],
                            ascending=[
                                True,   # clima
                                True,   # tipo de CC
                                False,  # mayor densidad primero
                                True    # nombre
                            ]
                        )
                        .drop(
                            columns=[
                                "_orden_clima",
                                "_orden_tipo"
                            ]
                        )
                    )

                    st.dataframe(
                        sg_resumen_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        """
**Nota:** Las densidades de demanda y consumo de Servicios Generales se calculan utilizando el Área Bruta Rentable (ABR) del centro comercial como variable de normalización. Aunque los servicios generales suministran principalmente áreas comunes, estacionamientos, pasillos, vestíbulos, elevadores y otros espacios no rentables, dichos servicios son indispensables para la operación y comercialización de las áreas rentables. Por esta razón, el ABR se considera una referencia adecuada para comparar la intensidad energética de Servicios Generales entre distintos centros comerciales.
"""
                    )

                # ------------------------------------------------------------
                # Gráfica de densidad de demanda por medidor
                # ------------------------------------------------------------

                st.markdown(
                    "### Densidad de demanda de Servicios Generales por medidor"
                )

                plot_rows = []

                for _, row in sg_df.iterrows():

                    centro = row.get(cc_col, "")
                    medidor = row.get(medidor_col, "") if medidor_col else ""

                    demanda_real = row.get("_demanda_real_kw", np.nan)

                    mall_match = cc_master_df[
                        cc_master_df["Nombre Comercial"]
                        .astype(str)
                        .str.upper()
                        .apply(
                            lambda x: x in str(centro).upper()
                            or str(centro).upper() in x
                        )
                    ]

                    if mall_match.empty:
                        continue

                    abr = mall_match.iloc[0].get("Área Bruta Rentable (m²)", np.nan)
                    abr = pd.to_numeric(
                        str(abr).replace(",", ""),
                        errors="coerce"
                    )

                    if pd.isna(demanda_real) or pd.isna(abr) or abr <= 0:
                        continue

                    densidad_w_m2 = demanda_real / abr * 1000

                    plot_rows.append({
                        "Etiqueta": f"{centro} ({medidor})",
                        "Centro Comercial": centro,
                        "Medidor": medidor,
                        "Densidad demanda (W/m² ABR)": densidad_w_m2
                    })

                sg_plot_df = pd.DataFrame(plot_rows)

                if sg_plot_df.empty:
                    st.info(
                        "No hay información suficiente para graficar densidad de demanda por medidor."
                    )

                else:
                    promedio = sg_plot_df["Densidad demanda (W/m² ABR)"].mean()
                    desv = sg_plot_df["Densidad demanda (W/m² ABR)"].std()

                    sg_plot_df = sg_plot_df.sort_values(
                        "Densidad demanda (W/m² ABR)",
                        ascending=False
                    ).reset_index(drop=True)

                    fig, ax = plt.subplots(figsize=(14, 5))

                    ax.scatter(
                        sg_plot_df["Etiqueta"],
                        sg_plot_df["Densidad demanda (W/m² ABR)"],
                        label="Densidad de demanda"
                    )

                    ax.axhline(
                        promedio,
                        label="Promedio"
                    )

                    ax.axhline(
                        promedio + desv,
                        label="+1 desv est"
                    )

                    ax.axhline(
                        max(promedio - desv, 0),
                        label="-1 desv est"
                    )

                    ax.set_ylabel("Densidad de demanda (W/m² ABR)")
                    ax.set_xlabel("Centro comercial (medidor)")
                    ax.set_title(
                        "Densidad de demanda de Servicios Generales"
                    )

                    ax.tick_params(
                        axis="x",
                        rotation=60,
                        labelsize=8
                    )

                    ax.legend(
                        loc="upper center",
                        bbox_to_anchor=(0.5, -0.35),
                        ncol=4,
                        fontsize=8
                    )

                    ax.grid(True, axis="y", alpha=0.3)

                    st.pyplot(fig)



with tab_anexo:

    st.markdown(
        '<div class="section-title">Anexo metodológico</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Metodología de estimación de demanda real anual")

    metodologia_df = pd.DataFrame({
        "Tarifa": ["GDMTH", "GDMTO", "GDBT", "PDBT"],
        "Criterio de demanda real anual": [
            "Máximo anual de KWMax",
            "Máximo anual de kW Totales",
            "Máximo anual de kW Totales",
            "Máximo anual estimado con perfil horario comercial"
        ],
        "Tipo de dato": [
            "Medido en recibo",
            "Medido en recibo",
            "Medido en recibo",
            "Estimado"
        ]
    })

    st.dataframe(metodologia_df, use_container_width=True)

    st.markdown("### Asignación de perfiles por giro comercial")

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


    st.markdown("""
    ### Uso mensual de perfiles NREL

    La estimación de demanda para usuarios PDBT utilizará perfiles horarios
    de NREL/DOE diferenciados por:

    - Zona climática
    - Giro""")

    st.markdown("### Perfil horario NREL utilizado para estimación PDBT")

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

    st.markdown("### Mapeo de centros comerciales a zona climática NREL")

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

    st.markdown("### Archivos de perfiles NREL disponibles")

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

    st.markdown("### Vista preliminar de asignación de perfiles por local")

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

    st.markdown("### Verificación de archivos NREL requeridos")

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

    st.markdown("### Referencias")

    st.markdown("""
**Fuente base del perfil:**  
NREL / DOE — End-Use Load Profiles for the U.S. Building Stock.  
Dataset público de perfiles de carga a resolución de 15 minutos para edificios comerciales y residenciales, desarrollado con modelos ResStock/ComStock calibrados con datos medidos.

**Referencia:**  
https://data.openei.org/submissions/4520

**Nota metodológica:**  
El archivo `perfil_pdbt_retail_nrel.csv` se usa como perfil horario comercial normalizado para estimar demanda máxima en usuarios PDBT. En esta versión puede contener un perfil temporal; cuando se sustituya por un perfil descargado de NREL/DOE, el código no necesita cambiar.
""")

