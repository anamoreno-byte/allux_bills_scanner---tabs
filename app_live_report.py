from pathlib import Path
import pandas as pd
import streamlit as st


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "output"
DATA_DIR = PROJECT_DIR / "data"

DEFAULT_PARSED_CSV = OUTPUT_DIR / "bills_parsed_v2_RUN_9232.csv"
DEFAULT_HISTORICO_CSV = OUTPUT_DIR / "bills_historico_v2_RUN_33733.csv"
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


tab_resumen, tab_calidad, tab_general, tab_cc = st.tabs([
    "Resumen Ejecutivo",
    "Calidad de Datos",
    "Análisis General",
    "Centro Comercial"
])

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
    # Data quality
    # ============================================================
with tab_calidad:
    st.markdown('<div class="section-title">2. Calidad de datos</div>', unsafe_allow_html=True)

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

    if not quality_df.empty:
        st.dataframe(quality_df, use_container_width=True)
        chart_df = quality_df.set_index("campo")[["cobertura_%"]]
        st.bar_chart(chart_df)
    else:
        st.info("No hay columnas suficientes para calcular cobertura de campos.")

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
                "Centro Comercial": mall_name,
                "Locales ocupados": locales_ocupados,
                "Locales ocupados con recibo": locales_con_recibo,
                "Locales ocupados sin recibo": locales_sin_recibo,
                "Cobertura de muestra (%)": round(cobertura, 1),
            })

        coverage_by_mall = pd.DataFrame(coverage_rows)

        st.dataframe(
            coverage_by_mall.sort_values("Centro Comercial"),
            use_container_width=True
        )

    else:
        st.warning(
            "No encontré columnas suficientes para calcular cobertura por centro comercial."
        )

    st.markdown("### Recibos sin número de servicio pero con medidor")

    missing_service_with_meter = parsed[
        parsed["no_servicio"].isna() &
        parsed["medidor"].notna()
    ]

    st.write("Recibos sin número de servicio pero con medidor:", len(missing_service_with_meter))

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

    st.write("Servicios con más de un medidor:", len(servicios_con_varios_medidores))

    st.dataframe(servicios_con_varios_medidores, use_container_width=True)

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

        st.dataframe(mall_quality, use_container_width=True)
    else:
        st.warning("No encontré columna de centro comercial.")


with tab_general:
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

        selected_cc = st.selectbox(
            "Selecciona un centro comercial",
            options=sorted(filtered[mall_col].dropna().unique())
        )

        cc_parser = filtered[filtered[mall_col] == selected_cc].copy()

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

        locales_ocupados = len(general_ocupados)
        locales_con_recibo = int(general_ocupados["_tiene_recibo"].sum())
        locales_sin_recibo = int(locales_ocupados - locales_con_recibo)
        cobertura_pct = round((locales_con_recibo / locales_ocupados * 100), 1) if locales_ocupados else 0

        st.markdown("### Calidad de muestra")

        col_ocupacion, col_muestra = st.columns(2)

        with col_ocupacion:
            st.subheader("Ocupación del centro comercial")

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
            st.subheader("Muestra disponible")

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

        st.markdown("### Recibos encontrados en parser sin match en Datos Generales")

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

    else:
        st.warning("No encontré columna de centro comercial.")
