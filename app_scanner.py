from pathlib import Path
import streamlit as st
import pandas as pd

from parser_service import scan_allux_folder


st.set_page_config(
    page_title="Allux Electricity Bills Scanner",
    layout="wide"
)

st.title("Allux Electricity Bills Scanner")
st.caption("Interfaz local para ejecutar el parser de recibos CFE / Allux-Fraternity")


# ------------------------------------------------------------
# Helper: robust file preview
# ------------------------------------------------------------

def read_text_fallback(path, max_lines=30):
    """
    Reads the first lines of a file as text when pandas cannot parse it.
    """
    p = Path(path)

    for encoding in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            with open(p, "r", encoding=encoding, errors="replace") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip("\n"))
            return "\n".join(lines)
        except Exception:
            pass

    return "No pude leer el archivo como texto."


def safe_preview_csv(path, title, nrows=50):
    """
    Safely previews CSV/TSV files in Streamlit.

    Strategy:
    1. Try comma-separated CSV.
    2. Try semicolon-separated CSV.
    3. Try tab-separated TSV.
    4. Try automatic separator detection.
    5. If all pandas attempts fail, show raw text preview.
    """

    st.markdown(f"### Vista previa: {title}")

    p = Path(path)

    if not p.exists():
        st.warning(f"No existe el archivo: {p}")
        return

    if p.stat().st_size == 0:
        st.warning(f"El archivo existe pero está vacío: {p}")
        return

    read_attempts = [
        {
            "label": "CSV utf-8 comma",
            "kwargs": {
                "sep": ",",
                "encoding": "utf-8",
                "engine": "python",
                "on_bad_lines": "skip",
                "dtype": str,
            },
        },
        {
            "label": "CSV utf-8-sig comma",
            "kwargs": {
                "sep": ",",
                "encoding": "utf-8-sig",
                "engine": "python",
                "on_bad_lines": "skip",
                "dtype": str,
            },
        },
        {
            "label": "CSV latin1 comma",
            "kwargs": {
                "sep": ",",
                "encoding": "latin1",
                "engine": "python",
                "on_bad_lines": "skip",
                "dtype": str,
            },
        },
        {
            "label": "CSV utf-8 semicolon",
            "kwargs": {
                "sep": ";",
                "encoding": "utf-8",
                "engine": "python",
                "on_bad_lines": "skip",
                "dtype": str,
            },
        },
        {
            "label": "TSV utf-8 tab",
            "kwargs": {
                "sep": "\t",
                "encoding": "utf-8",
                "engine": "python",
                "on_bad_lines": "skip",
                "dtype": str,
            },
        },
        {
            "label": "Auto separator utf-8",
            "kwargs": {
                "sep": None,
                "encoding": "utf-8",
                "engine": "python",
                "on_bad_lines": "skip",
                "dtype": str,
            },
        },
        {
            "label": "Auto separator latin1",
            "kwargs": {
                "sep": None,
                "encoding": "latin1",
                "engine": "python",
                "on_bad_lines": "skip",
                "dtype": str,
            },
        },
    ]

    last_error = None

    for attempt in read_attempts:
        try:
            df = pd.read_csv(
                p,
                nrows=nrows,
                **attempt["kwargs"]
            )

            if df.empty:
                continue

            st.caption(f"Leído con: {attempt['label']}")
            st.dataframe(df, use_container_width=True)
            return

        except Exception as e:
            last_error = e

    st.warning(
        f"No pude interpretar {title} como tabla CSV/TSV, "
        "pero el archivo sí existe. Muestro una vista previa en texto."
    )

    with st.expander(f"Vista previa en texto: {title}", expanded=True):
        st.code(read_text_fallback(p), language="text")

    with st.expander(f"Detalles técnicos del último error leyendo {title}"):
        st.exception(last_error)


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------

default_root = "/Users/jjesusricojericomelgoiza/Allux_Fraternity"

root_folder = st.text_input(
    "Carpeta raíz o centro comercial",
    value=default_root,
    help=(
        "Puedes poner la carpeta general Allux_Fraternity o directamente "
        "una carpeta de centro comercial que contenga '7. Recibos CFE'."
    )
)

st.info(
    "Puedes usar dos modos: "
    "\n\n"
    "1. Carpeta general con varios centros comerciales, por ejemplo "
    "`/Users/.../Allux_Fraternity`."
    "\n\n"
    "2. Carpeta de un solo centro comercial, por ejemplo "
    "`/Users/.../Allux_Fraternity/16. V1_Plaza Central`. "
    "Si la app detecta `7. Recibos CFE`, crea automáticamente un root temporal."
)

st.markdown("### Opciones de corrida")

col1, col2, col3, col4 = st.columns(4)

with col1:
    pages = st.number_input(
        "Páginas por PDF",
        min_value=1,
        max_value=10,
        value=2,
        step=1
    )

with col2:
    limit_enabled = st.checkbox("Usar límite", value=True)

    limit = st.number_input(
        "Límite de PDFs",
        min_value=1,
        max_value=100000,
        value=20,
        step=1,
        disabled=not limit_enabled
    )

with col3:
    fresh = st.checkbox(
        "Fresh run",
        value=True,
        help="Borra salidas previas y empieza una corrida nueva."
    )

with col4:
    resume = st.checkbox(
        "Resume",
        value=False,
        help="Intenta continuar usando progreso previo."
    )

run_button = st.button("Run scanner", type="primary")

st.divider()


# ------------------------------------------------------------
# Run scanner
# ------------------------------------------------------------

if run_button:
    try:
        with st.spinner("Procesando recibos..."):
            summary = scan_allux_folder(
                root_folder=root_folder,
                limit=int(limit) if limit_enabled else None,
                pages=int(pages),
                fresh=bool(fresh),
                resume=bool(resume),
            )

        st.success("Scanner terminado correctamente.")

        st.markdown("### Resumen de corrida")

        root_info = summary.get("root_info", {})

        mode_label = root_info.get("input_mode", "unknown")
        selected_root = root_info.get("selected_root", "")
        scanner_root = root_info.get("scanner_root", "")

        col_a, col_b, col_c, col_d = st.columns(4)

        col_a.metric("Estado", summary.get("status", "unknown"))
        col_b.metric("Tiempo total", f'{summary.get("elapsed_seconds", "NA")} s')
        col_c.metric("Código de salida", summary.get("returncode", "NA"))
        col_d.metric("Modo", mode_label)

        st.markdown("### Modo de lectura")

        if mode_label == "single_mall_folder":
            st.success(
                "La app detectó que seleccionaste un solo centro comercial. "
                "Se creó automáticamente un root temporal para el scanner."
            )
        elif mode_label == "multi_mall_root":
            st.info(
                "La app está tratando la carpeta seleccionada como raíz general "
                "que contiene uno o más centros comerciales."
            )
        else:
            st.warning("No pude identificar claramente el modo de lectura.")

        with st.expander("Ver detalle de rutas"):
            st.write("**Ruta seleccionada:**")
            st.code(selected_root, language="text")

            st.write("**Root usado por scanner.py:**")
            st.code(scanner_root, language="text")

            temporary_symlink = root_info.get("temporary_symlink")
            if temporary_symlink:
                st.write("**Enlace temporal creado:**")
                st.code(temporary_symlink, language="text")

            note = root_info.get("note")
            if note:
                st.write("**Nota:**")
                st.write(note)

        with st.expander("Ver comando ejecutado"):
            st.code(summary.get("command", ""), language="bash")

        with st.expander("Ver salida del scanner"):
            stdout_text = summary.get("stdout", "")
            if stdout_text.strip():
                st.text(stdout_text)
            else:
                st.info("El scanner no produjo salida estándar visible.")

        stderr_text = summary.get("stderr", "")
        if stderr_text.strip():
            with st.expander("Ver mensajes de error / advertencias"):
                st.text(stderr_text)

        outputs = summary.get("outputs", {})

        index_csv = outputs.get("index_csv")
        parsed_csv = outputs.get("parsed_csv")
        historico_csv = outputs.get("historico_csv")

        st.markdown("### Archivos generados")

        for label, path in [
            ("Índice de PDFs", index_csv),
            ("Recibos parseados", parsed_csv),
            ("Histórico de consumo", historico_csv),
        ]:
            if path and Path(path).exists():
                st.write(f"**{label}:** `{path}`")

                with open(path, "rb") as f:
                    st.download_button(
                        label=f"Descargar {Path(path).name}",
                        data=f,
                        file_name=Path(path).name,
                        mime="text/csv"
                    )
            else:
                st.warning(f"No se encontró: {label}")

        if index_csv:
            safe_preview_csv(index_csv, "índice de PDFs")

        if parsed_csv:
            safe_preview_csv(parsed_csv, "recibos parseados")

        if historico_csv:
            safe_preview_csv(historico_csv, "histórico de consumo")

    except Exception as e:
        st.error("Ocurrió un error al ejecutar el scanner.")
        st.exception(e)