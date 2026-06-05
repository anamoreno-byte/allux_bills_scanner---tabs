# parser/llamaextract_batch.py
from __future__ import annotations

import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from parser.llamaextract_validate import clean_llamaextract_row


# ------------------------------------------------------------
# Path metadata helpers
# ------------------------------------------------------------
def infer_mall_folder(pdf_path: Path) -> str | None:
    """
    Infer mall folder from paths like:
    .../08. V1_Espacio Aguascalientes/7. Recibos CFE/Soriana/file.pdf
    """
    for part in pdf_path.parts:
        if ". V1_" in part or part.startswith("V1_"):
            return part
    return None


def infer_recibos_subgroup(pdf_path: Path) -> str | None:
    """
    Infer subgroup inside 'Recibos CFE'.

    Examples:
    .../7. Recibos CFE/Homay.pdf          -> None
    .../7. Recibos CFE/Soriana/file.pdf   -> Soriana
    """
    parts = list(pdf_path.parts)

    for i, part in enumerate(parts):
        if "Recibos CFE" in part:
            if i + 1 < len(parts) - 1:
                return parts[i + 1]
            return None

    return None


def infer_recibos_year(pdf_path: Path) -> str | None:
    """
    Try to infer a year folder if present in the path.
    """
    for part in pdf_path.parts:
        if part.isdigit() and len(part) == 4:
            return part
    return None


# ------------------------------------------------------------
# Single PDF worker
# ------------------------------------------------------------
def process_one_pdf(
    pdf_path_str: str,
    raw_json_dir_str: str,
    schema_mode: str = "general",
) -> dict:
    """
    Process one PDF with LlamaExtract.

    Important:
    This function receives strings instead of Path objects because it may
    run inside a separate process.
    """
    start = time.time()

    pdf_path = Path(pdf_path_str)
    raw_json_dir = Path(raw_json_dir_str)

    try:
        # Import inside the worker process.
        # This avoids sharing LlamaExtract client/event-loop state
        # across processes or Streamlit reruns.
        from parser.llamaextract_client import extract_cfe_bill

        raw_result = extract_cfe_bill(pdf_path, schema_mode=schema_mode)

        raw_json_dir.mkdir(parents=True, exist_ok=True)
        raw_json_path = raw_json_dir / f"{pdf_path.stem}.json"

        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_result, f, ensure_ascii=False, indent=2)

        row = clean_llamaextract_row(raw_result)

        row["pdf_path"] = str(pdf_path)
        row["filename"] = pdf_path.name
        row["parent_folder"] = pdf_path.parent.name
        row["mall_folder"] = infer_mall_folder(pdf_path)
        row["recibos_subgroup"] = infer_recibos_subgroup(pdf_path)
        row["recibos_year"] = infer_recibos_year(pdf_path)

        row["parser_engine"] = "llamaextract"
        row["llamaextract_status"] = "ok"
        row["llamaextract_error"] = None
        row["processing_seconds"] = round(time.time() - start, 3)

        return row

    except Exception as e:
        return {
            "pdf_path": str(pdf_path),
            "filename": pdf_path.name,
            "parent_folder": pdf_path.parent.name,
            "mall_folder": infer_mall_folder(pdf_path),
            "recibos_subgroup": infer_recibos_subgroup(pdf_path),
            "recibos_year": infer_recibos_year(pdf_path),
            "parser_engine": "llamaextract",
            "llamaextract_status": "error",
            "llamaextract_error": str(e),
            "processing_seconds": round(time.time() - start, 3),
        }


# ------------------------------------------------------------
# CSV helpers
# ------------------------------------------------------------
def read_existing_csv_safely(csv_path: Path) -> pd.DataFrame:
    """
    Read an existing CSV if it exists and is not empty.
    Return an empty DataFrame otherwise.
    """
    if not csv_path.exists():
        return pd.DataFrame()

    if csv_path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ------------------------------------------------------------
# Main batch runner
# ------------------------------------------------------------
def run_llamaextract_batch(
    pdf_paths: list[Path],
    output_csv: Path,
    error_csv: Path,
    raw_json_dir: Path,
    max_workers: int = 1,
    resume: bool = True,
    mode: str = "sequential",
    schema_mode: str = "general",

) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run LlamaExtract over a list of PDFs.

    mode:
      - "sequential": safest mode. One PDF at a time.
      - "processes": parallel mode using ProcessPoolExecutor.
    """
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    error_csv.parent.mkdir(parents=True, exist_ok=True)
    raw_json_dir.mkdir(parents=True, exist_ok=True)

    old_rows = pd.DataFrame()
    done_paths: set[str] = set()

    if resume:
        old_rows = read_existing_csv_safely(output_csv)
        if not old_rows.empty and "pdf_path" in old_rows.columns:
            done_paths = set(old_rows["pdf_path"].dropna().astype(str))

    pending = [
        p for p in pdf_paths
        if str(p) not in done_paths
    ]
    print(
        f"[LlamaExtract Batch] mode={mode}, "
        f"max_workers={max_workers}, "
        f"pending={len(pending)}, "
        f"schema_mode={schema_mode}"
    )
    rows: list[dict] = []
    errors: list[dict] = []

    # --------------------------------------------------------
    # Sequential mode
    # --------------------------------------------------------
    if mode == "sequential" or max_workers <= 1:
        print("[LlamaExtract Batch] Using sequential mode")
        for pdf_path in pending:
            row = process_one_pdf(
                pdf_path_str=str(pdf_path),
                raw_json_dir_str=str(raw_json_dir),
                schema_mode=schema_mode,
            )

            if row.get("llamaextract_status") == "ok":
                rows.append(row)
            else:
                errors.append(row)

    # --------------------------------------------------------
    # Process-based parallel mode
    # --------------------------------------------------------
    elif mode == "processes":
        print("[LlamaExtract Batch] Using ProcessPoolExecutor")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_one_pdf,
                    str(pdf_path),
                    str(raw_json_dir),
                    schema_mode,
                ): pdf_path
                for pdf_path in pending
            }

            for future in as_completed(futures):
                row = future.result()

                if row.get("llamaextract_status") == "ok":
                    rows.append(row)
                else:
                    errors.append(row)

    else:
        raise ValueError(f"Unknown batch mode: {mode}")

    df_rows = pd.DataFrame(rows)
    df_errors = pd.DataFrame(errors)

    if resume and not old_rows.empty:
        df_rows = pd.concat([old_rows, df_rows], ignore_index=True)

    if not df_rows.empty:
        df_rows.to_csv(output_csv, index=False)

    if not df_errors.empty:
        df_errors.to_csv(error_csv, index=False)

    return df_rows, df_errors