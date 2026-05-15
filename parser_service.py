from pathlib import Path
import subprocess
import sys
import time
import shutil


PROJECT_DIR = Path(__file__).resolve().parent
SCANNER_PATH = PROJECT_DIR / "scanner.py"
OUTPUT_DIR = PROJECT_DIR / "output"
TMP_SINGLE_MALL_ROOT = PROJECT_DIR / "_tmp_single_mall_root"

BILLS_FOLDER_NAME = "7. Recibos CFE"


def prepare_scanner_root(root_folder: str) -> tuple[Path, dict]:
    """
    Prepares the root folder expected by scanner.py.

    scanner.py expects a root folder that contains mall folders:

        ROOT/
            16. V1_Plaza Central/
                7. Recibos CFE/
                    ...

    But users may select either:

    1. A general root folder:
        /Users/.../Allux_Fraternity

    2. A single mall folder:
        /Users/.../Allux_Fraternity/16. V1_Plaza Central

    If the selected folder contains '7. Recibos CFE', this function creates
    a temporary wrapper root containing a symlink to the selected mall folder.
    """

    selected_root = Path(root_folder).expanduser().resolve()

    if not selected_root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {selected_root}")

    if not selected_root.is_dir():
        raise NotADirectoryError(f"Root path is not a folder: {selected_root}")

    bills_folder = selected_root / BILLS_FOLDER_NAME

    # Case 1: user selected a single mall folder
    if bills_folder.exists() and bills_folder.is_dir():
        if TMP_SINGLE_MALL_ROOT.exists():
            shutil.rmtree(TMP_SINGLE_MALL_ROOT)

        TMP_SINGLE_MALL_ROOT.mkdir(parents=True, exist_ok=True)

        symlink_path = TMP_SINGLE_MALL_ROOT / selected_root.name

        symlink_path.symlink_to(selected_root, target_is_directory=True)

        info = {
            "input_mode": "single_mall_folder",
            "selected_root": str(selected_root),
            "scanner_root": str(TMP_SINGLE_MALL_ROOT),
            "temporary_symlink": str(symlink_path),
            "mall_folder": selected_root.name,
            "note": (
                "The selected folder looks like a single mall folder because it "
                f"contains '{BILLS_FOLDER_NAME}'. A temporary wrapper root was created."
            ),
        }

        return TMP_SINGLE_MALL_ROOT, info

    # Case 2: user selected a general root folder
    info = {
        "input_mode": "multi_mall_root",
        "selected_root": str(selected_root),
        "scanner_root": str(selected_root),
        "temporary_symlink": None,
        "mall_folder": None,
        "note": (
            "The selected folder is treated as a general root containing one or more mall folders."
        ),
    }

    return selected_root, info


def scan_allux_folder(
    root_folder: str,
    limit: int | None = None,
    pages: int = 2,
    fresh: bool = False,
    resume: bool = False,
    write_every: int = 25,
    sleep_ms: int = 0,
):
    """
    Runs scanner.py from Python and returns a summary dictionary.

    This wrapper does not modify scanner.py. It also supports two user modes:

    - root_folder is the general Allux_Fraternity folder
    - root_folder is a single mall folder containing '7. Recibos CFE'
    """

    scanner_root, root_info = prepare_scanner_root(root_folder)

    if not SCANNER_PATH.exists():
        raise FileNotFoundError(f"scanner.py not found at: {SCANNER_PATH}")

    cmd = [
        sys.executable,
        str(SCANNER_PATH),
        "--root",
        str(scanner_root),
        "--pages",
        str(pages),
        "--write-every",
        str(write_every),
        "--sleep-ms",
        str(sleep_ms),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    if fresh:
        cmd.append("--fresh")

    if resume:
        cmd.append("--resume")

    start = time.time()

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_DIR),
        capture_output=True,
        text=True,
    )

    elapsed = time.time() - start

    summary = {
        "status": "success" if result.returncode == 0 else "error",
        "returncode": result.returncode,
        "elapsed_seconds": round(elapsed, 2),
        "root_info": root_info,
        "root": str(scanner_root),
        "selected_root": root_info.get("selected_root"),
        "command": " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "outputs": {
            "index_csv": str(OUTPUT_DIR / "bills_index.csv"),
            "parsed_csv": str(OUTPUT_DIR / "bills_parsed_v2.csv"),
            "historico_csv": str(OUTPUT_DIR / "bills_historico_v2.csv"),
        },
    }

    if result.returncode != 0:
        raise RuntimeError(
            "scanner.py failed.\n\n"
            f"Root info:\n{root_info}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return summary