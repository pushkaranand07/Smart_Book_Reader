import os
import shutil
from pathlib import Path
import pytesseract


def detect_tesseract_path() -> str | None:
    """Detect and return the path to the Tesseract executable."""
    # 1. Check if tesseract is already in PATH
    which_path = shutil.which("tesseract")
    if which_path and os.path.exists(which_path):
        return which_path

    # 2. Check common Windows paths
    local_appdata = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")

    candidates = [
        os.path.join(local_appdata, r"Programs\Tesseract-OCR\tesseract.exe"),
        os.path.join(program_files, r"Tesseract-OCR\tesseract.exe"),
        os.path.join(program_files_x86, r"Tesseract-OCR\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract",
    ]

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate

    return None


def configure_tesseract() -> tuple[bool, str]:
    """Configure pytesseract with the discovered Tesseract path.

    Returns:
        (is_available, status_message)
    """
    cmd_path = detect_tesseract_path()
    if cmd_path:
        pytesseract.pytesseract.tesseract_cmd = cmd_path
        try:
            version = pytesseract.get_tesseract_version()
            return True, f"Tesseract OCR ready (v{version}) at {cmd_path}"
        except Exception as e:
            return False, f"Tesseract found at {cmd_path} but error initializing: {e}"
    else:
        return False, "Tesseract executable not found. Please verify Tesseract is installed."
