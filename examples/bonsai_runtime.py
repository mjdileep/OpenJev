"""Pinned Prism CUDA runtime for the Bonsai Colab; no system packages are patched."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

PRISM_TAG = "prism-b10709-9a9394a"
PRISM_ARCHIVE = f"llama-{PRISM_TAG}-bin-linux-cuda-12.4-x64.tar.gz"
PRISM_SHA256 = "f542fdcc818562359e947db65e0b11c4658dd5ca3bd240490448252e817d8e7a"
PYTHON_REF = "3691546f1c9e0c1bf93323dff02230bd959cf562"  # llama-cpp-python 0.3.35
PYTHON_SHA256 = "519d48447c43e560a4a8a3eb94cafe851ed9303720c527838fe900edf5cd46f9"


def download(url: str, path: Path, sha256: str | None = None) -> Path:
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".partial")
        print(f"Downloading {path.name}...", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    if sha256:
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != sha256:
            raise RuntimeError(f"Checksum mismatch: remove {path} and retry.")
    return path


def patch_bindings(source: str) -> str:
    # These are the two struct-layout differences in Prism's pinned llama.h.
    # Both fields must be present before ctypes binds functions returning structs.
    # Never use this shim with a different runtime or Python-wrapper revision.
    fields = [
        (
            '        ("devices", ctypes.c_void_p),  # NOTE: unnused',
            '        ("dspark_head_source", ctypes.c_void_p),',
        ),
        (
            '        ("abort_callback", ggml_abort_callback),',
            '        ("path_kv_mean_center", ctypes.c_char_p),',
        ),
    ]
    for anchor, addition in fields:
        if source.count(anchor) != 1 or addition in source:
            raise RuntimeError("Unexpected Python bindings; refusing an unsafe ABI patch.")
        source = source.replace(anchor, addition + "\n" + anchor)
    return source


def prepare(directory: str | Path = "/content/openjev-bonsai-runtime") -> Path:
    """Download the release and select an isolated Python wrapper before importing it."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("This helper targets a Linux x86_64 Colab GPU runtime.")
    if "llama_cpp" in sys.modules:
        raise RuntimeError(
            "Restart the Colab session, then run installation before loading a model."
        )
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    archive = download(
        f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{PRISM_TAG}/{PRISM_ARCHIVE}",
        root / PRISM_ARCHIVE,
        PRISM_SHA256,
    )
    with tarfile.open(archive) as bundle:
        bundle.extractall(root / "native", filter="data")
    native = next((root / "native").rglob("libllama.so")).parent
    python_archive = download(
        f"https://github.com/abetlen/llama-cpp-python/archive/{PYTHON_REF}.tar.gz",
        root / "llama-cpp-python.tar.gz",
        PYTHON_SHA256,
    )
    with tarfile.open(python_archive) as bundle:
        bundle.extractall(root / "sources", filter="data")
    source_root = root / "sources" / f"llama-cpp-python-{PYTHON_REF}"
    python_root = root / "python"
    python_root.mkdir(exist_ok=True)
    shutil.copytree(source_root / "llama_cpp", python_root / "llama_cpp", dirs_exist_ok=True)
    binding = python_root / "llama_cpp" / "llama_cpp.py"
    binding.write_text(patch_bindings(binding.read_text()))
    # Retain the wrapper's MIT license beside the private copy.
    shutil.copyfile(source_root / "LICENSE.md", python_root / "LICENSE.md")
    os.environ["LLAMA_CPP_LIB_PATH"] = str(native)
    sys.path.insert(0, str(python_root))
    import llama_cpp

    llama_cpp.llama_backend_init()
    info = llama_cpp.llama_print_system_info().decode()
    print(info, flush=True)
    if "CUDA" not in info or not llama_cpp.llama_supports_gpu_offload():
        raise RuntimeError(
            "Prism CUDA backend is unavailable. Check the GPU runtime and its driver."
        )
    print("Ready:", PRISM_TAG, "| Python bindings:", llama_cpp.__version__, flush=True)
    return native
