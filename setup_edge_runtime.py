"""
Automated Downloader and Runtime Configurator for MemoraOS EdgeGuard
Downloads:
1. llama-server.exe prebuilt binary (23 MB from llama.cpp GitHub releases) into bin/
2. SmolLM2-360M-Instruct-Q4_K_M.gguf (258 MB from HuggingFace) into models/
"""

import os
import sys
import zipfile
import requests

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
BIN_DIR = os.path.join(WORKSPACE_DIR, "bin")
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")

LLAMA_ZIP_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11934/llama-b11934-bin-win-cpu-x64.zip"
MODEL_GGUF_URL = "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"

MODEL_FILE_PATH = os.path.join(MODELS_DIR, "SmolLM2-360M-Instruct-Q4_K_M.gguf")
SERVER_EXE_PATH = os.path.join(BIN_DIR, "llama-server.exe")


def download_file(url: str, dest_path: str, label: str):
    print(f"\n[+] Downloading {label}...")
    print(f"    From: {url}")
    print(f"    To:   {dest_path}")
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    response = requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    
    downloaded = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = (downloaded / total_size) * 100
                    mb_down = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    sys.stdout.write(f"\r    Progress: {pct:5.1f}% [{mb_down:6.1f} / {mb_total:6.1f} MB]")
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f"\r    Downloaded: {downloaded / (1024 * 1024):6.1f} MB")
                    sys.stdout.flush()
    print("\n    [OK] Download complete.")


def setup():
    os.makedirs(BIN_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    zip_path = os.path.join(BIN_DIR, "llama_runtime.zip")

    # 1. Extract or download llama-server.exe
    if not os.path.exists(SERVER_EXE_PATH):
        if not os.path.exists(zip_path):
            download_file(LLAMA_ZIP_URL, zip_path, "llama.cpp Windows CPU runtime (23 MB)")
        print("\n[+] Extracting llama-server runtime binaries...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(BIN_DIR)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("    [OK] Extracted llama-server.exe and dependencies into bin/")
    else:
        print(f"[OK] Found existing {SERVER_EXE_PATH}")

    # 2. Download SmolLM2 258MB model if missing
    if not os.path.exists(MODEL_FILE_PATH):
        download_file(MODEL_GGUF_URL, MODEL_FILE_PATH, "SmolLM2-360M-Instruct Q4_K_M Base Model (258 MB)")
    else:
        print(f"[OK] Found existing {MODEL_FILE_PATH}")

    print("\n" + "=" * 60)
    print("ALL RUNTIME ASSETS ARE READY!")
    print(f"Server Binary: {SERVER_EXE_PATH}")
    print(f"Model GGUF:    {MODEL_FILE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    setup()
