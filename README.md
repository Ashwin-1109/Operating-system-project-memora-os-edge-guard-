# MemoraOS EdgeGuard — Real-Time Edge AI Memory Management

**Academic Proof of Concept Demonstration**  
*Operating Systems & Edge AI Systems Architecture*

---

## 1. Project Overview

**MemoraOS EdgeGuard** is a lightweight, academic proof of concept demonstrating adaptive memory management, request admission control, and resource safety for resource-constrained Edge AI workloads.

Edge devices hosting local Large Language Models (LLMs) via runtimes like `llama.cpp` often encounter severe latency spikes, memory thrashing, or out-of-memory (OOM) aborts when subjected to concurrent request bursts. 

MemoraOS EdgeGuard addresses this by interposing a deterministic, mathematically verifiable resource management controller before the inference runtime, evaluating:
1. **PASMA+** (*Placement-Aware Safe Memory Admission*): Estimates memory requirements and validates admission against logical memory pool partitions.
2. **PULSE-Fit** (*Adaptive Placement Selector*): Dynamically chooses between First Fit, Best Fit, Worst Fit, and Next Fit placement strategies based on pool external fragmentation.
3. **SAFE-Guard** (*Resource Safety*): Implements Dijkstra’s classic **Banker’s Algorithm** across multiple resource dimensions (Worker Slots and Memory Units) to prevent unsafe states.
4. **MemoraBrain** (*System Telemetry Advisor*): Deterministic rule advisor that monitors physical host RAM and process RSS via `psutil`.
5. **MemoraExplain** (*Decision Audit Logger*): Transparently explains why requests are admitted, deferred, or rejected with mathematical justification.

The application allows a direct, reproducible comparison between:
- **MODE A: NORMAL SYSTEM (EDGEGUARD OFF)** — Standard FIFO queuing without resource-admission filtering.
- **MODE B: MEMORAOS EDGEGUARD (EDGEGUARD ON)** — Managed admission control, logical memory reservation, and deadlock avoidance.

---

## 2. Directory Structure

```text
memoraos-edgeguard/
│
├── app.py                  # Streamlit dashboard, real-time memory charts, side-by-side comparison
├── inference.py            # llama.cpp HTTP client, psutil telemetry sampling, benchmark runner
├── edgeguard.py            # PASMA+, PULSE-Fit, SAFE-Guard (Banker's Algorithm), MemoraBrain & MemoraExplain
├── requirements.txt        # Minimal dependencies (streamlit, requests, psutil, pandas, pytest)
├── tests.py                # Automated test suite (15 unit tests covering algorithms and safety)
├── start.bat               # 1-click Windows launcher for the Streamlit dashboard
├── run_llama_server.bat    # 1-click launcher for the local llama.cpp server on port 8080
├── setup_edge_runtime.py   # Downloader for llama-server.exe (23 MB) & SmolLM2-360M GGUF (258 MB)
├── bin/                    # Local llama-server.exe and binary dependencies
├── models/                 # Local GGUF models (SmolLM2-360M-Instruct-Q4_K_M.gguf)
└── README.md               # Setup, execution, and scientific evaluation documentation
```

---

## 3. Prerequisites & Installation (Windows)

### Step 1: Install Python
Ensure Python 3.11 or 3.12 is installed and added to your system `PATH`.
Verify in your terminal:
```powershell
python --version
pip --version
```

### Step 2: Install Python Dependencies
Install required packages using:
```powershell
pip install -r requirements.txt
```

---

## 4. Setting Up the Local Edge AI Model (llama.cpp)

MemoraOS EdgeGuard is built to evaluate a **genuine local model** executing on CPU or GPU.

### Step 1: Download `llama.cpp` Server Executable
Download the official prebuilt Windows binary (`llama-server.exe` or `llama-bXXXX-bin-win-x64.zip`) from:
- [llama.cpp Official GitHub Releases](https://github.com/ggerganov/llama.cpp/releases)

### Step 2: Download a Lightweight GGUF Model
For laptop or edge CPU execution, select a small instruction-tuned model (~100 MB to 600 MB):
- **SmolLM-135M-Instruct** (Q4_K_M, ~100 MB)
- **Qwen2.5-0.5B-Instruct** (Q4_K_M, ~390 MB)
- **Llama-3.2-1B-Instruct** (Q4_K_M, ~700 MB)

Place the `.gguf` file in any folder (e.g. `D:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`).

### Step 3: Launch the `llama.cpp` Server
Run the documented CLI command in a separate terminal:
```cmd
llama-server.exe -m "D:\models\qwen2.5-0.5b-instruct-q4_k_m.gguf" --port 8080 -c 2048 -np 4
```
Key parameters:
- `-m`: Path to the local GGUF model file.
- `--port 8080`: HTTP port for OpenAI-compatible `/v1/chat/completions` and `/health`.
- `-c 2048`: Context window token size.
- `-np 4`: Number of concurrent inference execution slots.

---

## 5. Running the Application

### Option A: 1-Click Launch (`start.bat`)
Double-click `start.bat` in File Explorer or run in terminal:
```cmd
start.bat
```
*This automatically validates dependencies, launches the Streamlit server, and opens your default browser at `http://localhost:8501`.*

### Option B: Manual Command
```powershell
streamlit run app.py
```

---

## 6. Running the Automated Tests

To run the complete test suite verifying memory allocation strategies, coalescing, Banker's safe/unsafe state handling, and metrics:

```powershell
pytest tests.py -v
```

All 15 unit tests execute purely in-memory and do not require external server connectivity or model downloads.

---

## 7. Operating Systems Concepts Implemented

1. **Dijkstra's Banker's Algorithm (SAFE-Guard)**:
   - Resource vectors: `Available = [Worker Slots, Memory Units (MiB)]`
   - Matrices: `Max[P_i]`, `Allocation[P_i]`, `Need[P_i] = Max[P_i] - Allocation[P_i]`
   - Tests provisional states for a valid safe execution sequence before admitting requests. Unsafe states are deferred to prevent deadlock/starvation.

2. **Dynamic Contiguous Memory Placement (PULSE-Fit)**:
   - **First Fit**: Scans for first suitable free partition.
   - **Best Fit**: Selects smallest partition $\ge$ request size to minimize remnants.
   - **Worst Fit**: Allocates largest block to maximize remnant size.
   - **Next Fit**: Circular allocation starting from the last allocation pointer.
   - **Coalescing**: Automatic merging of adjacent free blocks upon workload release.

3. **External Fragmentation Measurement**:
   $$\text{External Fragmentation (EF)} = 1.0 - \left( \frac{\text{Largest Free Block}}{\text{Total Free Memory}} \right)$$

4. **Admission Control vs. Unmanaged Queuing**:
   - Demonstrates that under heavy concurrent stress, admitting requests only when safe prevents memory thrashing and unbounded queue delays.
