#  LightGBM CPU Benchmark v0.1

A lightweight, high-precision CPU performance benchmarking tool that evaluates system hardware using real-world machine learning workloads.

## 📌 Overview

Unlike traditional synthetic arithmetic loops, this tool stress-tests CPU microarchitecture, memory bandwidth, cache efficiency, and multi-thread scheduling through **LightGBM** gradient-boosted decision tree training. It profiles host hardware, builds a deterministic matrix dataset, performs cold-start warm-ups, and computes standardized execution metrics across multiple test passes.

##  Key Features

* **Real-World ML Workload:** Measures CPU compute capacity using intensive tree-building algorithms and matrix operations.
* **Deterministic Dataset:** Uses fixed-seed NumPy generation to construct a consistent 100,000 row × 32 feature synthetic matrix for accurate cross-system hardware comparisons.
* **Cold-Start Elimination:** Executes an unmeasured initial warm-up run to bypass library initialization and thread-pool setup latency.
* **Robust Scoring:** Evaluates performance using median runtime, peak runtime, row throughput, and a normalized performance index.

## ⚙️ Benchmark Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Data Matrix** | 100,000 × 32 | Single-precision (`float32`) features with linear target classification |
| **Boosting Rounds** | 200 | Total trees constructed per test pass |
| **Tree Architecture** | 63 Leaves | `max_depth = -1`, `learning_rate = 0.08`, sub-sampling enabled |
| **Execution Protocol** | 1 Warm-up + 3 Test Runs | Results derived from median execution latency |
| **Threading** | Auto-detected | Fully utilizes all available logical CPU cores |

##  Quick Start >_

### 1. Install Dependencies
```bash
pip install numpy lightgbm
