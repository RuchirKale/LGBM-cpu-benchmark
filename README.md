# ⚡ Lightweight CPU Benchmark

A high-efficiency, end-to-end CPU performance benchmarking tool powered by machine learning workload simulation.

## 🚀 Key Features

* **System Profiling:** Automatically detects and logs hardware configuration, core count, and system architecture.
* **Synthetic Dataset Generation:** Dynamically generates structured data to stress test memory bandwidth and single/multi-thread performance.
* **ML Workload Stress Test:** Trains a LightGBM model to evaluate real-world CPU performance during intensive matrix operations and decision tree training.

## 🛠️ Quick Start

```bash
# Clone the repository
git clone [https://github.com/your-username/cpu-benchmark.git](https://github.com/your-username/cpu-benchmark.git)
cd cpu-benchmark

# Install dependencies
pip install -r requirements.txt

# Run the benchmark
python benchmark.py
