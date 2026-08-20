import os
import time
import statistics
import platform

import numpy as np
import lightgbm as lgb


# =========================
# LGBM CPU BENCHMARK v0.1
# =========================

ROWS = 100_000
FEATURES = 32
ROUNDS = 200
RUNS = 3

# Reference used for the prototype score.
REFERENCE_TIME = 30.0
REFERENCE_SCORE = 10_000


def create_dataset():
    print("Generating dataset...")

    rng = np.random.default_rng(42)

    X = rng.random(
        (ROWS, FEATURES),
        dtype=np.float32
    )

    weights = rng.normal(
        0,
        1,
        FEATURES
    ).astype(np.float32)

    logits = (
        X @ weights
        + rng.normal(0, 0.5, ROWS).astype(np.float32)
    )

    y = (
        logits > np.median(logits)
    ).astype(np.int8)

    return X, y


def benchmark(X, y, threads):
    dataset = lgb.Dataset(
        X,
        label=y,
        free_raw_data=False
    )

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,

        # CPU configuration
        "device_type": "cpu",
        "num_threads": threads,

        # Fixed benchmark configuration
        "learning_rate": 0.08,
        "num_leaves": 63,
        "max_depth": -1,

        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,

        "seed": 42,
        "feature_fraction_seed": 42,
        "bagging_seed": 42,
    }

    start = time.perf_counter()

    lgb.train(
        params,
        dataset,
        num_boost_round=ROUNDS
    )

    end = time.perf_counter()

    return end - start


def calculate_score(runtime):
    """
    Higher score = better CPU.

    Prototype formula:
        score = reference_score *
                reference_time / runtime
    """

    return (
        REFERENCE_SCORE
        * REFERENCE_TIME
        / runtime
    )


def rating(score):

    if score >= 210_000:
        return "Blazing Fast"

    elif score >= 170_000:
        return "Fast"

    elif score >= 100_500:
        return "Good "

    elif score >= 50_000:
        return "Good "

    else:
        return "Entry-level "


def main():

    threads = os.cpu_count() or 1

    print()
    print("=" * 50)
    print("          LGBM CPU BENCHMARK v0.1")
    print("=" * 50)

    print()
    print("CPU:")
    print(platform.processor())

    print(f"Logical threads: {threads}")
    print(f"Dataset:         {ROWS:,} rows × {FEATURES} features")
    print(f"Boosting rounds: {ROUNDS}")
    print(f"Measured runs:   {RUNS}")

    print()
    print("Generating benchmark workload...")

    X, y = create_dataset()

    # -------------------------
    # Warm-up
    # -------------------------

    print()
    print(" Warm-up run...")

    benchmark(
        X,
        y,
        threads
    )

    # -------------------------
    # Actual benchmark
    # -------------------------

    times = []

    print()
    print("Running benchmark...")

    for i in range(RUNS):

        print(
            f"Run {i + 1}/{RUNS}...",
            end=" ",
            flush=True
        )

        runtime = benchmark(
            X,
            y,
            threads
        )

        times.append(runtime)

        print(
            f"{runtime:.3f} seconds"
        )

    # -------------------------
    # Results
    # -------------------------

    median_time = statistics.median(times)

    best_time = min(times)

    throughput = ROWS / median_time

    score = calculate_score(
        median_time
    )

    print()
    print("=" * 50)
    print("                 RESULTS")
    print("=" * 50)

    print()
    print(
        f"⏱ Median runtime : {median_time:.3f} seconds"
    )

    print(
        f" Best runtime   : {best_time:.3f} seconds"
    )

    print(
        f" Throughput     : {throughput:,.0f} rows/sec"
    )

    print(
        f" CPU threads    : {threads}"
    )

    print()
    print(
        f" CPU SCORE      : {score:,.0f}"
    )

    print(
        f" Rating         : {rating(score)}"
    )

    print()
    print("=" * 50)


if __name__ == "__main__":
    main()