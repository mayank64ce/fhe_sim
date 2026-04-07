#!/usr/bin/env python3
"""
example_calibrate.py — End-to-end calibration example for fhe_sim.

This script demonstrates how to:
  1. Load timing measurements from a CSV file
  2. Calibrate ArchParam against those measurements
  3. Save the fitted parameters
  4. Evaluate prediction accuracy (MAPE)

Prerequisites:
  pip install scipy

Usage:
  python -m fhe_sim.example_calibrate
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fhe_sim import Simulator, ArchParam, CacheStyle
from fhe_sim.calibrate import (
    calibrate,
    load_entries_from_csv,
    save_calibration,
    load_calibration,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Adjust these to point at your own timing data and calibration programs.
CALIBRATION_DIR = os.path.join(
    os.path.dirname(__file__), "..", "tools", "calibration"
)
CSV_PATH = os.path.join(CALIBRATION_DIR, "timings_v111.csv")
OUTPUT_PATH = os.path.join(CALIBRATION_DIR, "fitted_arch_example.json")


def main() -> None:
    # ── 1. Load timing measurements ───────────────────────────────────────
    entries = load_entries_from_csv(CSV_PATH)
    print(f"Loaded {len(entries)} calibration entries from {CSV_PATH}\n")

    # ── 2. Define a starting ArchParam ────────────────────────────────────
    # Use CPU_ARCH as the starting point since these timings come from a CPU.
    # Algorithmic flags (karatsuba, key_compression, etc.) must match the
    # actual OpenFHE build and are NOT fitted — only numerical parameters
    # are optimised.
    init_arch = ArchParam(
        funits=4, sets=1,
        add_lat=1, mult_lat=4, ntt_lat=4, auto_lat=1,
        cache_style=CacheStyle.NONE,
        clock_freq_GHz=3.0,
        memory_bandwidth_GBps=50.0,
        per_rotate_overhead_s=0.04,
        per_mult_overhead_s=0.06,
        per_add_overhead_s=1e-6,
    )

    # ── 3. Run calibration ────────────────────────────────────────────────
    # All 9 numerical parameters are fitted here.  If you know your clock
    # frequency or memory bandwidth from hardware specs, pin them with
    # fix_clock_freq_GHz / fix_bandwidth_GBps to reduce the search space.
    print("Running calibration (this may take a minute)...\n")
    fitted = calibrate(
        entries=entries,
        arch_init=init_arch,
        # fix_clock_freq_GHz=3.0,     # uncomment to pin clock
        # fix_bandwidth_GBps=50.0,    # uncomment to pin bandwidth
        maxiter=2000,
        verbose=True,
    )

    # ── 4. Save fitted parameters ─────────────────────────────────────────
    save_calibration(fitted, OUTPUT_PATH)
    print(f"\nSaved fitted ArchParam to {OUTPUT_PATH}")

    # ── 5. Evaluate accuracy ──────────────────────────────────────────────
    print("\n=== Prediction Accuracy ===\n")
    print(f"{'Entry':<60s} {'Predicted':>10s} {'Measured':>10s} {'Error':>8s}")
    print("-" * 92)

    total_abs_pct = 0.0
    for entry in entries:
        result = Simulator(
            entry.cpp_file,
            entry.header_file,
            entry.config_file,
            arch=fitted,
        ).run()
        pred = result.predicted_latency_s
        meas = entry.measured_time_s
        pct_err = (pred - meas) / meas * 100
        total_abs_pct += abs(pct_err)

        label = os.path.basename(entry.cpp_file).replace(".cpp", "")
        config = os.path.basename(entry.config_file).replace(".json", "")
        print(f"  {label + ' / ' + config:<56s} "
              f"{pred*1e3:>8.1f} ms {meas*1e3:>8.1f} ms {pct_err:>+7.1f}%")

    mape = total_abs_pct / len(entries)
    print(f"\n  MAPE: {mape:.1f}%")

    # ── 6. (Optional) Reload and use the fitted params ────────────────────
    reloaded = load_calibration(OUTPUT_PATH)
    print(f"\nReloaded ArchParam: funits={reloaded.funits}, "
          f"clock={reloaded.clock_freq_GHz:.3f} GHz, "
          f"bw={reloaded.memory_bandwidth_GBps:.1f} GB/s")


if __name__ == "__main__":
    main()
