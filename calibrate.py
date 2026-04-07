"""
calibrate.py — Fit ArchParam to real wall-clock timing measurements.

Uses scipy.optimize.minimize (Nelder-Mead) to minimise the sum of squared
relative errors between simulated and measured latencies.

Free parameters (in log-space where appropriate):
  log_p_eff  : log(funits * sets)  — total parallelism (confounded, treated as one)
  add_lat    : pipeline initiation latency for adds (cycles)
  mult_lat   : pipeline initiation latency for mults (cycles)
  auto_lat   : pipeline initiation latency for automorphisms (cycles)
  log_clock  : log(clock_freq_GHz)    [optional, fix with fix_clock_freq_GHz]
  log_bw     : log(memory_bandwidth_GBps)  [optional, fix with fix_bandwidth_GBps]

Usage
-----
    entries = load_entries_from_csv("timings.csv")
    fitted  = calibrate(entries, arch_init=GPU_ARCH)
    fitted.save("calibrated.json")
"""
from __future__ import annotations

import csv
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from .arch_params import ArchParam
from .cost_model import CostModel
from .simulator import Simulator

from .config import load_config
from .interpreter import Interpreter, extract_member_types


@dataclass
class CalibrationEntry:
    """One measurement: a compiled eval() file and its wall-clock latency."""
    cpp_file:        str
    header_file:     str
    config_file:     str
    measured_time_s: float


def _simulate_entry(entry: CalibrationEntry, arch: ArchParam) -> float:
    """Run Simulator on one entry and return predicted latency in seconds."""
    sim = Simulator(
        cpp_file    = entry.cpp_file,
        header_file = entry.header_file,
        config_file = entry.config_file,
        arch        = arch,
    )
    result = sim.run()
    return result.predicted_latency_s


def _build_arch(
    params_vec: list[float],
    base_arch:  ArchParam,
    fix_clock:  Optional[float],
    fix_bw:     Optional[float],
) -> ArchParam:
    """
    Reconstruct an ArchParam from the optimiser parameter vector.

    Vector layout:
      [0] log_p_eff   → p_eff = exp(params_vec[0])  (total parallelism)
      [1] add_lat     → rounded to nearest int
      [2] mult_lat    → rounded to nearest int
      [3] auto_lat    → rounded to nearest int
      [4] log_clock   → clock_freq_GHz = exp(params_vec[4])  (if not fixed)
      [5] log_bw      → memory_bandwidth_GBps = exp(params_vec[5])  (if not fixed)
      [next] log overheads (rotate, mult, add)
    """
    idx = 0
    p_eff    = max(1.0, math.exp(params_vec[idx]));  idx += 1
    add_lat  = max(1,   round(params_vec[idx]));      idx += 1
    mult_lat = max(1,   round(params_vec[idx]));      idx += 1
    auto_lat = max(1,   round(params_vec[idx]));      idx += 1

    if fix_clock is not None:
        clock = fix_clock
    else:
        clock = max(0.1, math.exp(params_vec[idx]));  idx += 1

    if fix_bw is not None:
        bw = fix_bw
    else:
        bw = max(1.0, math.exp(params_vec[idx]));     idx += 1

    # Per-operation-type overheads (in seconds)
    rot_overhead  = max(0.0, math.exp(params_vec[idx])); idx += 1
    mult_overhead = max(0.0, math.exp(params_vec[idx])); idx += 1
    add_overhead  = max(0.0, math.exp(params_vec[idx])); idx += 1

    # Distribute p_eff: keep arch.sets fixed, adjust funits
    new_sets  = base_arch.sets
    new_funits = max(1, round(p_eff / max(new_sets, 1)))

    arch = deepcopy(base_arch)
    arch.funits               = new_funits
    arch.sets                 = new_sets
    arch.add_lat              = add_lat
    arch.mult_lat             = mult_lat
    arch.auto_lat             = auto_lat
    arch.clock_freq_GHz       = clock
    arch.memory_bandwidth_GBps = bw
    arch.per_rotate_overhead_s = rot_overhead
    arch.per_mult_overhead_s   = mult_overhead
    arch.per_add_overhead_s    = add_overhead
    return arch


def _initial_vector(
    arch:        ArchParam,
    fix_clock:   Optional[float],
    fix_bw:      Optional[float],
) -> list[float]:
    """Build the initial parameter vector from arch."""
    p_eff = arch.funits * arch.sets
    vec   = [
        math.log(max(p_eff, 1)),
        float(arch.add_lat),
        float(arch.mult_lat),
        float(arch.auto_lat),
    ]
    if fix_clock is None:
        vec.append(math.log(max(arch.clock_freq_GHz, 0.1)))
    if fix_bw is None:
        vec.append(math.log(max(arch.memory_bandwidth_GBps, 1.0)))
    # Per-operation-type overheads (log-space)
    vec.append(math.log(max(arch.per_rotate_overhead_s, 1e-6)))
    vec.append(math.log(max(arch.per_mult_overhead_s, 1e-6)))
    vec.append(math.log(max(arch.per_add_overhead_s, 1e-7)))
    return vec


def calibrate(
    entries:              list[CalibrationEntry],
    arch_init:            ArchParam,
    fix_clock_freq_GHz:   Optional[float] = None,
    fix_bandwidth_GBps:   Optional[float] = None,
    maxiter:              int = 2000,
    xatol:                float = 1e-4,
    fatol:                float = 1e-6,
    verbose:              bool = False,
) -> ArchParam:
    """
    Fit ArchParam parameters to timing measurements.

    Parameters
    ----------
    entries              : list of CalibrationEntry
    arch_init            : starting ArchParam (other fields, e.g. cache_style, are kept fixed)
    fix_clock_freq_GHz   : if not None, pin clock frequency and don't fit it
    fix_bandwidth_GBps   : if not None, pin memory bandwidth and don't fit it
    maxiter              : maximum Nelder-Mead iterations
    xatol / fatol        : convergence tolerances
    verbose              : print per-iteration loss if True

    Returns
    -------
    Fitted ArchParam
    """
    try:
        from scipy.optimize import minimize
    except ImportError as e:
        raise ImportError(
            "scipy is required for calibration. Install it with: pip install scipy"
        ) from e

    if not entries:
        raise ValueError("calibrate() requires at least one CalibrationEntry.")

    x0 = _initial_vector(arch_init, fix_clock_freq_GHz, fix_bandwidth_GBps)
    iteration = [0]

    def objective(x: list[float]) -> float:
        arch = _build_arch(x, arch_init, fix_clock_freq_GHz, fix_bandwidth_GBps)
        loss = 0.0
        for entry in entries:
            try:
                predicted = _simulate_entry(entry, arch)
                rel_err   = (predicted - entry.measured_time_s) / entry.measured_time_s
                loss     += rel_err ** 2
            except Exception:
                loss += 1e6   # penalise failed simulations heavily
        if verbose:
            iteration[0] += 1
            if iteration[0] % 50 == 0:
                print(f"  iter {iteration[0]:4d}  loss={loss:.6f}")
        return loss

    result = minimize(
        objective,
        x0,
        method  = "Nelder-Mead",
        options = {
            "maxiter": maxiter,
            "xatol":   xatol,
            "fatol":   fatol,
            "disp":    verbose,
        },
    )

    fitted_arch = _build_arch(
        result.x, arch_init, fix_clock_freq_GHz, fix_bandwidth_GBps
    )

    if verbose:
        print(f"\nCalibration finished: success={result.success}, "
              f"loss={result.fun:.6f}, iters={result.nit}")
        print(f"  funits={fitted_arch.funits}, sets={fitted_arch.sets}")
        print(f"  add_lat={fitted_arch.add_lat}, mult_lat={fitted_arch.mult_lat}, "
              f"auto_lat={fitted_arch.auto_lat}")
        print(f"  clock={fitted_arch.clock_freq_GHz:.3f} GHz, "
              f"bw={fitted_arch.memory_bandwidth_GBps:.1f} GB/s")
        print(f"  per_rotate_overhead={fitted_arch.per_rotate_overhead_s*1e3:.3f} ms")
        print(f"  per_mult_overhead={fitted_arch.per_mult_overhead_s*1e3:.3f} ms")
        print(f"  per_add_overhead={fitted_arch.per_add_overhead_s*1e3:.4f} ms")

    return fitted_arch


def save_calibration(arch: ArchParam, path: str) -> None:
    """Save a calibrated ArchParam to a JSON file."""
    arch.save(path)


def load_calibration(path: str) -> ArchParam:
    """Load a previously calibrated ArchParam from a JSON file."""
    return ArchParam.load(path)


def load_entries_from_csv(csv_path: str) -> list[CalibrationEntry]:
    """
    Load calibration entries from a CSV file.

    Expected columns (in any order, with header row):
      cpp_file, header_file, config_file, measured_time_s

    Example CSV
    -----------
    cpp_file,header_file,config_file,measured_time_s
    solution1.cpp,solution1.h,config_d10.json,0.00312
    solution2.cpp,solution2.h,config_d20.json,0.00891
    """
    entries = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(CalibrationEntry(
                cpp_file        = row["cpp_file"].strip(),
                header_file     = row["header_file"].strip(),
                config_file     = row["config_file"].strip(),
                measured_time_s = float(row["measured_time_s"].strip()),
            ))
    return entries
