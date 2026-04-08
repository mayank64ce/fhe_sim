# fhe_sim — FHE Latency + Accuracy Simulator

## 1. Overview

`fhe_sim` predicts **latency** and **accuracy** of an OpenFHE CKKS `eval()`
function without compiling or running any FHE code.

**Two branches from one parse:**

```
  C++ eval() source
       │
       ▼
  ┌──────────┐
  │  Parser  │  (tree-sitter → AST)
  └────┬─────┘
       │
       ├──────────────────┐
       ▼                  ▼
  ┌──────────┐     ┌───────────┐
  │ Latency  │     │ Accuracy  │
  │ Predictor│     │ Simulator │
  └──────────┘     └───────────┘
       │                 │
       ▼                 ▼
  predicted         predicted output
  wall-clock (s)    + accuracy metrics
```

**Latency branch:** The C++ interpreter produces an ordered log of FHE
operations with ciphertext levels. The cost model maps each operation to
hardware cycles and DRAM bytes following the SimFHE pipeline model. A
roofline model combines compute and memory time into a wall-clock prediction.

**Accuracy branch:** A numerical interpreter executes the same operations
on plaintext float vectors (numpy arrays) — the CKKS analog of Concrete ML's
simulate mode for TFHE. No encryption, no polynomials, just slot-level
arithmetic. Produces predicted output values that can be compared to ground
truth for accuracy metrics (MAE, max error, correct slot percentage).

### What you need

| File | Description |
|------|-------------|
| `solution.cpp` | C++ file containing the `eval()` method |
| `solution.h` | Header declaring the class with member types |
| `config.json` | FHE scheme parameters (mult_depth, ring_dimension, etc.) |

Optional for accuracy: `plaintext_input.txt` and `expected_output.txt`
(one float per line, one per CKKS slot).

### What you do NOT need

- No OpenFHE installation
- No Docker or compilation
- No encrypted keys or ciphertexts

---

## 2. Quick Start

### Latency only

```python
from fhe_sim import Simulator

sim = Simulator("solution.cpp", "solution.h", "config.json")
result = sim.run()

print(result.latency.predicted_latency_s)  # seconds
print(result.latency.bottleneck)           # "compute" or "memory"
```

### Latency + Accuracy (unified API)

```python
result = sim.run(
    plaintext_input="plaintext_input.txt",
    expected_output="expected_output.txt",
)

# Latency
print(f"Latency: {result.latency.predicted_latency_s * 1e3:.2f} ms")

# Accuracy
print(f"MAE:           {result.accuracy.mae:.6e}")
print(f"Max error:     {result.accuracy.max_error:.6e}")
print(f"Correct slots: {result.accuracy.correct_ratio:.2%}")

# You can also pass numpy arrays directly
import numpy as np
result = sim.run(
    plaintext_input=np.random.uniform(-1, 1, 16384),
    expected_output=expected_values,
    accuracy_threshold=0.01,  # per-slot correctness threshold
)
```

### Compare architectures

```python
from fhe_sim import Simulator, CPU_ARCH, GPU_ARCH, ASIC_ARCH

for name, arch in [("CPU", CPU_ARCH), ("GPU", GPU_ARCH), ("ASIC", ASIC_ARCH)]:
    result = Simulator("sol.cpp", "sol.h", "config.json", arch=arch).run()
    print(f"{name}: {result.latency.predicted_latency_s * 1e3:.3f} ms")
```

### Use calibrated hardware parameters

```python
from fhe_sim import Simulator
from fhe_sim.arch_params import ArchParam

arch = ArchParam.load("fitted_arch.json")
sim = Simulator("solution.cpp", "solution.h", "config.json", arch=arch)
result = sim.run()
```

---

## 3. `config.json` Format

```json
{
    "mult_depth": 12,
    "ring_dimension": 32768,
    "scale_mod_size": 49,
    "first_mod_size": 60,
    "num_large_digits": 3,
    "batch_size": 16384,
    "enable_bootstrapping": false,
    "levels_available_after_bootstrap": 10,
    "level_budget": [4, 4],
    "indexes_for_rotation_key": [1, 2, 4]
}
```

Only `mult_depth` is strictly required. If `ring_dimension` is omitted, it is
derived from the HE security standard table. Other fields have sensible
defaults for CKKS.

---

## 4. Hardware Configuration

### `ArchParam` fields

| Field | Type | Description |
|-------|------|-------------|
| `funits` | int | Modular mul/add functional units per set |
| `sets` | int | Independent sets of functional units (total parallelism = funits x sets) |
| `add_lat` | int | Pipeline initiation latency for additions (cycles) |
| `mult_lat` | int | Pipeline initiation latency for multiplications (cycles) |
| `ntt_lat` | int | Per-stage latency for dedicated NTT unit (if `dedicated_ntt_unit=True`) |
| `auto_lat` | int | Pipeline initiation latency for automorphisms (cycles) |
| `karatsuba` | bool | Use Karatsuba: 2 mults + 4 adds instead of 3 mults + 1 add |
| `key_compression` | bool | Store one polynomial per key-switch key (halves key DRAM) |
| `rescale_fusion` | bool | Fuse rescale into mod_down (saves one DRAM pass) |
| `dedicated_ntt_unit` | bool | Model NTT with a separate hardware unit |
| `cache_style` | CacheStyle | On-chip cache amount (NONE < CONST < BETA < ALPHA) |
| `clock_freq_GHz` | float | Processor clock frequency |
| `memory_bandwidth_GBps` | float | Usable DRAM bandwidth (GB/s) |
| `per_rotate_overhead_s` | float | Fixed per-rotation overhead (seconds) |
| `per_mult_overhead_s` | float | Fixed per-multiplication overhead (seconds) |
| `per_add_overhead_s` | float | Fixed per-addition overhead (seconds) |

### Preset configurations

```python
from fhe_sim import CPU_ARCH, GPU_ARCH, ASIC_ARCH

# CPU_ARCH:  4 funits, 1 set, 3 GHz, 50 GB/s, no cache
# GPU_ARCH:  128 funits, 16 sets, 1.5 GHz, 900 GB/s, no cache
# ASIC_ARCH: 256 funits, 32 sets, 1 GHz, 200 GB/s, full alpha-cache,
#            karatsuba + key_compression + rescale_fusion enabled
```

---

## 5. Calibration Guide

Calibration fits hardware parameters to real timing measurements using
gradient-free Nelder-Mead optimisation.

### Collect timing data

Measure wall-clock time for the `eval()` call in your OpenFHE program:

```cpp
auto t0 = std::chrono::high_resolution_clock::now();
solver.eval();
auto t1 = std::chrono::high_resolution_clock::now();
double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
```

Collect measurements across different configs (varying `mult_depth`, `N`, or
operation mix) to give the optimiser enough signal.

### CSV format

```
cpp_file,header_file,config_file,measured_time_s
solution.cpp,solution.h,config_d10.json,0.00312
solution.cpp,solution.h,config_d20.json,0.00891
```

### Run calibration

```python
from fhe_sim import calibrate, GPU_ARCH
from fhe_sim.calibrate import load_entries_from_csv

entries = load_entries_from_csv("timings.csv")
fitted = calibrate(entries, arch_init=GPU_ARCH, verbose=True)
fitted.save("calibrated.json")
```

### Runnable example

```bash
python -m fhe_sim.example_calibrate
```

Uses timing data from `tools/calibration/timings_v111.csv` collected from
OpenFHE v1.1.1 in Docker.

### How many data points

- **Minimum:** 5-10 measurements with different `mult_depth` and `N`.
- **Recommended:** 20-50 measurements spanning rotation-heavy, mult-heavy,
  and add-heavy workloads.

---

## 6. Validation Results

Validated against actual FHE runs (OpenFHE v1.1.1 in Docker) across 5
programs and 5 CKKS configurations:

### Accuracy branch

| Metric | Result |
|--------|--------|
| Max Correct% disagreement (sim vs FHE) | **0.0000 pct pts** |
| Max MAE disagreement (sim vs FHE) | **6.0e-11** |

The simulator produces accuracy metrics identical to actual FHE to within
the CKKS noise floor (~1e-10).

### Latency branch (after calibration on 52 data points)

| Metric | Result |
|--------|--------|
| MAPE | **17.9%** |
| Spearman rho | **0.977** |

The simulator correctly rank-orders program/config speed in 97.7% of
pairwise comparisons.

---

## 7. Files Reference

| File | Purpose |
|------|---------|
| `__init__.py` | Package entry point; re-exports main API |
| `simulator.py` | `Simulator` — unified entry point returning `SimulationResult` |
| `accuracy.py` | `NumericalInterpreter` — slot-level numpy executor; `AccuracyResult` |
| `interpreter.py` | C++ parser (tree-sitter) producing FHE operation log with levels |
| `cost_model.py` | `CostModel` maps op_log to `PredictionResult` via roofline model |
| `arch_params.py` | `ArchParam`, `CacheStyle`, presets (`CPU_ARCH`, `GPU_ARCH`, `ASIC_ARCH`) |
| `hw_model.py` | `Cost` dataclass; primitive cycle formulas |
| `op_model.py` | Full FHE operation costs with DRAM accounting |
| `config.py` | `FHEConfig`, `load_config()` — FHE scheme parameter handling |
| `types.py` | `FHEType`, `FHEOp`, `OpType`, `OpCount` |
| `calibrate.py` | `calibrate()`, `CalibrationEntry`, CSV loader |
| `example_calibrate.py` | Runnable calibration example |

> The hardware cost model (cycle counting, DRAM traffic accounting, NTT
> butterfly model, key-switch pipeline) is ported from
> [SimFHE](https://github.com/bu-icsg/SimFHE) by Agrawal et al.
