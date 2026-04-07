# fhe_sim — SimFHE-style FHE Latency Simulator

## 1. Overview

`fhe_sim` predicts the wall-clock latency of an OpenFHE `eval()` function on
a parameterised hardware target, using a roofline cost model.

**How it works:**

1. `fhe_sim`'s C++ interpreter parses your `eval()` function and produces an
   ordered log of FHE operations (`EvalMult`, `EvalRotate`, etc.) with their
   ciphertext levels.
2. `fhe_sim`'s cost model maps each operation to a `Cost` object that
   accumulates hardware cycles (`add_cyc`, `mult_cyc`, `auto_cyc`, `ntt_cyc`)
   and DRAM bytes (`dram_rd`, `dram_wr`, `dram_key_rd`) following the
   SimFHE/poly.py and SimFHE/evaluator.py pipeline models.
3. The roofline model separates compute and memory:
   - `compute_time = total_cycles / (clock_freq_GHz × 10^9)`
   - `memory_time  = total_dram_bytes / (memory_bandwidth_GBps × 10^9)`
   - `predicted_latency = max(compute_time, memory_time)`

**Key formula** (from SimFHE):

```
add_cyc  = N × limbs / (funits × sets) + add_lat
mult_cyc = N × limbs / (funits × sets) + mult_lat
```

Cost is **not** a weighted sum — compute and memory are kept separate so the
bottleneck is identified by `max()`.

**Differences from `fhe_sim`:**

| Feature | fhe_sim | fhe_sim |
|---------|---------|---------|
| Output | Op counts | Latency in seconds |
| Hardware model | None | ArchParam (cycles + DRAM) |
| Cost accumulation | Counter | Explicit `Cost` objects |
| Calibration | No | Yes (scipy Nelder-Mead) |
| Presets | No | CPU, GPU, ASIC |

---

## 2. Quick Start

```python
from fhe_sim import Simulator, GPU_ARCH, CPU_ARCH, ASIC_ARCH

# Predict latency on a GPU
sim = Simulator(
    cpp_file    = "path/to/yourSolution.cpp",
    header_file = "path/to/yourSolution.h",
    config_file = "path/to/config.json",
    arch        = GPU_ARCH,
)
result = sim.run()
print(result)
# === fhe_sim Prediction ===
# Predicted latency :  12.345 ms
#   Compute time    :   8.210 ms
#   Memory time     :  12.345 ms
#   Bottleneck      : memory
# ...

# Access individual fields
print(f"Predicted: {result.predicted_latency_s * 1e3:.2f} ms")
print(f"Bottleneck: {result.bottleneck}")

# Inspect per-operation breakdown
for op_type, level, cost in result.per_op_costs:
    print(f"  {op_type.value:30s} lvl={level:3d}  "
          f"mult={cost.mult_cyc:>12,.0f} cyc  "
          f"dram={cost.total_dram_bytes/1e6:.1f} MB")
```

### Compare architectures

```python
from fhe_sim import Simulator, CPU_ARCH, GPU_ARCH, ASIC_ARCH

for name, arch in [("CPU", CPU_ARCH), ("GPU", GPU_ARCH), ("ASIC", ASIC_ARCH)]:
    result = Simulator("sol.cpp", "sol.h", "config.json", arch=arch).run()
    print(f"{name}: {result.predicted_latency_s * 1e3:.3f} ms  [{result.bottleneck}]")
```

---

## 3. Hardware Configuration

### `ArchParam` fields

| Field | Type | Description |
|-------|------|-------------|
| `funits` | int | Modular mul/add functional units per set |
| `sets` | int | Independent sets of functional units (total parallelism = funits × sets) |
| `add_lat` | int | Pipeline initiation latency for additions (cycles) |
| `mult_lat` | int | Pipeline initiation latency for multiplications (cycles) |
| `ntt_lat` | int | Per-stage latency for dedicated NTT unit (if `dedicated_ntt_unit=True`) |
| `auto_lat` | int | Pipeline initiation latency for automorphisms (cycles) |
| `karatsuba` | bool | Use Karatsuba: 2 mults + 4 adds instead of 3 mults + 1 add per EvalMult |
| `key_compression` | bool | Store one polynomial per key-switch key (halves key DRAM) |
| `rescale_fusion` | bool | Fuse rescale into mod_down (saves one DRAM pass per EvalMult) |
| `dedicated_ntt_unit` | bool | Model NTT with a separate hardware unit |
| `cache_style` | CacheStyle | On-chip cache amount (NONE < CONST < BETA < ALPHA) |
| `clock_freq_GHz` | float | Processor clock frequency |
| `memory_bandwidth_GBps` | float | Usable DRAM bandwidth (reads + writes combined) |

### `CacheStyle` enum

| Value | Meaning |
|-------|---------|
| `NONE` | No cache; every limb round-trips through DRAM |
| `CONST` | Small constant cache; eliminates a-term write-back before key-switch hoisting |
| `BETA` | Cache holds dnum × alpha limbs; mod_raise intermediate stays on-chip per digit |
| `ALPHA` | Cache holds alpha limbs; full BConv stays on-chip without DRAM round-trips |

### Preset configurations

```python
from fhe_sim import CPU_ARCH, GPU_ARCH, ASIC_ARCH

# CPU_ARCH:  4 funits, 1 set, 3 GHz, 50 GB/s, no cache
# GPU_ARCH:  128 funits, 16 sets, 1.5 GHz, 900 GB/s, no cache
# ASIC_ARCH: 256 funits, 32 sets, 1 GHz, 200 GB/s, full alpha-cache,
#             karatsuba + key_compression + rescale_fusion enabled
```

### Custom configuration

```python
from fhe_sim import ArchParam, CacheStyle

my_arch = ArchParam(
    funits=64, sets=8,
    add_lat=2, mult_lat=8, ntt_lat=8, auto_lat=2,
    karatsuba=True,
    key_compression=False,
    rescale_fusion=True,
    dedicated_ntt_unit=False,
    cache_style=CacheStyle.CONST,
    clock_freq_GHz=2.0,
    memory_bandwidth_GBps=300.0,
)

# Save and reload
my_arch.save("my_arch.json")
reloaded = ArchParam.load("my_arch.json")
```

---

## 4. Calibration Guide

Calibration fits the unknown hardware parameters (`funits × sets`, `add_lat`,
`mult_lat`, `auto_lat`, and optionally `clock_freq_GHz`, `memory_bandwidth_GBps`)
to real timing measurements using gradient-free Nelder-Mead optimisation.

### What data to collect

Run your actual OpenFHE program and measure wall-clock time for the `eval()` call:

```cpp
auto start = std::chrono::high_resolution_clock::now();
solution.eval(cc, ciphertexts, publicKey);
auto end   = std::chrono::high_resolution_clock::now();
double elapsed_s = std::chrono::duration<double>(end - start).count();
```

Collect measurements across **different configurations** (varying `mult_depth`, `N`,
or operation mix) to ensure the optimiser can disentangle the parameters.

### CSV format

Create a CSV file with these columns:

```
cpp_file,header_file,config_file,measured_time_s
solution_d10.cpp,solution_d10.h,config_d10.json,0.00312
solution_d20.cpp,solution_d20.h,config_d20.json,0.00891
solution_d30.cpp,solution_d30.h,config_d30.json,0.02145
```

### Step-by-step calibration example

```python
from fhe_sim import calibrate, GPU_ARCH
from fhe_sim.calibrate import load_entries_from_csv, save_calibration

# 1. Load timing measurements
entries = load_entries_from_csv("timings.csv")

# 2. Calibrate — pin clock and bandwidth from hardware specs, fit parallelism + latencies
fitted = calibrate(
    entries             = entries,
    arch_init           = GPU_ARCH,          # starting point
    fix_clock_freq_GHz  = 1.41,              # from GPU spec sheet
    fix_bandwidth_GBps  = 900.0,             # from GPU spec sheet
    verbose             = True,
)

# 3. Save the calibrated parameters
fitted.save("calibrated_gpu.json")
print(f"Fitted: funits={fitted.funits}, sets={fitted.sets}")
print(f"  add_lat={fitted.add_lat}, mult_lat={fitted.mult_lat}")
```

### Which parameters to fix vs. fit

**Fix if known from hardware spec:**
- `clock_freq_GHz` — usually available from the manufacturer
- `memory_bandwidth_GBps` — usually available from the manufacturer or `nvidia-smi`

**Fit from measurements:**
- `funits × sets` (total parallelism) — depends on how many modular arithmetic
  units are actually utilised for FHE workloads; hard to know from specs alone
- `add_lat`, `mult_lat`, `auto_lat` — pipeline depths depend on the specific
  implementation and compiler/runtime; fit these even if nominal clock is known

**Keep fixed (algorithmic flags):**
- `karatsuba`, `key_compression`, `rescale_fusion`, `cache_style` — these must
  match what your actual OpenFHE build does; do not fit them automatically

### How many data points are needed

- **Minimum:** 5–10 measurements, ideally from different `mult_depth` and `N`
  values, so the optimiser sees variation in both compute and memory cost.
- **Recommended:** 15–30 measurements spanning a range of operation mixes
  (rotate-heavy vs. mult-heavy workloads).

### Common pitfalls

**Warning — compute-bound training data:**
If all your measurements are compute-bound (`bottleneck == "compute"`), the
optimiser has no signal to fit `memory_bandwidth_GBps`. Use `fix_bandwidth_GBps`
with the known spec value, or add memory-bound workloads (large `N`, low `dnum`).

**Warning — memory-bound training data:**
Conversely, if all measurements are memory-bound, `funits × sets` and latency
terms are poorly constrained. Add smaller-`N` or ASIC-style workloads.

**Checking fit quality:**
```python
from fhe_sim.calibrate import load_entries_from_csv
from fhe_sim import Simulator

fitted = ArchParam.load("calibrated_gpu.json")
for entry in load_entries_from_csv("timings.csv"):
    result = Simulator(entry.cpp_file, entry.header_file,
                       entry.config_file, arch=fitted).run()
    rel_err = abs(result.predicted_latency_s - entry.measured_time_s) \
              / entry.measured_time_s
    print(f"  {entry.cpp_file}: predicted={result.predicted_latency_s*1e3:.2f} ms  "
          f"measured={entry.measured_time_s*1e3:.2f} ms  "
          f"rel_err={rel_err*100:.1f}%")
```

---

## 5. Runnable Calibration Example

`example_calibrate.py` is a self-contained script that loads real timing data,
runs calibration, and prints per-entry prediction accuracy:

```bash
# From the project root
python -m fhe_sim.example_calibrate
```

It uses the timing measurements in `tools/calibration/timings_v111.csv` (20
data points: 4 programs × 5 FHE configs) collected from OpenFHE v1.1.1 running
inside Docker. The script will:

1. Load the CSV entries
2. Fit 9 parameters (parallelism, pipeline latencies, clock, bandwidth,
   per-operation overheads) via Nelder-Mead optimisation
3. Save the fitted `ArchParam` to a JSON file
4. Print a table of predicted vs. measured times with per-entry error and MAPE

See the source at [`fhe_sim/example_calibrate.py`](example_calibrate.py) for
how to adapt this to your own timing data.

---

## 6. Files Reference

| File | Purpose |
|------|---------|
| `__init__.py` | Package entry point; re-exports main API |
| `arch_params.py` | `ArchParam`, `CacheStyle`, presets (`CPU_ARCH`, `GPU_ARCH`, `ASIC_ARCH`) |
| `hw_model.py` | `Cost` dataclass; primitive cycle formulas (`poly_add`, `poly_mult`, `poly_ntt`, `poly_automorph`, `basis_convert`) |
| `op_model.py` | Full FHE operation costs: `op_eval_add`, `op_eval_mult_ctpt`, `op_eval_mult_ctct`, `op_eval_rotate`; all DRAM accounting and cache-style checks |
| `cost_model.py` | `CostModel` maps op_log → `PredictionResult`; roofline bottleneck analysis |
| `simulator.py` | `Simulator` chains fhe_sim interpreter → CostModel |
| `calibrate.py` | `calibrate()`, `CalibrationEntry`, `load_entries_from_csv()`, `save_calibration()`, `load_calibration()` |
| `example_calibrate.py` | Runnable end-to-end calibration example (see Section 5) |
| `README.md` | This file |

> The hardware cost model in fhe_sim (cycle counting, DRAM traffic accounting,
> NTT butterfly model, and key-switch pipeline) is ported from
> [SimFHE](https://github.com/bu-icsg/SimFHE) by Agrawal et al. Thanks to the SimFHE authors for
> making their simulator publicly available.