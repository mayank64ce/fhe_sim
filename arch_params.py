from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import json


class CacheStyle(Enum):
    """
    NONE  : no cache; every limb round-trips through DRAM
    CONST : small constant cache; eliminates a-term write-back before key-switch hoisting
    BETA  : cache holds dnum × alpha limbs; mod_raise intermediate stays on-chip per digit
    ALPHA : cache holds alpha limbs; full BConv stays on-chip without DRAM round-trips
    """
    NONE  = 0
    CONST = 1
    BETA  = 2
    ALPHA = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented


@dataclass
class ArchParam:
    """
    Hardware parameters for the latency-throughput pipeline cost model.

    Compute
    -------
    funits : int
        Number of modular mul/add functional units per set.
    sets : int
        Number of independent sets of functional units.
        Total parallelism = funits * sets.
    add_lat : int
        Pipeline initiation latency for modular additions (cycles).
        add_cyc = N * limbs / (funits * sets) + add_lat
    mult_lat : int
        Pipeline initiation latency for modular multiplications (cycles).
    ntt_lat : int
        Per-stage pipeline initiation latency for dedicated NTT units.
        Used only when dedicated_ntt_unit=True.
    auto_lat : int
        Pipeline initiation latency for automorphism (permutation) steps.

    Algorithmic flags
    -----------------
    karatsuba : bool
        Karatsuba cross-term in EvalMult: 2 muls + 4 adds instead of 3 muls + 1 add.
        Only helps when mult is the bottleneck.
    key_compression : bool
        Store one polynomial per key-switch key (halves key DRAM traffic).
    rescale_fusion : bool
        Fuse rescale into mod_down after EvalMult (saves one DRAM pass).
    dedicated_ntt_unit : bool
        Model NTT with a dedicated hardware unit (ntt_cyc model) rather than
        the standard butterfly add+mult model.
    cache_style : CacheStyle
        Amount of on-chip cache available for intermediate limbs.

    Hardware clocks / bandwidth
    ---------------------------
    clock_freq_GHz : float
        Processor clock frequency.
    memory_bandwidth_GBps : float
        Usable DRAM bandwidth (reads + writes combined), in GB/s.
    """
    # Parallelism
    funits: int = 32
    sets:   int = 8

    # Pipeline latencies
    add_lat:  int = 4
    mult_lat: int = 16
    ntt_lat:  int = 16
    auto_lat: int = 2

    # Algorithmic optimisations
    karatsuba:          bool = False
    key_compression:    bool = False
    rescale_fusion:     bool = False
    dedicated_ntt_unit: bool = False

    # Cache
    cache_style: CacheStyle = CacheStyle.NONE

    # Hardware
    clock_freq_GHz:        float = 1.0
    memory_bandwidth_GBps: float = 100.0

    # Per-operation fixed overheads (seconds) — captures software costs
    # that don't scale with N (function calls, allocation, cache misses, etc.)
    per_rotate_overhead_s: float = 0.0
    per_mult_overhead_s:   float = 0.0
    per_add_overhead_s:    float = 0.0

    def key_factor(self) -> int:
        """1 if key-compressed, else 2 (full key has two polynomials)."""
        return 1 if self.key_compression else 2

    def save(self, path: str) -> None:
        d = {k: (v.name if isinstance(v, CacheStyle) else v)
             for k, v in self.__dict__.items()}
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ArchParam":
        with open(path) as f:
            d = json.load(f)
        if "cache_style" in d and isinstance(d["cache_style"], str):
            d["cache_style"] = CacheStyle[d["cache_style"]]
        return cls(**d)


# ── Preset configurations ──────────────────────────────────────────────────

CPU_ARCH = ArchParam(
    funits=4, sets=1,
    add_lat=1, mult_lat=4, ntt_lat=4, auto_lat=1,
    cache_style=CacheStyle.NONE,
    clock_freq_GHz=3.0,
    memory_bandwidth_GBps=50.0,
    per_rotate_overhead_s=0.0,
    per_mult_overhead_s=0.0,
    per_add_overhead_s=0.0,
)

GPU_ARCH = ArchParam(
    funits=128, sets=16,
    add_lat=4, mult_lat=16, ntt_lat=16, auto_lat=2,
    cache_style=CacheStyle.NONE,
    clock_freq_GHz=1.5,
    memory_bandwidth_GBps=900.0,
)

ASIC_ARCH = ArchParam(
    funits=256, sets=32,
    add_lat=4, mult_lat=16, ntt_lat=16, auto_lat=2,
    karatsuba=True,
    key_compression=True,
    rescale_fusion=True,
    cache_style=CacheStyle.ALPHA,
    clock_freq_GHz=1.0,
    memory_bandwidth_GBps=200.0,
)
