"""
fhe_sim — SimFHE-style latency simulator with C++ interpreter and calibration.

Quick start
-----------
    from fhe_sim import Simulator, GPU_ARCH

    sim = Simulator(
        cpp_file    = "yourSolution.cpp",
        header_file = "yourSolution.h",
        config_file = "config.json",
        arch        = GPU_ARCH,
    )
    result = sim.run()
    print(result)
"""

from .arch_params import (
    ArchParam,
    CacheStyle,
    CPU_ARCH,
    GPU_ARCH,
    ASIC_ARCH,
)
from .cost_model import PredictionResult
from .simulator import Simulator
from .calibrate import calibrate
from .types import OpType, FHEOp, FHEType, OpCount
from .config import FHEConfig, load_config

__all__ = [
    "Simulator",
    "ArchParam",
    "CacheStyle",
    "PredictionResult",
    "calibrate",
    "CPU_ARCH",
    "GPU_ARCH",
    "ASIC_ARCH",
    "OpType",
    "FHEOp",
    "FHEType",
    "OpCount",
    "FHEConfig",
    "load_config",
]
