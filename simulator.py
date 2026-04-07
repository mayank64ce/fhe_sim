from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from .config import load_config, FHEConfig
from .interpreter import Interpreter, extract_member_types

from .arch_params import ArchParam, GPU_ARCH
from .cost_model import CostModel, PredictionResult


class Simulator:
    """
    Predict latency of an OpenFHE eval() function.

    Parameters
    ----------
    cpp_file    : path to yourSolution.cpp  (contains the eval() body)
    header_file : path to yourSolution.h    (contains class member declarations)
    config_file : path to config.json       (FHE scheme parameters)
    arch        : ArchParam                 (hardware parameters; default: GPU_ARCH)

    Usage
    -----
    sim = Simulator("yourSolution.cpp", "yourSolution.h", "config.json", arch=GPU_ARCH)
    result = sim.run()
    print(result)
    """

    def __init__(
        self,
        cpp_file:    str,
        header_file: str,
        config_file: str,
        arch:        ArchParam = GPU_ARCH,
    ):
        self.cpp_file    = Path(cpp_file)
        self.header_file = Path(header_file)
        self.config_file = Path(config_file)
        self.arch        = arch

    def run(self) -> PredictionResult:
        config       = load_config(str(self.config_file))
        member_types = extract_member_types(self.header_file.read_text())
        interp = Interpreter(
            member_types           = member_types,
            initial_level          = config.mult_depth,
            levels_after_bootstrap = config.levels_available_after_bootstrap,
        )
        op_log = interp.run(self.cpp_file.read_text())
        return CostModel(self.arch).predict(op_log, config)
