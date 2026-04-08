from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .config import load_config, FHEConfig
from .interpreter import Interpreter, extract_member_types
from .accuracy import NumericalInterpreter, AccuracyResult
from .arch_params import ArchParam, GPU_ARCH
from .cost_model import CostModel, PredictionResult


@dataclass
class SimulationResult:
    """Combined result from both branches of the simulator."""

    latency:  PredictionResult
    accuracy: Optional[AccuracyResult] = None

    def __str__(self) -> str:
        parts = [str(self.latency)]
        if self.accuracy is not None:
            parts.append("")
            parts.append(str(self.accuracy))
        return "\n".join(parts)


class Simulator:
    """
    Predict latency and/or accuracy of an OpenFHE eval() function.

    Parameters
    ----------
    cpp_file    : path to yourSolution.cpp  (contains the eval() body)
    header_file : path to yourSolution.h    (contains class member declarations)
    config_file : path to config.json       (FHE scheme parameters)
    arch        : ArchParam                 (hardware parameters; default: GPU_ARCH)

    Usage
    -----
    sim = Simulator("solution.cpp", "solution.h", "config.json")

    # Latency only
    result = sim.run()

    # Latency + accuracy
    result = sim.run(
        plaintext_input="testcase1/plaintext_input.txt",
        expected_output="testcase1/expected_output.txt",
    )
    print(result)
    print(result.latency.predicted_latency_s)
    print(result.accuracy.correct_ratio)
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

    def run(
        self,
        plaintext_input:  Optional[Union[str, Path, np.ndarray]] = None,
        expected_output:  Optional[Union[str, Path, np.ndarray]] = None,
        accuracy_threshold: float = 0.001,
    ) -> SimulationResult:
        """
        Run the simulator.

        If plaintext_input is provided, runs both latency and accuracy branches.
        Otherwise, runs latency only (original behavior).

        Parameters
        ----------
        plaintext_input    : path to plaintext_input.txt or numpy array
        expected_output    : path to expected_output.txt or numpy array
        accuracy_threshold : per-slot error threshold for correct_ratio metric
        """
        config       = load_config(str(self.config_file))
        member_types = extract_member_types(self.header_file.read_text())
        cpp_src      = self.cpp_file.read_text()

        if plaintext_input is not None:
            # Unified path: numerical interpreter gives both op_log and accuracy
            input_slots = self._load_array(plaintext_input)
            num_interp = NumericalInterpreter(
                member_types           = member_types,
                initial_level          = config.mult_depth,
                levels_after_bootstrap = config.levels_available_after_bootstrap,
                input_slots            = input_slots,
                scale_mod_size         = config.scale_mod_size,
            )
            op_log = num_interp.run(cpp_src)

            # Latency from the same op_log
            latency = CostModel(self.arch).predict(op_log, config)

            # Accuracy
            if num_interp.output_slots is not None:
                exp = (self._load_array(expected_output)
                       if expected_output is not None else None)
                if exp is not None:
                    accuracy = AccuracyResult.compute(
                        num_interp.output_slots, exp, accuracy_threshold
                    )
                else:
                    accuracy = AccuracyResult(
                        predicted_output=num_interp.output_slots
                    )
            else:
                accuracy = None

            return SimulationResult(latency=latency, accuracy=accuracy)

        else:
            # Latency-only path (original behavior)
            interp = Interpreter(
                member_types           = member_types,
                initial_level          = config.mult_depth,
                levels_after_bootstrap = config.levels_available_after_bootstrap,
            )
            op_log = interp.run(cpp_src)
            latency = CostModel(self.arch).predict(op_log, config)
            return SimulationResult(latency=latency)

    @staticmethod
    def _load_array(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        return np.loadtxt(str(source), dtype=np.float64)
