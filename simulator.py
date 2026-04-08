from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .config import load_config, FHEConfig
from .types import FHEType
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

    # Latency + accuracy from test_case.json
    result = sim.run(test_case="tests/testcase1/test_case.json")

    # Latency + accuracy from raw arrays
    result = sim.run(
        plaintext_input=np.array([...]),
        expected_output=np.array([...]),
    )
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
        test_case:        Optional[Union[str, Path]] = None,
        plaintext_input:  Optional[Union[str, Path, np.ndarray, dict]] = None,
        expected_output:  Optional[Union[str, Path, np.ndarray]] = None,
        accuracy_threshold: float = 0.001,
        run_index:        int = 0,
    ) -> SimulationResult:
        """
        Run the simulator.

        Parameters
        ----------
        test_case          : path to test_case.json (extracts input, output,
                             and threshold automatically)
        plaintext_input    : path to plaintext_input.txt, numpy array, or
                             dict mapping member names to numpy arrays
                             (ignored if test_case is provided)
        expected_output    : path to expected_output.txt or numpy array
                             (ignored if test_case is provided)
        accuracy_threshold : per-slot error threshold for correct_ratio metric
                             (overridden by test_case if it contains one)
        run_index          : which run to use from test_case.json (default: 0)
        """
        # If test_case.json is provided, extract input/output from it
        if test_case is not None:
            plaintext_input, expected_output, accuracy_threshold = \
                self._load_test_case(test_case, run_index, accuracy_threshold)

        config       = load_config(str(self.config_file))
        member_types = extract_member_types(self.header_file.read_text())
        cpp_src      = self.cpp_file.read_text()

        if plaintext_input is not None:
            # Normalise to dict {member_name: np.ndarray}
            if isinstance(plaintext_input, dict):
                input_slots = {
                    k: self._load_array(v) for k, v in plaintext_input.items()
                }
            else:
                input_slots = self._load_array(plaintext_input)

            num_interp = NumericalInterpreter(
                member_types           = member_types,
                initial_level          = config.mult_depth,
                levels_after_bootstrap = config.levels_available_after_bootstrap,
                input_slots            = input_slots,
                scale_mod_size         = config.scale_mod_size,
            )
            op_log = num_interp.run(cpp_src)

            latency = CostModel(self.arch).predict(op_log, config)

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
            interp = Interpreter(
                member_types           = member_types,
                initial_level          = config.mult_depth,
                levels_after_bootstrap = config.levels_available_after_bootstrap,
            )
            op_log = interp.run(cpp_src)
            latency = CostModel(self.arch).predict(op_log, config)
            return SimulationResult(latency=latency)

    def _load_test_case(
        self,
        path: Union[str, Path],
        run_index: int = 0,
        default_threshold: float = 0.001,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
        """
        Load input/output from a test_case.json file.

        Format (matches white_box challenge format):
        [{
            "scheme": "CKKS",
            "significant_slots_number": 16384,
            "accuracy_threshold": 0.001,         // optional
            "runs": [{
                "input": [{"name": "x", "value": [...]}],
                "output": [...]
            }]
        }]

        Returns (input_dict, expected_output, threshold).
        input_dict maps header Ciphertext member names → numpy arrays.
        """
        with open(str(path)) as f:
            data = json.load(f)

        tc = data[0]
        run = tc["runs"][run_index]

        # Get input Ciphertext member names from header (excluding m_OutputC)
        member_types = extract_member_types(self.header_file.read_text())
        ct_input_members = [
            name for name, t in member_types.items()
            if t == FHEType.CIPHERTEXT and name != "m_OutputC"
        ]

        # Map test_case inputs to header members by order
        inputs_json = run["input"]
        input_dict: dict[str, np.ndarray] = {}

        if len(inputs_json) == 1 and len(ct_input_members) == 1:
            # Single input — direct mapping
            input_dict[ct_input_members[0]] = np.array(
                inputs_json[0]["value"], dtype=np.float64
            )
        elif len(inputs_json) == len(ct_input_members):
            # Multi-input — match by order
            for member_name, inp in zip(ct_input_members, inputs_json):
                input_dict[member_name] = np.array(
                    inp["value"], dtype=np.float64
                )
        else:
            # Fallback: single input as m_InputC
            input_dict["m_InputC"] = np.array(
                inputs_json[0]["value"], dtype=np.float64
            )

        # Output
        expected_output = np.array(run["output"], dtype=np.float64)

        # Threshold
        threshold = tc.get("accuracy_threshold", default_threshold)

        return input_dict, expected_output, threshold

    @staticmethod
    def _load_array(source: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source
        return np.loadtxt(str(source), dtype=np.float64)
