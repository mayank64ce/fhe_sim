"""
Numerical accuracy simulator for CKKS programs.

Executes the same eval() C++ code on plaintext float vectors (numpy arrays),
producing predicted outputs that can be compared to expected outputs.
This is the CKKS analog of Concrete ML's simulate mode for TFHE.

No encryption, no polynomials — just slot-level arithmetic + optional noise.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .types import FHEType, FHEOp, OpType, FHE_METHOD_RETURN_TYPE
from .interpreter import (
    Interpreter, ExecContext, extract_member_types,
    _make_parser, _text, _fhe_type_from_cpp,
)
from .config import FHEConfig


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AccuracyResult:
    """Accuracy metrics from numerical simulation."""

    predicted_output: np.ndarray
    expected_output:  Optional[np.ndarray] = None

    # Populated when expected_output is provided
    mse:              Optional[float] = None
    mae:              Optional[float] = None
    max_error:        Optional[float] = None
    correct_ratio:    Optional[float] = None   # fraction of slots within threshold
    threshold:        float = 0.001            # per-slot correctness threshold

    def __str__(self) -> str:
        lines = ["=== Accuracy Prediction ==="]
        lines.append(f"Output slots     : {len(self.predicted_output)}")
        if self.expected_output is not None:
            lines.append(f"MSE              : {self.mse:.6e}")
            lines.append(f"MAE              : {self.mae:.6e}")
            lines.append(f"Max error        : {self.max_error:.6e}")
            lines.append(
                f"Correct slots    : {self.correct_ratio:.4%}"
                f"  (threshold={self.threshold})"
            )
        return "\n".join(lines)

    @staticmethod
    def compute(
        predicted: np.ndarray,
        expected: np.ndarray,
        threshold: float = 0.001,
    ) -> "AccuracyResult":
        err = np.abs(predicted - expected)
        return AccuracyResult(
            predicted_output = predicted,
            expected_output  = expected,
            mse              = float(np.mean((predicted - expected) ** 2)),
            mae              = float(np.mean(err)),
            max_error        = float(np.max(err)),
            correct_ratio    = float(np.mean(err < threshold)),
            threshold        = threshold,
        )


# ---------------------------------------------------------------------------
# Numerical interpreter
# ---------------------------------------------------------------------------

class NumericalInterpreter(Interpreter):
    """
    Subclass of Interpreter that executes FHE ops on numpy slot vectors.

    Instead of producing only an op log, it also computes actual output values
    by interpreting each FHE call as a numpy operation on the plaintext slots.
    """

    def __init__(
        self,
        member_types:           dict[str, FHEType],
        initial_level:          int,
        levels_after_bootstrap: int,
        input_slots:            np.ndarray,
        scale_mod_size:         int = 50,
        noise_budget_bits:      float = 0.0,
    ):
        super().__init__(member_types, initial_level, levels_after_bootstrap)
        self._input_slots = np.asarray(input_slots, dtype=np.float64)
        self._num_slots = len(input_slots)
        self._scale_mod_size = scale_mod_size

        # Noise injection: sigma per rescale ≈ sqrt(N_ring) / 2^scale_mod_size
        # If noise_budget_bits > 0, use that as override for sigma = 2^(-noise_budget_bits)
        self._noise_sigma = (2.0 ** (-noise_budget_bits)
                             if noise_budget_bits > 0 else 0.0)

        # Will hold the output after run()
        self.output_slots: Optional[np.ndarray] = None

    # -- Override: store values for ALL types, not just PLAIN ----------------

    def _should_store_value(self, fhe_t: FHEType, val) -> bool:
        return val is not None

    # -- Override: run() seeds m_InputC and extracts m_OutputC ---------------

    def run(self, eval_src: str) -> list[FHEOp]:
        parser = _make_parser()
        tree = parser.parse(eval_src.encode())

        body = self._find_eval_body(tree.root_node)
        if body is None:
            raise ValueError("Could not find eval() function definition.")

        ctx = ExecContext(
            env       = {"m_InputC": self._input_slots.copy()},
            type_env  = dict(self._global_type_env),
            level_env = dict(self._global_level_env),
        )
        self._exec_compound(body, ctx)

        self.output_slots = ctx.env.get("m_OutputC")
        return self.op_log

    # -- Override: execute FHE calls numerically ----------------------------

    def _eval_fhe_call(self, method, args_node, ctx):
        arg_results = self._eval_arg_list(args_node, ctx)
        arg_vals   = [v for v, _, _ in arg_results]
        arg_types  = [ft for _, ft, _ in arg_results]
        arg_levels = [lvl for _, ft, lvl in arg_results
                      if ft == FHEType.CIPHERTEXT and lvl is not None]

        input_level = min(arg_levels) if arg_levels else self._initial_level

        # Record op for the latency branch
        op = self._classify_fhe_op(method, arg_types)
        if op is not None:
            self.op_log.append(FHEOp(op_type=op, level=input_level))

        return_type = FHE_METHOD_RETURN_TYPE.get(method, FHEType.CIPHERTEXT)

        # Compute output level (same logic as parent)
        if return_type == FHEType.CIPHERTEXT:
            if method == "EvalMult":
                output_level = max(0, input_level - 1)
            elif method == "EvalBootstrap":
                output_level = self._levels_after_bootstrap
            else:
                output_level = input_level
        else:
            output_level = None

        # Numerical computation
        result = self._compute_numerical(method, arg_vals, arg_types)

        return result, return_type, output_level

    def _compute_numerical(self, method: str, arg_vals: list,
                           arg_types: list[FHEType]):
        if method == "EvalAdd":
            return self._np_binop(arg_vals, lambda a, b: a + b)

        if method == "EvalSub":
            return self._np_binop(arg_vals, lambda a, b: a - b)

        if method in ("EvalAddInPlace",):
            return self._np_binop(arg_vals, lambda a, b: a + b)

        if method in ("EvalSubInPlace",):
            return self._np_binop(arg_vals, lambda a, b: a - b)

        if method == "EvalMult":
            result = self._np_binop(arg_vals, lambda a, b: a * b)
            if result is not None and self._noise_sigma > 0:
                result = result + np.random.normal(
                    0, self._noise_sigma, result.shape
                )
            return result

        if method == "EvalNegate":
            a = arg_vals[0] if arg_vals else None
            return -a if a is not None else None

        if method == "EvalRotate":
            ct_val = arg_vals[0] if arg_vals else None
            k = arg_vals[1] if len(arg_vals) > 1 else None
            if isinstance(ct_val, np.ndarray) and k is not None:
                return np.roll(ct_val, -int(k))
            return ct_val

        if method == "EvalBootstrap":
            return arg_vals[0] if arg_vals else None

        if method in ("EvalSquare",):
            a = arg_vals[0] if arg_vals else None
            if a is not None:
                result = a * a
                if self._noise_sigma > 0 and isinstance(result, np.ndarray):
                    result = result + np.random.normal(
                        0, self._noise_sigma, result.shape
                    )
                return result
            return None

        if method == "EvalChebyshevSeries":
            # Can't simulate without knowing the coefficients at AST level
            # Return input unchanged as fallback
            return arg_vals[0] if arg_vals else None

        # MakeCKKSPackedPlaintext etc. — return None (plaintext encoding)
        return None

    @staticmethod
    def _np_binop(arg_vals: list, op):
        if len(arg_vals) < 2:
            return None
        a, b = arg_vals[0], arg_vals[1]
        if a is None or b is None:
            return None
        return op(a, b)
