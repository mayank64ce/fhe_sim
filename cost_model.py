from __future__ import annotations
import math
from dataclasses import dataclass, field
from .arch_params import ArchParam
from .hw_model import Cost, poly_ntt, limb_bytes
from .op_model import (
    op_eval_add, op_eval_mult_ctct, op_eval_mult_ctpt, op_eval_rotate,
)

from .types import OpType, FHEOp
from .config import FHEConfig


@dataclass
class PredictionResult:
    predicted_latency_s: float
    compute_time_s:      float
    memory_time_s:       float
    bottleneck:          str          # "compute" | "memory"
    total_cost:          Cost
    per_op_costs:        list = field(default_factory=list)  # [(OpType, level, Cost)]

    def __str__(self) -> str:
        lines = [
            "=== fhe_sim Prediction ===",
            f"Predicted latency : {self.predicted_latency_s * 1e3:.3f} ms",
            f"  Compute time    : {self.compute_time_s * 1e3:.3f} ms",
            f"  Memory time     : {self.memory_time_s * 1e3:.3f} ms",
            f"  Bottleneck      : {self.bottleneck}",
            "",
            "--- Total hardware cost ---",
            f"  Add cycles      : {self.total_cost.add_cyc:>20,.0f}",
            f"  Mult cycles     : {self.total_cost.mult_cyc:>20,.0f}",
            f"  Auto cycles     : {self.total_cost.auto_cyc:>20,.0f}",
            f"  NTT cycles      : {self.total_cost.ntt_cyc:>20,.0f}",
            f"  DRAM reads      : {self.total_cost.dram_rd / 1e6:>20.1f} MB",
            f"  DRAM writes     : {self.total_cost.dram_wr / 1e6:>20.1f} MB",
            f"  Key DRAM reads  : {self.total_cost.dram_key_rd / 1e6:>20.1f} MB",
        ]
        return "\n".join(lines)


class CostModel:
    def __init__(self, arch: ArchParam):
        self.arch = arch

    def predict(self, op_log: list, config: FHEConfig) -> PredictionResult:
        N    = config.ring_dimension
        logq = config.scale_mod_size
        K    = config.special_primes
        dnum = config.num_large_digits

        total     = Cost()
        per_op    = []

        for fhe_op in op_log:
            l = max(fhe_op.level, 1)
            c = self._cost_for_op(fhe_op.op_type, N, l, K, dnum, logq)
            total   += c
            per_op.append((fhe_op.op_type, fhe_op.level, c))

        return self._to_result(total, per_op)

    def _cost_for_op(self, op_type, N, l, K, dnum, logq) -> Cost:
        a = self.arch
        if op_type in (OpType.EVAL_ADD_CTCT, OpType.EVAL_ADD_CTPT,
                       OpType.EVAL_ADD_INPLACE):
            return op_eval_add(N, l, logq, a)
        if op_type == OpType.EVAL_MULT_CTCT:
            return op_eval_mult_ctct(N, l, K, dnum, logq, a)
        if op_type == OpType.EVAL_MULT_CTPT:
            return op_eval_mult_ctpt(N, l, logq, a)
        if op_type == OpType.EVAL_ROTATE:
            return op_eval_rotate(N, l, K, dnum, logq, a)
        if op_type == OpType.EVAL_BOOTSTRAP:
            return self._cost_bootstrap(N, l, K, dnum, logq)
        if op_type == OpType.MAKE_PACKED_PLAINTEXT:
            logN = int(math.log2(N))
            return poly_ntt(N, l, logN, a)
        return Cost()

    def _cost_bootstrap(self, N, l, K, dnum, logq) -> Cost:
        """Approximate: 2 linear transforms (O(logN) rotations each) + 5 EvalMult chain."""
        c = Cost()
        n_rotations = 2 * int(math.log2(max(N // 2, 2)))
        for _ in range(2 * n_rotations):
            c += op_eval_rotate(N, l, K, dnum, logq, self.arch)
        for _ in range(5):
            c += op_eval_mult_ctct(N, l, K, dnum, logq, self.arch)
        return c

    def _to_result(self, total: Cost, per_op: list) -> PredictionResult:
        a = self.arch
        total_cyc     = total.total_compute_cycles
        compute_time  = total_cyc / (a.clock_freq_GHz * 1e9)
        total_dram    = total.dram_rd + total.dram_wr + total.dram_key_rd
        memory_time   = total_dram / (a.memory_bandwidth_GBps * 1e9)
        # CPU model: compute + memory stalls + per-operation-type overheads
        ROTATE_OPS = {OpType.EVAL_ROTATE}
        MULT_OPS   = {OpType.EVAL_MULT_CTCT, OpType.EVAL_MULT_CTPT, OpType.EVAL_BOOTSTRAP}
        ADD_OPS    = {OpType.EVAL_ADD_CTCT, OpType.EVAL_ADD_CTPT, OpType.EVAL_ADD_INPLACE}
        n_rotate = sum(1 for op_type, _, _ in per_op if op_type in ROTATE_OPS)
        n_mult   = sum(1 for op_type, _, _ in per_op if op_type in MULT_OPS)
        n_add    = sum(1 for op_type, _, _ in per_op if op_type in ADD_OPS)
        overhead = (n_rotate * a.per_rotate_overhead_s
                  + n_mult   * a.per_mult_overhead_s
                  + n_add    * a.per_add_overhead_s)
        latency  = compute_time + memory_time + overhead
        bottleneck    = "compute" if compute_time >= memory_time else "memory"
        return PredictionResult(
            predicted_latency_s = latency,
            compute_time_s      = compute_time,
            memory_time_s       = memory_time,
            bottleneck          = bottleneck,
            total_cost          = total,
            per_op_costs        = per_op,
        )
