from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class FHEType(Enum):
    CIPHERTEXT = "ct"
    PLAINTEXT = "pt"
    PLAIN = "plain"      # non-FHE scalar / vector (int, double, ...)
    UNKNOWN = "unknown"


class OpType(Enum):
    EVAL_MULT_CTCT          = "EvalMult_ctct"
    EVAL_MULT_CTPT          = "EvalMult_ctpt"
    EVAL_ROTATE             = "EvalRotate"
    EVAL_ADD_CTCT           = "EvalAdd_ctct"
    EVAL_ADD_CTPT           = "EvalAdd_ctpt"
    EVAL_ADD_INPLACE        = "EvalAddInPlace"
    EVAL_BOOTSTRAP          = "EvalBootstrap"
    MAKE_PACKED_PLAINTEXT   = "MakeCKKSPackedPlaintext"


# Alias used in interpreter.py
FHEMethod = str

# Return type of each OpenFHE method
FHE_METHOD_RETURN_TYPE: dict[str, FHEType] = {
    "EvalMult":                 FHEType.CIPHERTEXT,
    "EvalAdd":                  FHEType.CIPHERTEXT,
    "EvalRotate":               FHEType.CIPHERTEXT,
    "EvalSub":                  FHEType.CIPHERTEXT,
    "EvalAddInPlace":           FHEType.CIPHERTEXT,
    "EvalSubInPlace":           FHEType.CIPHERTEXT,
    "EvalBootstrap":            FHEType.CIPHERTEXT,
    "EvalNegate":               FHEType.CIPHERTEXT,
    "EvalSquare":               FHEType.CIPHERTEXT,
    "EvalChebyshevSeries":      FHEType.CIPHERTEXT,
    "MakeCKKSPackedPlaintext":  FHEType.PLAINTEXT,
    "MakeBFVPackedPlaintext":   FHEType.PLAINTEXT,
    "MakePackedPlaintext":      FHEType.PLAINTEXT,
    "MakePlaintext":            FHEType.PLAINTEXT,
}


@dataclass
class FHEOp:
    op_type: OpType
    level: int = 0          # input ciphertext level when the op is called
    line: Optional[int] = None


@dataclass
class OpCount:
    counts: dict = field(default_factory=dict)

    def add(self, op: OpType, n: int = 1):
        self.counts[op] = self.counts.get(op, 0) + n

    def total(self) -> int:
        return sum(self.counts.values())

    def __repr__(self):
        lines = []
        for op in OpType:
            v = self.counts.get(op, 0)
            if v:
                lines.append(f"  {op.value}: {v}")
        return "\n".join(lines) if lines else "  (none)"
