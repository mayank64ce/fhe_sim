"""
Primitive-level decomposition of CKKS FHE operations.

Derived directly from the OpenFHE source code (v1.2.4):
  - keyswitch-hybrid.cpp   : EvalKeySwitchPrecomputeCore, EvalFastKeySwitchCoreExt
  - dcrtpoly-impl.h        : ApproxModDown, DropLastElementAndScale

All counts are in units of N-element polynomial operations:
  n_ntt     : number of size-N NTTs  (each costs N × log2(N) butterfly work)
  n_mul     : number of N-element polynomial mul_mod operations
  n_add     : number of N-element polynomial add_mod operations
  n_shuffle : number of N-element automorphism permutations
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .types import OpType

if TYPE_CHECKING:
    from .config import FHEConfig


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class PrimitiveCounts:
    n_ntt:     int = 0
    n_mul:     int = 0
    n_add:     int = 0
    n_shuffle: int = 0

    def __iadd__(self, other: "PrimitiveCounts") -> "PrimitiveCounts":
        self.n_ntt     += other.n_ntt
        self.n_mul     += other.n_mul
        self.n_add     += other.n_add
        self.n_shuffle += other.n_shuffle
        return self

    def __add__(self, other: "PrimitiveCounts") -> "PrimitiveCounts":
        return PrimitiveCounts(
            n_ntt     = self.n_ntt     + other.n_ntt,
            n_mul     = self.n_mul     + other.n_mul,
            n_add     = self.n_add     + other.n_add,
            n_shuffle = self.n_shuffle + other.n_shuffle,
        )

    def __repr__(self) -> str:
        return (f"PrimitiveCounts(ntt={self.n_ntt}, mul={self.n_mul}, "
                f"add={self.n_add}, shuffle={self.n_shuffle})")


# ---------------------------------------------------------------------------
# Hybrid key switching
# keyswitch-hybrid.cpp: EvalKeySwitchPrecomputeCore + EvalFastKeySwitchCoreExt
#                       + ApproxModDown (dcrtpoly-impl.h)
# ---------------------------------------------------------------------------

def _key_switch(l: int, K: int, dnum: int) -> PrimitiveCounts:
    """
    Primitive counts for one Hybrid RLWE key switch at level l.

    Parameters
    ----------
    l    : ciphertext level (number of data RNS primes)
    K    : number of special RNS primes  (config.special_primes)
    dnum : number of key-switch decomposition digits  (config.num_large_digits)

    Source map
    ----------
    BConv-Up   → EvalKeySwitchPrecomputeCore, keyswitch-hybrid.cpp:350-409
    Key-mul    → EvalFastKeySwitchCoreExt,    keyswitch-hybrid.cpp:455-475
    ModDown×2  → ApproxModDown,               dcrtpoly-impl.h:988-1027
                 (called for ct0 and ct1 in EvalFastKeySwitchCore:423-433)
    """
    l    = max(l, 1)
    K    = max(K, 1)
    dnum = max(dnum, 1)

    alpha = math.ceil(l / dnum)   # source primes per digit
    comp  = l + K - alpha          # complement primes per digit (BConv target)

    # ── BConv-Up (per digit × dnum) ────────────────────────────────────────
    # keyswitch-hybrid.cpp:384  partCtClone.SetFormat(COEFFICIENT) → alpha iNTTs
    # keyswitch-hybrid.cpp:394  partsCtCompl.SetFormat(EVALUATION) → comp  NTTs
    ntt_bconv_up = dnum * (alpha + comp)          # = dnum * (l + K)
    # ApproxSwitchCRTBasis: comp target primes, each sums over alpha source primes
    mul_bconv_up = dnum * comp * alpha
    add_bconv_up = dnum * comp * max(0, alpha - 1)

    # ── Key multiplication (EvalFastKeySwitchCoreExt:455-475) ───────────────
    # All in EVALUATION domain — no NTTs.
    # 2 output polys (cTilda0, cTilda1) × dnum digits × (l+K) primes each.
    # j=0 accumulates into a zero-initialised poly → treat first digit as copy.
    mul_keymul = 2 * dnum * (l + K)
    add_keymul = 2 * max(0, dnum - 1) * (l + K)

    # ── ModDown × 2 (ApproxModDown:988-1027, called for ct0 and ct1) ───────
    # Per call:
    #   line 1002  partP.SetFormat(COEFFICIENT)     → K  iNTTs on P part
    #   line 1023  partPSwitched.SetFormat(EVAL)     → l  NTTs on switched result
    #   line 1010  ApproxSwitchCRTBasis(K → l)       → l*K muls + l*(K-1) adds
    #   line 1024  (Q_part - switched) * PInvModq    → l  muls + l  adds
    ntt_moddown = 2 * (K + l)
    mul_moddown = 2 * (l * K + l)       # BConv: l*K;  combine: l   (per call)
    add_moddown = 2 * l * K             # BConv: l*(K-1); combine: l → l*K per call

    return PrimitiveCounts(
        n_ntt = ntt_bconv_up + ntt_moddown,
        n_mul = mul_bconv_up + mul_keymul + mul_moddown,
        n_add = add_bconv_up + add_keymul + add_moddown,
    )


# ---------------------------------------------------------------------------
# Rescale
# ckksrns-leveledshe.cpp: ModReduceInternalInPlace → DropLastElementAndScale
# dcrtpoly-impl.h:715-733
# ---------------------------------------------------------------------------

def _rescale(l: int) -> PrimitiveCounts:
    """
    Primitive counts for one CKKS rescale (level l → l-1).

    Called on both ciphertext elements (cv[0] and cv[1]).

    Per element (DropLastElementAndScale:715-733):
      line 718  lastPoly.SetFormat(COEFFICIENT)        → 1 iNTT  (top prime)
      line 728  tmp.SwitchFormat() [if EVALUATION]     → (l-1) NTTs  (remaining primes)
      line 726  tmp *= QlQlInvModqlDivqlModq[i]        → (l-1) muls
      line 729  m_vectors[i] *= qlInvModq[i]           → (l-1) muls
      line 730  m_vectors[i] += tmp                    → (l-1) adds
    """
    l = max(l, 1)
    rem = max(0, l - 1)   # remaining primes after dropping the top one
    return PrimitiveCounts(
        n_ntt = 2 * l,      # 2 elements × (1 iNTT + rem NTTs)
        n_mul = 4 * rem,    # 2 elements × 2*rem scalar muls
        n_add = 2 * rem,    # 2 elements × rem adds
    )


# ---------------------------------------------------------------------------
# High-level operation decomposition
# ---------------------------------------------------------------------------

def decompose_op(op_type: OpType, level: int, K: int, dnum: int) -> PrimitiveCounts:
    """
    Decompose a high-level FHE operation into primitive counts.

    op_type : the operation type
    level   : input ciphertext level (number of RNS data primes at call time)
    K       : number of special primes  (config.special_primes)
    dnum    : decomposition digits      (config.num_large_digits)
    """
    l = max(level, 1)

    if op_type == OpType.EVAL_MULT_CTCT:
        # base-leveledshe.cpp:628-631  EvalMultCore (n1=n2=2 fast path)
        #   cvr[0]  = cv1[0] * cv2[0]                   → 1 DCRTPoly mul
        #   cvr[1]  = (cv1[0]*cv2[1]) += (cv1[1]*cv2[0]) → 2 DCRTPoly muls + 1 add
        #   cvr[2]  = cv1[1] * cv2[1]                   → 1 DCRTPoly mul
        # Each DCRTPoly op acts on l primes → multiply counts by l
        pc = PrimitiveCounts(n_mul=4 * l, n_add=1 * l)
        # Relinearisation: key switch on cvr[2]
        pc += _key_switch(l, K, dnum)
        # Add key-switch result back into (cvr[0], cvr[1]): 2 DCRTPoly adds × l primes
        pc.n_add += 2 * l
        # Rescale (FIXEDAUTO applies automatically after EvalMult)
        pc += _rescale(l)
        return pc

    if op_type == OpType.EVAL_MULT_CTPT:
        # rns-leveledshe.cpp:226-243
        # c[0] *= pt,  c[1] *= pt   → 2 DCRTPoly muls × l primes
        pc = PrimitiveCounts(n_mul=2 * l)
        pc += _rescale(l)
        return pc

    if op_type == OpType.EVAL_ROTATE:
        # base-leveledshe.cpp:420-463  (EvalFastRotationExt path)
        # AutomorphismTransform on both (c0, c1): 2 × l shuffles
        # Key switch on σ(c1)
        # Add key-switch result into σ(c0): l adds
        pc = PrimitiveCounts(n_shuffle=2 * l)
        pc += _key_switch(l, K, dnum)
        pc.n_add += l
        return pc

    if op_type in (OpType.EVAL_ADD_CTCT, OpType.EVAL_ADD_CTPT,
                   OpType.EVAL_ADD_INPLACE):
        # base-leveledshe.cpp:570-586  EvalAddCoreInPlace
        # cv1[0] += cv2[0],  cv1[1] += cv2[1]  → 2 DCRTPoly adds × l primes
        return PrimitiveCounts(n_add=2 * l)

    if op_type == OpType.EVAL_BOOTSTRAP:
        # Approximate model: Slot2Coeff + poly approx + Coeff2Slot
        # Each linear transform ≈ O(log N) rotations; poly approx ≈ depth EvalMult chain
        approx_degree = 5
        ks_count = 2 * int(math.log2(max(l, 2)))
        pc = PrimitiveCounts()
        for _ in range(2 * ks_count):
            pc += _key_switch(l, K, dnum)
            pc.n_shuffle += l
        for _ in range(approx_degree):
            pc += decompose_op(OpType.EVAL_MULT_CTCT, l, K, dnum)
        return pc

    if op_type == OpType.MAKE_PACKED_PLAINTEXT:
        # One NTT per RNS prime for plaintext encoding
        return PrimitiveCounts(n_ntt=l)

    return PrimitiveCounts()


# ---------------------------------------------------------------------------
# Accumulate over an entire operation log
# ---------------------------------------------------------------------------

def total_primitive_counts(op_log: list, config: "FHEConfig") -> PrimitiveCounts:
    """Sum primitive counts over all FHEOps, using level-aware key-switch params."""
    from .config import get_ks_params

    total = PrimitiveCounts()
    for op in op_log:
        K, dnum = get_ks_params(config, op.level)
        total += decompose_op(op.op_type, op.level, K, dnum)
    return total
