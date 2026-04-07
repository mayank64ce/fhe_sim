"""
op_model.py — FHE operation cost model.

Port of SimFHE/evaluator.py. Each public function takes
  (N, l, K, dnum, logq, arch, rd_in=True, wr_out=True)
and returns a Cost object with cycles and DRAM bytes accumulated explicitly.

Notation
--------
N     : ring dimension (polynomial degree)
l     : number of Q-basis RNS limbs at current level (≥ 1)
K     : number of special P-basis primes (from config.special_primes)
dnum  : key-switch decomposition digit count (config.num_large_digits)
logq  : bits per RNS prime (config.scale_mod_size)
alpha : limbs per digit = ceil(l / dnum)
pq    : total PQ-basis limbs = l + K
"""
from __future__ import annotations
import math
from .arch_params import ArchParam, CacheStyle
from .hw_model import (
    Cost, limb_bytes, poly_add, poly_mult, poly_ntt, poly_automorph,
    basis_convert,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _limb_sz(N: int, limbs: int, logq: int) -> int:
    """Alias for limb_bytes — byte size of a polynomial with `limbs` RNS primes."""
    return limb_bytes(N, limbs, logq)


def _alpha(l: int, dnum: int) -> int:
    """Number of limbs per key-switch digit."""
    return math.ceil(l / max(dnum, 1))


# ── primitive operations ────────────────────────────────────────────────────

def _op_mod_raise(
    N: int,
    in_limbs: int,
    out_limbs: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
) -> Cost:
    """
    Raise modulus of one polynomial from in_limbs → out_limbs primes.

    Pipeline (SimFHE evaluator.mod_raise):
      1. iNTT over in_limbs
      2. [cache_style < ALPHA] write limbs out + read back slot-wise for BConv
      3. BConv from in_limbs → delta_limbs (= out_limbs - in_limbs)
      4. [cache_style < ALPHA] write delta limbs out + read back limb-wise for NTT
      5. NTT over delta_limbs
      6. [wr_out] write out delta_limbs

    DRAM is only charged for the delta (new) limbs from NTT step onward,
    since the old in_limbs are assumed to remain in the pipeline from rd_in.
    """
    c = Cost()
    logN       = int(math.log2(N))
    delta_limbs = out_limbs - in_limbs   # new limbs produced by BConv (K primes)

    if rd_in:
        c.dram_rd += _limb_sz(N, in_limbs, logq)

    # Step 1: iNTT over in_limbs
    c += poly_ntt(N, in_limbs, logN, arch)

    # Step 2: intermediate DRAM round-trips if cache too small
    if arch.cache_style < CacheStyle.ALPHA:
        c.dram_wr += _limb_sz(N, in_limbs, logq)
        c.dram_rd += _limb_sz(N, in_limbs, logq)

    # Step 3: basis conversion in_limbs → delta_limbs
    c += basis_convert(N, in_limbs, delta_limbs, arch)

    # Step 4: intermediate DRAM round-trips for delta limbs
    if arch.cache_style < CacheStyle.ALPHA:
        c.dram_wr += _limb_sz(N, delta_limbs, logq)
        c.dram_rd += _limb_sz(N, delta_limbs, logq)

    # Step 5: NTT over delta_limbs
    c += poly_ntt(N, delta_limbs, logN, arch)

    # Step 6: write out new limbs
    if wr_out:
        c.dram_wr += _limb_sz(N, delta_limbs, logq)

    return c


def _op_mod_down_reduce(
    N: int,
    pq_limbs: int,
    q_limbs: int,
    logq: int,
    arch: ArchParam,
) -> Cost:
    """
    Given x in PQ basis, reduce the P part and bring it into Q basis.

    Pipeline (SimFHE evaluator.mod_down_reduce):
      1. iNTT over p_limbs (= pq_limbs - q_limbs)
      2. [cache_style < ALPHA] write + read for BConv
      3. BConv from p_limbs → q_limbs
      4. [cache_style < ALPHA] write + read for NTT
      5. NTT over q_limbs

    No rd_in / wr_out — those are handled by _op_mod_down.
    """
    c = Cost()
    logN    = int(math.log2(N))
    p_limbs = pq_limbs - q_limbs   # number of special P primes

    # Step 1: iNTT over P limbs
    c += poly_ntt(N, p_limbs, logN, arch)

    # Step 2: intermediate round-trips
    if arch.cache_style < CacheStyle.ALPHA:
        c.dram_wr += _limb_sz(N, p_limbs, logq)
        c.dram_rd += _limb_sz(N, p_limbs, logq)

    # Step 3: BConv P → Q
    c += basis_convert(N, p_limbs, q_limbs, arch)

    # Step 4: intermediate round-trips for Q output
    if arch.cache_style < CacheStyle.ALPHA:
        c.dram_wr += _limb_sz(N, q_limbs, logq)
        c.dram_rd += _limb_sz(N, q_limbs, logq)

    # Step 5: NTT over Q limbs to return to eval representation
    c += poly_ntt(N, q_limbs, logN, arch)

    return c


def _op_mod_down_divide(N: int, q_limbs: int, arch: ArchParam) -> Cost:
    """
    Given x mod Q and BConv(x mod P) both in Q basis, compute x/P in Q basis.

    One poly_add + one poly_mult over q_limbs.
    (SimFHE evaluator.mod_down_divide)
    """
    c = Cost()
    c += poly_add(N, q_limbs, arch)
    c += poly_mult(N, q_limbs, arch)
    return c


def _op_mod_down(
    N: int,
    pq_limbs: int,
    q_limbs: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
) -> Cost:
    """
    Full mod_down: x in PQ basis → x/P in Q basis.

    (SimFHE evaluator.mod_down)
    """
    c = Cost()
    p_limbs = pq_limbs - q_limbs

    if rd_in:
        # Read the P limbs to be reduced
        c.dram_rd += _limb_sz(N, p_limbs, logq)

    c += _op_mod_down_reduce(N, pq_limbs, q_limbs, logq, arch)

    if rd_in:
        # Read the original Q limbs to complete the divide step
        c.dram_rd += _limb_sz(N, q_limbs, logq)

    c += _op_mod_down_divide(N, q_limbs, arch)

    if wr_out:
        c.dram_wr += _limb_sz(N, q_limbs, logq)

    return c


def _op_rescale(
    N: int,
    l: int,
    logq: int,
    logN: int,
    arch: ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
) -> Cost:
    """
    Rescale by dropping the last limb: l limbs → l-1 limbs.

    Pipeline (SimFHE evaluator.mod_reduce_rescale):
      - iNTT on the 1 limb being dropped
      - NTT on the remaining (l-1) limbs
      - (l-1) poly_mult + (l-1) poly_add for the correction step
      - rd_in reads the 1 dropped limb; wr_out writes the (l-1) output limbs
    """
    c = Cost()
    out_limbs = l - 1

    if rd_in:
        c.dram_rd += _limb_sz(N, 1, logq)      # read the single limb being dropped

    # iNTT on 1 limb
    c += poly_ntt(N, 1, logN, arch)

    # NTT on remaining limbs (to bring correction into eval rep)
    c += poly_ntt(N, out_limbs, logN, arch)

    # Correction: out_limbs mults + out_limbs adds
    c += poly_mult(N, out_limbs, arch)
    c += poly_add(N, out_limbs, arch)

    if rd_in:
        c.dram_rd += _limb_sz(N, out_limbs, logq)  # read output limbs for correction

    if wr_out:
        c.dram_wr += _limb_sz(N, out_limbs, logq)

    return c


def _op_key_switch_hoisting(
    N: int,
    l: int,
    K: int,
    dnum: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = False,
    wr_out: bool = True,
) -> Cost:
    """
    Key-switch hoisting: decompose one polynomial into dnum alpha-limb digits
    and mod-raise each digit from Q → PQ basis.

    (SimFHE evaluator.key_switch_hoisting)
    Each digit has alpha = ceil(l/dnum) Q limbs and is raised to pq = alpha+K limbs.
    """
    c = Cost()
    alpha    = _alpha(l, dnum)
    pq_limbs = alpha + K          # digit context after mod-raise

    if rd_in:
        c.dram_rd += _limb_sz(N, l, logq)   # read the full Q polynomial

    for _ in range(dnum):
        # mod_raise for each digit: alpha limbs → alpha+K limbs
        # rd_in=False because hoisting feeds directly from decompose
        # wr_out: write out the new K limbs (delta) per digit
        c += _op_mod_raise(N, alpha, pq_limbs, logq, arch,
                           rd_in=False, wr_out=wr_out)

    return c


def _op_key_switch_inner_product(
    N: int,
    l: int,
    K: int,
    dnum: int,
    logq: int,
    arch: ArchParam,
    automorph: bool = False,
    rd_in: bool = True,
    wr_out: bool = False,
    key_cached: bool = False,
) -> Cost:
    """
    Key-switch inner product: multiply each mod-raised digit by its key polynomial.

    For each of dnum digits:
      - [rd_in]   read the pq-limb digit polynomial
      - [automorph] apply automorphism to digit
      - read key polynomial from DRAM (unless key_cached)
      - 2 × poly_mult (multiply_plain: 2 mults for 2-poly ciphertext key)
      - poly_add accumulate (skip first iteration)

    (SimFHE evaluator.key_switch_inner_product)
    """
    c = Cost()
    alpha    = _alpha(l, dnum)
    pq_limbs = alpha + K
    key_sz   = _limb_sz(N, pq_limbs, logq)

    for i in range(dnum):
        if rd_in:
            c.dram_rd += key_sz

        if automorph:
            c += poly_automorph(N, pq_limbs, arch)

        # Key read: key_factor = 2 (two polys per key-switch key) unless compressed
        if not key_cached:
            c.dram_key_rd += arch.key_factor() * key_sz

        # multiply_plain: 2 mults (multiply both ciphertext polys by key poly)
        c += poly_mult(N, pq_limbs, arch)
        c += poly_mult(N, pq_limbs, arch)

        # accumulate: add to running sum (skip i=0)
        if i != 0:
            c += poly_add(N, pq_limbs, arch)
            c += poly_add(N, pq_limbs, arch)

    if wr_out:
        # Write out the accumulated 2-poly result
        c.dram_wr += 2 * key_sz

    return c


def _op_key_switch(
    N: int,
    l: int,
    K: int,
    dnum: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = False,
    wr_out: bool = False,
) -> Cost:
    """
    Full hybrid key switch (Han and Ki '19).

    Pipeline:
      1. hoisting: decompose + mod-raise all digits
      2. inner_product: multiply raised digits by key, accumulate
      3. 2 × mod_down (PQ → Q) for the two result polynomials
      4. poly_add fixup with original b polynomial

    DRAM control:
      limb_rdwr = cache_style < CONST
      (if no constant cache, inner product result must be written to DRAM and
       mod_down must re-read it)

    (SimFHE evaluator.key_switch)
    """
    c = Cost()
    pq_limbs  = l + K
    limb_rdwr = arch.cache_style < CacheStyle.CONST

    # Step 1: hoisting
    c += _op_key_switch_hoisting(N, l, K, dnum, logq, arch,
                                 rd_in=rd_in, wr_out=True)

    # Step 2: inner product
    c += _op_key_switch_inner_product(N, l, K, dnum, logq, arch,
                                      automorph=False,
                                      rd_in=True,
                                      wr_out=limb_rdwr,
                                      key_cached=False)

    # Step 3: two mod_downs (for both polynomials in the key-switch output)
    # First mod_down writes output; second does not (b fixup happens after)
    c += _op_mod_down(N, pq_limbs, l, logq, arch,
                      rd_in=limb_rdwr, wr_out=wr_out)
    c += _op_mod_down(N, pq_limbs, l, logq, arch,
                      rd_in=limb_rdwr, wr_out=False)

    # Step 4: read the original b-polynomial and add it in for fixup
    if rd_in:
        c.dram_rd += _limb_sz(N, l, logq)
    c += poly_add(N, l, arch)

    if wr_out:
        c.dram_wr += _limb_sz(N, l, logq)

    return c


def _partial_multiply_inner(c: Cost, N: int, l: int, arch: ArchParam) -> None:
    """
    Compute (a0*b1 + a1*b0,  b0*b1) from (a0, a1) and (b0, b1).
    Modifies `c` in place.

    karatsuba=False: 3 mults + 1 add
      b0*b1, a0*b1, a1*b0 → add the two cross terms
    karatsuba=True:  2 mults + 4 adds
      m1 = b0*b1
      (a0+b0)*(a1+b1) = a0a1 + cross + b0b1
      subtract m1 and a0a1 to isolate cross

    (SimFHE evaluator.partial_multiply_inner, non-squaring branch)
    """
    if arch.karatsuba:
        c += poly_mult(N, l, arch)   # b_0 * b_1
        c += poly_add(N, l, arch)    # a_0 + b_0
        c += poly_add(N, l, arch)    # a_1 + b_1
        c += poly_mult(N, l, arch)   # (a_0 + b_0)(a_1 + b_1)
        c += poly_add(N, l, arch)    # subtract a_0*a_1 term
        c += poly_add(N, l, arch)    # subtract b_0*b_1 term
    else:
        c += poly_mult(N, l, arch)   # b_0 * b_1
        c += poly_mult(N, l, arch)   # a_0 * b_1
        c += poly_mult(N, l, arch)   # a_1 * b_0
        c += poly_add(N, l, arch)    # a_0*b_1 + a_1*b_0


# ── public operation functions ──────────────────────────────────────────────

def op_eval_add(
    N: int,
    l: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = True,
    wr_out: bool = True,
) -> Cost:
    """
    EvalAdd (ciphertext + ciphertext or ciphertext + plaintext).

    2 × poly_add (one for each polynomial in the ciphertext pair).
    DRAM: reads 4 polynomials (2 per ct × 2 cts), writes 2 polynomials.

    (SimFHE evaluator.add)
    """
    c = Cost()

    if rd_in:
        # 4 polynomials: (a0, b0) from ct1 + (a1, b1) from ct2
        c.dram_rd += 4 * _limb_sz(N, l, logq)

    c += poly_add(N, l, arch)
    c += poly_add(N, l, arch)

    if wr_out:
        c.dram_wr += 2 * _limb_sz(N, l, logq)

    return c


def op_eval_mult_ctpt(
    N: int,
    l: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = True,
    wr_out: bool = True,
) -> Cost:
    """
    EvalMult with a plaintext: multiply each ciphertext polynomial by plaintext.

    2 × poly_mult + 2 × rescale (one rescale per ct polynomial).
    DRAM: reads 3 polynomials (2 ct + 1 pt), writes 2 polynomials.

    (SimFHE evaluator.multiply_plain + mod_reduce_rescale ×2)
    """
    c = Cost()
    logN = int(math.log2(N))

    if rd_in:
        # 2 ciphertext polys + 1 plaintext poly
        c.dram_rd += 3 * _limb_sz(N, l, logq)

    # multiply_plain: 2 mults (one per ct polynomial)
    c += poly_mult(N, l, arch)
    c += poly_mult(N, l, arch)

    # 2 rescales (drop one limb from each ct polynomial)
    c += _op_rescale(N, l, logq, logN, arch, rd_in=False, wr_out=False)
    c += _op_rescale(N, l, logq, logN, arch, rd_in=False, wr_out=False)

    if wr_out:
        c.dram_wr += 2 * _limb_sz(N, l - 1, logq)

    return c


def op_eval_mult_ctct(
    N: int,
    l: int,
    K: int,
    dnum: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = True,
    wr_out: bool = True,
) -> Cost:
    """
    EvalMult ciphertext × ciphertext (full pipeline).

    Pipeline:
      1. a0*a1 (key-switch input polynomial)
      2. key_switch_hoisting on a0*a1
      3. key_switch_inner_product
      4. Either fused or non-fused rescale path:
         Fused (rescale_fusion=True):
           - partial_multiply_inner on inputs
           - 1 poly_mult (scale by P)
           - 2 poly_add (accumulate with key-switch output)
           - 2 × mod_down (PQ → l-1 limbs directly)
         Non-fused (rescale_fusion=False):
           - 2 × mod_down (PQ → l)
           - partial_multiply_inner
           - 2 poly_add
           - 2 × rescale (l → l-1)

    DRAM control mirrors SimFHE evaluator.multiply:
      limb_rdwr    = cache_style < CONST
      reorder_rdwr = limb_rdwr  (we skip mod_down_reorder for simplicity)
      sq_mult      = 2          (non-squaring)

    (SimFHE evaluator.multiply, sqr=False)
    """
    c = Cost()
    logN      = int(math.log2(N))
    pq_limbs  = l + K
    sq_mult   = 2
    limb_rdwr    = arch.cache_style < CacheStyle.CONST
    reorder_rdwr = limb_rdwr   # simplified: skip mod_down_reorder

    # Step 1: compute a0*a1 (key-switch input)
    if rd_in:
        # read a0, a1 (2 polynomials × sq_mult=2 ct = 4 polys for non-square,
        # but a0*a1 only needs one polynomial pair from each ct)
        # SimFHE: rd_in reads sq_mult * poly_ctxt.size_in_bytes
        c.dram_rd += sq_mult * _limb_sz(N, l, logq)

    c += poly_mult(N, l, arch)   # a0 * a1

    if limb_rdwr:
        c.dram_wr += _limb_sz(N, l, logq)   # write out a0*a1

    # Step 2: hoisting on a0*a1
    c += _op_key_switch_hoisting(N, l, K, dnum, logq, arch,
                                 rd_in=limb_rdwr, wr_out=True)

    # Step 3: inner product
    c += _op_key_switch_inner_product(N, l, K, dnum, logq, arch,
                                      automorph=False,
                                      rd_in=True,
                                      wr_out=limb_rdwr,
                                      key_cached=False)

    if arch.rescale_fusion:
        # ── Fused path ─────────────────────────────────────────────────
        # Read in all 4 input polynomials (a0, a1, b0, b1)
        if rd_in:
            c.dram_rd += 2 * sq_mult * _limb_sz(N, l, logq)

        # partial_multiply_inner: computes (a0b1+a1b0, b0b1)
        _partial_multiply_inner(c, N, l, arch)

        # scale by P (one poly_mult to bring into the right scale)
        c += poly_mult(N, l, arch)

        # read inner product output and add
        if limb_rdwr:
            c.dram_rd += 2 * _limb_sz(N, l, logq)

        # 2 poly_add: accumulate partial_multiply result + inner_product result
        c += poly_add(N, l, arch)
        c += poly_add(N, l, arch)

        # fused mod_down: PQ → l-1 limbs (rescale combined into mod_down)
        out_limbs = l - 1
        c += _op_mod_down(N, pq_limbs, out_limbs, logq, arch,
                          rd_in=reorder_rdwr, wr_out=wr_out)
        c += _op_mod_down(N, pq_limbs, out_limbs, logq, arch,
                          rd_in=reorder_rdwr, wr_out=wr_out)

    else:
        # ── Non-fused path ──────────────────────────────────────────────
        # mod_down the inner product output (PQ → l)
        c += _op_mod_down(N, pq_limbs, l, logq, arch,
                          rd_in=reorder_rdwr, wr_out=limb_rdwr)
        c += _op_mod_down(N, pq_limbs, l, logq, arch,
                          rd_in=reorder_rdwr, wr_out=limb_rdwr)

        # Read in all 4 input polynomials for partial multiply
        if rd_in:
            c.dram_rd += 2 * sq_mult * _limb_sz(N, l, logq)

        # partial_multiply_inner: computes (a0b1+a1b0, b0b1)
        _partial_multiply_inner(c, N, l, arch)

        # Read mod_down outputs if they were written to DRAM
        if limb_rdwr:
            c.dram_rd += 2 * _limb_sz(N, l, logq)

        # 2 poly_add fixup
        c += poly_add(N, l, arch)
        c += poly_add(N, l, arch)

        # 2 rescales: l → l-1
        c += _op_rescale(N, l, logq, logN, arch,
                         rd_in=reorder_rdwr, wr_out=wr_out)
        c += _op_rescale(N, l, logq, logN, arch,
                         rd_in=reorder_rdwr, wr_out=wr_out)

    return c


def op_eval_rotate(
    N: int,
    l: int,
    K: int,
    dnum: int,
    logq: int,
    arch: ArchParam,
    rd_in: bool = True,
    wr_out: bool = True,
) -> Cost:
    """
    EvalRotate: automorphism + key switch.

    Pipeline:
      1. rotate_inner: 2 × poly_automorph (on a and b polynomials of ct)
         [rd_in / wr_out governed by limb_rdwr]
      2. key_switch on the rotated ciphertext

    DRAM control:
      limb_rdwr = cache_style < CONST
      (if no const cache, rotate_inner result must be written before key_switch)

    (SimFHE evaluator.rotate)
    """
    c = Cost()
    limb_rdwr = arch.cache_style < CacheStyle.CONST

    # rotate_inner: automorph both polynomials
    if rd_in:
        c.dram_rd += 2 * _limb_sz(N, l, logq)

    c += poly_automorph(N, l, arch)   # automorph a polynomial
    c += poly_automorph(N, l, arch)   # automorph b polynomial

    if limb_rdwr:
        c.dram_wr += 2 * _limb_sz(N, l, logq)

    # key_switch on the rotated a polynomial
    c += _op_key_switch(N, l, K, dnum, logq, arch,
                        rd_in=limb_rdwr, wr_out=wr_out)

    return c
