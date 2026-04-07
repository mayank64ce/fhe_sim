from __future__ import annotations
import math
from dataclasses import dataclass
from .arch_params import ArchParam, CacheStyle


@dataclass
class Cost:
    """Accumulated hardware cost for a sequence of polynomial operations."""
    add_cyc:  float = 0.0
    mult_cyc: float = 0.0
    auto_cyc: float = 0.0   # automorphism (best-case model)
    ntt_cyc:  float = 0.0   # dedicated NTT unit (if dedicated_ntt_unit=True)

    dram_rd:      int = 0   # limb reads (bytes)
    dram_wr:      int = 0   # limb writes (bytes)
    dram_key_rd:  int = 0   # key-switch key reads (bytes)

    def __iadd__(self, other: "Cost") -> "Cost":
        self.add_cyc     += other.add_cyc
        self.mult_cyc    += other.mult_cyc
        self.auto_cyc    += other.auto_cyc
        self.ntt_cyc     += other.ntt_cyc
        self.dram_rd     += other.dram_rd
        self.dram_wr     += other.dram_wr
        self.dram_key_rd += other.dram_key_rd
        return self

    def __add__(self, other: "Cost") -> "Cost":
        c = Cost()
        c += self
        c += other
        return c

    @property
    def total_compute_cycles(self) -> float:
        return self.add_cyc + self.mult_cyc + self.auto_cyc + self.ntt_cyc

    @property
    def total_dram_bytes(self) -> int:
        return self.dram_rd + self.dram_wr + self.dram_key_rd


def limb_bytes(N: int, limbs: int, logq: int) -> int:
    """Byte size of one ciphertext polynomial with `limbs` RNS primes."""
    return N * limbs * logq // 8


def poly_add(N: int, limbs: int, arch: ArchParam) -> Cost:
    """One N-point polynomial addition over `limbs` RNS primes."""
    c = Cost()
    c.add_cyc = N * limbs / (arch.funits * arch.sets) + arch.add_lat
    return c


def poly_mult(N: int, limbs: int, arch: ArchParam) -> Cost:
    """One N-point polynomial multiplication over `limbs` RNS primes."""
    c = Cost()
    c.mult_cyc = N * limbs / (arch.funits * arch.sets) + arch.mult_lat
    return c


def poly_ntt(N: int, limbs: int, logN: int, arch: ArchParam) -> Cost:
    """
    One NTT or iNTT over `limbs` RNS primes.

    Ported from SimFHE poly.ntt → compute_phi + compute_tf + ntt_common.

    Two models selected by arch.dedicated_ntt_unit:
      False (default): butterfly model — NTT butterflies use standard add+mult units.
        phi generation:  N   × limbs muls
        tf  generation:  N/2 × limbs muls
        butterflies:     (N/2)×logN×limbs muls  +  N×logN×limbs adds
      True: dedicated NTT unit — ntt_cyc = (ntt_lat + N*limbs/(2*P)) * logN
    """
    c = Cost()
    P = arch.funits * arch.sets

    phi_muls   = N * limbs
    tf_muls    = (N // 2) * limbs
    btfly_muls = (N // 2) * logN * limbs
    btfly_adds = 2 * btfly_muls

    if arch.dedicated_ntt_unit:
        c.ntt_cyc  = (arch.ntt_lat + N * limbs / (2 * P)) * logN
        # phi and tf generation still uses standard mult units
        c.mult_cyc = (phi_muls + tf_muls) / P + arch.mult_lat
    else:
        # All through standard add/mult functional units
        c.mult_cyc = (phi_muls + tf_muls + btfly_muls) / P + arch.mult_lat
        c.add_cyc  = btfly_adds / P + arch.add_lat

    return c


def poly_automorph(N: int, limbs: int, arch: ArchParam) -> Cost:
    """
    Automorphism (coefficient permutation) over `limbs` RNS primes.

    Uses best-case model (SimFHE poly.automorph auto_cyc_fm_bc):
      auto_cyc = N * limbs / (funits * sets) + auto_lat
    """
    c = Cost()
    c.auto_cyc = N * limbs / (arch.funits * arch.sets) + arch.auto_lat
    return c


def basis_convert(N: int, in_limbs: int, out_limbs: int, arch: ArchParam) -> Cost:
    """
    RNS basis conversion from `in_limbs` primes to `out_limbs` primes.

    Source: SimFHE evaluator.py:basis_convert
      sw_add = N × in_limbs × out_limbs
      sw_mul = N × in_limbs × (1 + out_limbs)
    """
    c = Cost()
    P = arch.funits * arch.sets
    sw_add = N * in_limbs * out_limbs
    sw_mul = N * in_limbs * (1 + out_limbs)
    c.add_cyc  = sw_add / P + arch.add_lat
    c.mult_cyc = sw_mul / P + arch.mult_lat
    return c
