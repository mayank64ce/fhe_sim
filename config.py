import json
from dataclasses import dataclass
from typing import Optional

# 128-bit security standard table: (max_logQ, N)
# Source: https://homomorphicencryption.org/standard/
_HE_SECURITY_TABLE = [
    (109,   4_096),
    (218,   8_192),
    (438,   16_384),
    (881,   32_768),
    (1_761, 65_536),
    (3_523, 131_072),
]

_OPENFHE_DEFAULTS = {
    "scheme":               "CKKS",
    "scale_mod_size":       50,
    "first_mod_size":       60,
    "enable_bootstrapping": False,
}


def _default_num_large_digits(mult_depth: int) -> int:
    """
    OpenFHE's default numLargeDigits selection logic.
    Source: scheme/scheme-utils.h ComputeNumLargeDigits()
    """
    if mult_depth > 3:
        return 3
    if mult_depth > 0:
        return 2
    return 1


@dataclass
class FHEConfig:
    scheme:                           str
    mult_depth:                       int
    ring_dimension:                   int       # N
    scale_mod_size:                   int
    first_mod_size:                   int
    batch_size:                       Optional[int]
    enable_bootstrapping:             bool
    levels_available_after_bootstrap: int
    level_budget:                     list
    indexes_for_rotation_key:         list
    num_large_digits:                 int       # dnum: key-switch decomposition digits
    # BFV-specific
    plaintext_modulus:                Optional[int] = None
    max_relin_sk_deg:                 Optional[int] = None

    @property
    def logQ(self) -> int:
        return self.first_mod_size + self.mult_depth * self.scale_mod_size

    @property
    def num_slots(self) -> int:
        return self.batch_size if self.batch_size else self.ring_dimension // 2

    @property
    def special_primes(self) -> int:
        """
        K: number of special RNS primes for Hybrid key switching.
        OpenFHE sets K = ceil(numQ / dnum) where numQ = mult_depth + 1
        (one first_mod prime plus mult_depth scale_mod primes).
        Source: verified via GenCryptoContext + GetParamsP().GetParams().size()
        """
        import math
        return math.ceil((self.mult_depth + 1) / max(1, self.num_large_digits))


def get_ks_params(config: "FHEConfig", level: int) -> tuple[int, int]:
    """
    Return (K, dnum) for key switching at a given ciphertext level.

    K    : number of special primes  = config.special_primes
    dnum : decomposition digits      = config.num_large_digits

    Note: dnum can optionally be adjusted per-level, but OpenFHE uses a fixed
    dnum (set at context creation time) regardless of current level.
    """
    return config.special_primes, config.num_large_digits


def _derive_ring_dimension(logQ: int) -> int:
    for max_logQ, N in _HE_SECURITY_TABLE:
        if logQ <= max_logQ:
            return N
    raise ValueError(
        f"logQ={logQ} bits exceeds the maximum supported by the HE security "
        f"standard table (max={_HE_SECURITY_TABLE[-1][0]}). "
        f"A larger ring dimension would be needed."
    )


def load_config(path: str) -> FHEConfig:
    with open(path) as f:
        raw = json.load(f)

    scheme          = raw.get("scheme", _OPENFHE_DEFAULTS["scheme"])
    mult_depth      = raw["mult_depth"]
    scale_mod_size  = raw.get("scale_mod_size", _OPENFHE_DEFAULTS["scale_mod_size"])
    first_mod_size  = raw.get("first_mod_size", _OPENFHE_DEFAULTS["first_mod_size"])

    logQ = first_mod_size + mult_depth * scale_mod_size

    if "ring_dimension" in raw:
        ring_dimension = raw["ring_dimension"]
    else:
        ring_dimension = _derive_ring_dimension(logQ)

    num_large_digits = raw.get("num_large_digits", _default_num_large_digits(mult_depth))

    return FHEConfig(
        scheme                          = scheme,
        mult_depth                      = mult_depth,
        ring_dimension                  = ring_dimension,
        scale_mod_size                  = scale_mod_size,
        first_mod_size                  = first_mod_size,
        batch_size                      = raw.get("batch_size"),
        enable_bootstrapping            = raw.get("enable_bootstrapping", False),
        levels_available_after_bootstrap= raw.get("levels_available_after_bootstrap", 0),
        level_budget                    = raw.get("level_budget", []),
        indexes_for_rotation_key        = raw.get("indexes_for_rotation_key", []),
        num_large_digits                = num_large_digits,
        plaintext_modulus               = raw.get("plaintext_modulus"),
        max_relin_sk_deg                = raw.get("max_relin_sk_deg"),
    )
