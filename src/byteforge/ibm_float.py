import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register

_VALID_WIDTHS = frozenset((32, 64))


@register("ibm_float")
@register("ibm32", bit_width=32)
@register("ibm64", bit_width=64)
class IBMFloat(Encoding):
    """IBM hexadecimal floating point encoding (32-bit or 64-bit).

    Bit layout (MSB -> LSB): [sign 1 bit][exponent 7 bits][mantissa 24 or 56 bits]

    Decode: value = (-1)^s * (m / 2^mantissa_bits) * 16^(e - 64)
      - Base-16 exponent with bias of 64
      - m = 0 means zero regardless of sign and exponent

    Reference: https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point
    """

    def __init__(self, bit_width: int) -> None:
        if bit_width not in _VALID_WIDTHS:
            raise ValueError(f"IBMFloat bit_width must be 32 or 64, got {bit_width}")
        super().__init__(bit_width)
        self._mant_bits = bit_width - 8  # 24 for IBM32, 56 for IBM64
        self._mant_mask = np.uint64((1 << self._mant_bits) - 1)
        self._exp_shift = np.uint64(self._mant_bits)
        self._sign_shift = np.uint64(bit_width - 1)

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        fval = np.asarray(values, dtype=np.float64)
        mbits = self._mant_bits
        result = np.zeros(fval.shape, dtype=np.uint64)

        nonzero = fval != 0.0
        if not np.any(nonzero):
            return result.astype(self._dn_dtype)

        vals = fval[nonzero]
        sign = (vals < 0).astype(np.uint64)
        abs_vals = np.abs(vals)

        # IBM uses base-16 exponent: value = (m/2^mbits) * 16^(e-64)
        # Mantissa fraction m/2^mbits must be in [1/16, 1)
        # hex_exp = e - 64 = ceil(log16(abs_val)) = ceil(log2(abs_val) / 4)
        log2_vals = np.log2(abs_vals)
        hex_exp = np.ceil(log2_vals / 4.0).astype(np.int64)

        # Compute mantissa fraction = abs_val / 16^hex_exp
        mant_frac = abs_vals / np.exp2(4.0 * hex_exp.astype(np.float64))

        # Normalize: if mant_frac rounds up to >= 1, increment exponent
        too_large = mant_frac >= 1.0
        hex_exp = np.where(too_large, hex_exp + 1, hex_exp)
        mant_frac = np.where(
            too_large,
            abs_vals / np.exp2(4.0 * hex_exp.astype(np.float64)),
            mant_frac,
        )

        m_val = np.round(mant_frac * np.float64(1 << mbits)).astype(np.uint64)
        m_val = np.minimum(m_val, self._mant_mask)

        e_biased = np.clip(hex_exp + 64, 0, 127).astype(np.uint64)

        result[nonzero] = (
            (sign << self._sign_shift) | (e_biased << self._exp_shift) | m_val
        )
        return result.astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)
        mbits = self._mant_bits

        s = (arr >> self._sign_shift) & np.uint64(1)
        e = (arr >> self._exp_shift) & np.uint64(0x7F)
        m = arr & self._mant_mask

        # value = (-1)^s * (m / 2^mbits) * 16^(e - 64)
        sign = np.where(s == np.uint64(0), 1.0, -1.0)
        M = m.astype(np.float64) / np.float64(1 << min(mbits, 52))
        if mbits > 52:
            # For IBM64 (56-bit mantissa), avoid precision loss from 1<<56
            M = M / np.float64(1 << (mbits - 52))
        E = e.astype(np.int64) - 64

        # 16^E = 2^(4*E)
        result = sign * M * np.exp2(4.0 * E.astype(np.float64))

        # m == 0 means zero
        result = np.where(m == np.uint64(0), 0.0, result)
        return result

    @property
    def value_range(self) -> tuple[float, float]:
        mbits = self._mant_bits
        # Max: s=0, e=127, m=all-1s => ((2^mbits-1)/2^mbits) * 16^63
        max_mant = (2.0**mbits - 1) / 2.0**mbits
        max_val = max_mant * 16.0**63
        return (-max_val, max_val)

    def __repr__(self) -> str:
        return (
            f"IBMFloat(bit_width={self.bit_width}, "
            f"mantissa={self._mant_bits}, exponent=7, base=16)"
        )
