import os

import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register

_HAS_C = False
if not os.environ.get("BYTEFORGE_NO_C"):
    try:
        from byteforge._c.ufunc import (
            ti32_decode as _c_ti32_decode,
        )
        from byteforge._c.ufunc import (
            ti32_encode as _c_ti32_encode,
        )
        from byteforge._c.ufunc import (
            ti40_decode as _c_ti40_decode,
        )
        from byteforge._c.ufunc import (
            ti40_encode as _c_ti40_encode,
        )

        _HAS_C = True
    except ImportError:
        pass

_VALID_WIDTHS = frozenset((32, 40))


@register("ti_float")
@register("ti32", bit_width=32)
@register("ti40", bit_width=40)
class TIFloat(Encoding):
    """Texas Instruments floating point encoding (32-bit or 40-bit).

    Bit layout (MSB -> LSB): [exponent 8 bits][sign 1 bit][mantissa 23 or 31 bits]

    Decode: value = ((-2)^s + m / 2^mantissa_bits) * 2^e
      - s=0: value = (1 + m/2^mbits) * 2^e   (positive, significand in [1, 2))
      - s=1: value = (-2 + m/2^mbits) * 2^e   (negative, significand in [-2, -1))
      - e = -128 (two's complement): value is zero regardless of s and m

    Reference: https://www.ti.com/lit/an/spra400/spra400.pdf
    """

    def __init__(self, bit_width: int) -> None:
        if bit_width not in _VALID_WIDTHS:
            raise ValueError(f"TIFloat bit_width must be 32 or 40, got {bit_width}")
        super().__init__(bit_width)
        self._mant_bits = bit_width - 9  # 23 for TI32, 31 for TI40
        self._mant_mask = (1 << self._mant_bits) - 1
        self._sign_shift = self._mant_bits
        self._exp_shift = self._mant_bits + 1

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        fval = np.asarray(values, dtype=np.float64)

        if _HAS_C:
            if self.bit_width == 32:
                return _c_ti32_encode(fval)  # type: ignore[possibly-undefined]
            else:
                return _c_ti40_encode(fval)  # type: ignore[possibly-undefined]
        return self._encode_py(fval)

    def _encode_py(self, fval: np.ndarray) -> np.ndarray:
        mbits = self._mant_bits
        result = np.zeros(fval.shape, dtype=np.uint64)

        # Zero case: e_field = 0x80 (two's complement -128)
        is_zero = fval == 0.0
        result[is_zero] = 0x80 << self._exp_shift

        nonzero = ~is_zero
        if not np.any(nonzero):
            return result.astype(self._dn_dtype)

        vals = fval[nonzero]
        abs_vals = np.abs(vals)
        sign = (vals < 0).astype(np.uint64)

        # Exponent: for both positive and negative, |value| is in [2^e, 2^(e+1))
        e = np.floor(np.log2(abs_vals)).astype(np.int64)

        # Mantissa fraction:
        #   s=0: value = (1 + m_frac) * 2^e  =>  m_frac = value / 2^e - 1
        #   s=1: value = (-2 + m_frac) * 2^e  =>  m_frac = value / 2^e + 2
        scaled = vals / np.exp2(e.astype(np.float64))
        m_frac = np.where(sign == 0, scaled - 1.0, scaled + 2.0)
        m_val = np.clip(
            np.round(m_frac * float(1 << mbits)).astype(np.int64),
            0,
            (1 << mbits) - 1,
        ).astype(np.uint64)

        # Clamp exponent to [-127, 127] (-128 is reserved for zero)
        e = np.clip(e, -127, 127)
        e_u = e.astype(np.uint64) & 0xFF

        result[nonzero] = (
            (e_u << self._exp_shift) | (sign << self._sign_shift) | m_val
        )
        return result.astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)

        if _HAS_C:
            if self.bit_width == 32:
                return _c_ti32_decode(arr.astype(np.uint32))  # type: ignore[possibly-undefined]
            else:
                return _c_ti40_decode(arr)  # type: ignore[possibly-undefined]
        return self._decode_py(arr)

    def _decode_py(self, arr: np.ndarray) -> np.ndarray:
        mbits = self._mant_bits

        # Extract fields
        e_u = (arr >> self._exp_shift) & 0xFF
        s = (arr >> self._sign_shift) & 1
        m = arr & self._mant_mask

        # Two's complement exponent (8-bit signed)
        e_i = e_u.astype(np.int64)
        e = np.where(e_i >= 128, e_i - 256, e_i)

        # value = ((-2)^s + m / 2^mbits) * 2^e
        S = np.where(s == 0, 1.0, -2.0)
        M = m.astype(np.float64) / float(1 << mbits)
        result = (S + M) * np.exp2(e.astype(np.float64))

        # e == -128 means zero
        result = np.where(e == -128, 0.0, result)
        return result

    @property
    def value_range(self) -> tuple[float, float]:
        mbits = self._mant_bits
        # Max positive: e=127, s=0, m=all-1s => (2 - 2^(-mbits)) * 2^127
        max_val = (2.0 - 2.0 ** (-mbits)) * 2.0**127
        # Max negative: e=127, s=1, m=0 => -2 * 2^127
        min_val = -2.0 * 2.0**127
        return (min_val, max_val)

    def __repr__(self) -> str:
        return (
            f"TIFloat(bit_width={self.bit_width}, "
            f"mantissa={self._mant_bits}, exponent=8)"
        )
