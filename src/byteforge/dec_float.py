import os
from typing import Union

import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register

_HAS_C = False
if not os.environ.get("BYTEFORGE_NO_C"):
    try:
        from byteforge._c.ufunc import (
            dec32_decode as _c_dec32_decode,
        )
        from byteforge._c.ufunc import (
            dec32_encode as _c_dec32_encode,
        )
        from byteforge._c.ufunc import (
            dec64_decode as _c_dec64_decode,
        )
        from byteforge._c.ufunc import (
            dec64_encode as _c_dec64_encode,
        )
        from byteforge._c.ufunc import (
            dec64g_decode as _c_dec64g_decode,
        )
        from byteforge._c.ufunc import (
            dec64g_encode as _c_dec64g_encode,
        )

        _HAS_C = True
    except ImportError:
        pass

_DEC_VALID_WIDTHS = frozenset((32, 64))


@register("dec_float")
@register("dec32", bit_width=32)
@register("dec64", bit_width=64)
class DECFloat(Encoding):
    """DEC (VAX) floating point encoding: F4 (32-bit) and D4 (64-bit).

    Bit layout (MSB -> LSB): [sign 1 bit][exponent 8 bits][mantissa 23 or 55 bits]

    Decode: value = (-1)^s * (m / 2^(mantissa_bits+1) + 0.5) * 2^(e - 128)
      - Hidden bit at 0.5 (not 1.0 like IEEE 754)
      - e = 0 means zero (reserved)

    Reference: https://pubs.usgs.gov/of/2005/1424/
    """

    def __init__(
        self, bit_width: int, *, encode_errors: Union[str, int, float] = "clamp"
    ) -> None:
        if bit_width not in _DEC_VALID_WIDTHS:
            raise ValueError(f"DECFloat bit_width must be 32 or 64, got {bit_width}")
        super().__init__(bit_width, encode_errors=encode_errors)
        self._exp_bits = 8
        self._bias = 128
        self._mant_bits = bit_width - 1 - self._exp_bits  # 23 or 55
        self._init_masks()

    def _init_masks(self) -> None:
        """Compute derived bit masks and shift amounts from field widths."""
        self._mant_mask = (1 << self._mant_bits) - 1
        self._exp_mask = (1 << self._exp_bits) - 1
        self._exp_shift = self._mant_bits
        self._sign_shift = self._mant_bits + self._exp_bits

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        fval = np.asarray(values, dtype=np.float64)

        if _HAS_C:
            if self.bit_width == 32:
                result = _c_dec32_encode(fval)  # type: ignore[possibly-undefined]
            elif self._exp_bits == 11:
                result = _c_dec64g_encode(fval)  # type: ignore[possibly-undefined]
            else:
                result = _c_dec64_encode(fval)  # type: ignore[possibly-undefined]
        else:
            result = self._encode_py(fval)
        lo, hi = self.value_range
        return self._apply_encode_overflow(fval, lo, hi, result)

    def _encode_py(self, fval: np.ndarray) -> np.ndarray:
        """Pure-Python DEC float encode.

        Computes biased exponent and mantissa with hidden bit at 0.5, then
        packs into the ``[S1|E8/11|M]`` bit layout.

        Args:
            fval: Float64 array of values to encode.

        Returns:
            Unsigned integer array of encoded bit patterns.
        """
        result = np.zeros(fval.shape, dtype=np.uint64)

        nonzero = fval != 0.0
        if not np.any(nonzero):
            return result.astype(self._dn_dtype)

        vals = fval[nonzero]
        sign = (vals < 0).astype(np.uint64)
        abs_vals = np.abs(vals)

        # DEC: value = (0.5 + m_frac) * 2^(e - bias)
        # where m_frac = m / 2^(mant_bits+1), range [0, 0.5)
        # significand is in [0.5, 1.0)
        # => e_unbiased = floor(log2(abs_val)) + 1
        e_unbiased = np.floor(np.log2(abs_vals)).astype(np.int64) + 1

        # significand = abs_val / 2^e_unbiased, should be in [0.5, 1.0)
        significand = abs_vals / np.exp2(e_unbiased.astype(np.float64))

        # Extract mantissa: m = round((significand - 0.5) * 2^(mant_bits+1))
        m_frac = significand - 0.5
        m_val = np.round(m_frac * np.exp2(self._mant_bits + 1)).astype(np.uint64)
        m_val = np.minimum(m_val, self._mant_mask)

        # Biased exponent, clamped (e=0 reserved for zero)
        e_biased = np.clip(
            e_unbiased + self._bias, 1, (1 << self._exp_bits) - 1
        ).astype(np.uint64)

        result[nonzero] = (
            (sign << self._sign_shift)
            | (e_biased << self._exp_shift)
            | m_val
        )
        return result.astype(self._dn_dtype)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)

        if _HAS_C:
            if self.bit_width == 32:
                return _c_dec32_decode(  # type: ignore[possibly-undefined]
                    arr.astype(np.uint32)
                )
            elif self._exp_bits == 11:
                return _c_dec64g_decode(arr)  # type: ignore[possibly-undefined]
            else:
                return _c_dec64_decode(arr)  # type: ignore[possibly-undefined]
        return self._decode_py(arr)

    def _decode_py(self, arr: np.ndarray) -> np.ndarray:
        """Pure-Python DEC float decode.

        Extracts sign, biased exponent, and mantissa from bit patterns and
        reconstructs float64 values via ``(-1)^s * (0.5 + m_frac) * 2^(e-bias)``.

        Args:
            arr: Validated uint64 array of DEC float bit patterns.

        Returns:
            Float64 array of decoded values.
        """
        s = (arr >> self._sign_shift) & 1
        e = (arr >> self._exp_shift) & self._exp_mask
        m = arr & self._mant_mask

        # value = (-1)^s * (m / 2^(mant_bits+1) + 0.5) * 2^(e - bias)
        sign = np.where(s == 0, 1.0, -1.0)
        M = m.astype(np.float64) / np.exp2(self._mant_bits + 1) + 0.5
        E = e.astype(np.int64) - self._bias

        result = sign * M * np.exp2(E.astype(np.float64))

        # e == 0 is reserved (zero)
        result = np.where(e == 0, 0.0, result)
        return result

    @property
    def value_range(self) -> tuple[float, float]:
        max_e = (1 << self._exp_bits) - 1
        max_m_frac = ((1 << self._mant_bits) - 1) / 2.0 ** (self._mant_bits + 1)
        max_val = (0.5 + max_m_frac) * 2.0 ** (max_e - self._bias)
        return (-max_val, max_val)

    def __repr__(self) -> str:
        label = {32: "F4", 64: "D4"}.get(self.bit_width, str(self.bit_width))
        return (
            f"DECFloat(bit_width={self.bit_width}, format={label}, "
            f"mantissa={self._mant_bits}, exponent={self._exp_bits})"
        )


@register("dec_float_g")
@register("dec64g", bit_width=64)
class DECFloatG(DECFloat):
    """DEC (VAX) G4 floating point encoding (64-bit, 11-bit exponent).

    Bit layout (MSB -> LSB): [sign 1 bit][exponent 11 bits][mantissa 52 bits]

    Same decode formula as DECFloat but with 11-bit exponent and bias of 1024.
    """

    def __init__(
        self, bit_width: int = 64, *, encode_errors: Union[str, int, float] = "clamp"
    ) -> None:
        if bit_width != 64:
            raise ValueError(f"DECFloatG bit_width must be 64, got {bit_width}")
        Encoding.__init__(self, bit_width, encode_errors=encode_errors)
        self._exp_bits = 11
        self._bias = 1024
        self._mant_bits = 52
        self._init_masks()

    def __repr__(self) -> str:
        return (
            f"DECFloatG(bit_width={self.bit_width}, format=G4, "
            f"mantissa={self._mant_bits}, exponent={self._exp_bits})"
        )
