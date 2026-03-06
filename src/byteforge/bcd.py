import os
from typing import Union

import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register

_HAS_C = False
if not os.environ.get("BYTEFORGE_NO_C"):
    try:
        from byteforge._c.ufunc import bcd_decode as _c_bcd_decode
        from byteforge._c.ufunc import bcd_encode as _c_bcd_encode

        _HAS_C = True
    except ImportError:
        pass


@register("bcd")
class BCD(Encoding):
    """Binary-Coded Decimal encoding.

    Each 4-bit nibble stores a single decimal digit (0-9).
    The bit_width must be a multiple of 4.

    Args:
        bit_width: Number of bits (must be a multiple of 4, range 4-64).
        errors: How to handle invalid BCD nibbles (>=10) during decode:

            - ``"raise"`` (default): raise ``ValueError``
            - ``"nan"``: return ``float64`` with ``np.nan`` for invalid elements
            - numeric value: substitute this sentinel for invalid elements
    """

    def __init__(self, bit_width: int, *, errors: Union[str, int, float] = "raise") -> None:
        if bit_width % 4 != 0:
            raise ValueError(f"BCD bit_width must be a multiple of 4, got {bit_width}")
        if not 4 <= bit_width <= 64:
            raise ValueError(f"BCD bit_width must be 4-64, got {bit_width}")
        if isinstance(errors, str) and errors not in ("raise", "nan"):
            raise ValueError(f"errors must be 'raise', 'nan', or a numeric value, got {errors!r}")
        if not isinstance(errors, (str, int, float)):
            raise TypeError(f"errors must be str, int, or float, got {type(errors).__name__}")
        super().__init__(bit_width)
        if isinstance(errors, (int, float)) and not isinstance(errors, bool):
            max_val = np.iinfo(self._dn_dtype).max
            if int(errors) < 0 or int(errors) > max_val:
                raise ValueError(
                    f"Sentinel value {errors} does not fit in output dtype "
                    f"{self._dn_dtype.__name__} (max {max_val})"
                )
        self._max_digits = bit_width // 4
        self._max_bcd_value = 10**self._max_digits - 1
        self._decode_errors = errors

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        self._check_overflow(np.asarray(values), 0, self._max_bcd_value)
        arr = np.clip(
            np.round(np.asarray(values, dtype=np.float64)),
            0,
            self._max_bcd_value,
        ).astype(np.uint64)

        if _HAS_C:
            return _c_bcd_encode(arr, np.uint8(self._max_digits)).astype(self._dn_dtype)  # type: ignore[possibly-undefined]
        return self._encode_py(arr)

    def _encode_py(self, arr: np.ndarray) -> np.ndarray:
        result = np.zeros_like(arr, dtype=np.uint64)
        remaining = arr.copy()
        for i in range(self._max_digits):
            digit = remaining % 10
            result |= digit << (i * 4)
            remaining //= 10
        return result.astype(self._dn_dtype)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)

        if _HAS_C:
            return self._decode_c(arr)
        return self._decode_py(arr)

    def _decode_c(self, arr: np.ndarray) -> np.ndarray:
        raw = _c_bcd_decode(arr, np.uint8(self._max_digits))  # type: ignore[possibly-undefined]
        sentinel = np.uint64(np.iinfo(np.uint64).max)
        invalid = raw == sentinel
        if np.any(invalid):
            if self._decode_errors == "raise":
                # Rescan in Python for detailed error message
                bad_idx = np.where(invalid)[0][0]
                dn_val = arr[bad_idx]
                for pos in range(self._max_digits):
                    nib = int((dn_val >> (pos * 4)) & 0xF)
                    if nib >= 10:
                        raise ValueError(
                            f"Invalid BCD nibble {nib} at position {pos} in DN {dn_val}"
                        )
            if self._decode_errors == "nan":
                out = raw.astype(np.float64)
                out[invalid] = np.nan
                return out
            raw[invalid] = np.uint64(self._decode_errors)
        if self._decode_errors == "nan":
            return raw.astype(np.float64)
        return raw.astype(self._dn_dtype)

    def _decode_py(self, arr: np.ndarray) -> np.ndarray:
        result = np.zeros_like(arr, dtype=np.uint64)
        invalid = np.zeros(arr.shape, dtype=bool)
        multiplier = 1
        for i in range(self._max_digits):
            nibble = (arr >> (i * 4)) & 0xF
            bad = nibble > 9
            if np.any(bad):
                if self._decode_errors == "raise":
                    bad_idx = np.where(bad)[0][0]
                    raise ValueError(
                        f"Invalid BCD nibble {nibble[bad_idx]} at position {i} in DN {arr[bad_idx]}"
                    )
                invalid |= bad
            result += nibble * multiplier
            multiplier *= 10

        if self._decode_errors == "nan":
            out = result.astype(np.float64)
            out[invalid] = np.nan
            return out

        if np.any(invalid):
            result[invalid] = np.uint64(self._decode_errors)

        return result.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, self._max_bcd_value)

    @classmethod
    def from_range(cls, *, max_value: int, **kwargs: object) -> "BCD":
        """Construct from the maximum decimal value that needs to be represented.

        Args:
            max_value: The largest value to encode.
            **kwargs: Forwarded to the constructor (e.g. ``errors``).

        Returns:
            A BCD encoding with the minimum required bit width.

        Raises:
            ValueError: If ``max_value`` is negative.
        """
        if max_value < 0:
            raise ValueError(f"max_value must be >= 0, got {max_value}")
        n_digits = len(str(max_value))
        return cls(n_digits * 4, **kwargs)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        extras = f", errors={self._decode_errors!r}" if self._decode_errors != "raise" else ""
        return f"BCD(bit_width={self.bit_width}, max_digits={self._max_digits}{extras})"

    def __str__(self) -> str:
        extras = f", errors={self._decode_errors!r}" if self._decode_errors != "raise" else ""
        return f"BCD({self.bit_width}{extras})"
