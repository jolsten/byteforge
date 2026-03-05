from typing import Union

import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register


@register("bcd")
class BCD(Encoding):
    """Binary-Coded Decimal encoding.

    Each 4-bit nibble stores a single decimal digit (0-9).
    The bit_width must be a multiple of 4.

    Parameters
    ----------
    bit_width : int
        Number of bits (must be a multiple of 4, range 4-64).
    errors : str, int, or float
        How to handle invalid BCD nibbles (>=10) during decode:
        - ``"raise"`` (default): raise ``ValueError``
        - ``"nan"``: return ``float64`` with ``np.nan`` for invalid elements
        - numeric value: substitute this sentinel for invalid elements
    """

    def __init__(
        self, bit_width: int, *, errors: Union[str, int, float] = "raise"
    ) -> None:
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
        self._errors = errors

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.clip(
            np.round(np.asarray(values, dtype=np.float64)),
            0,
            self._max_bcd_value,
        ).astype(np.uint64)

        result = np.zeros_like(arr, dtype=np.uint64)
        remaining = arr.copy()
        for i in range(self._max_digits):
            digit = remaining % np.uint64(10)
            result |= digit << np.uint64(i * 4)
            remaining //= np.uint64(10)
        return result.astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)

        result = np.zeros_like(arr, dtype=np.uint64)
        invalid = np.zeros(arr.shape, dtype=bool)
        multiplier = np.uint64(1)
        for i in range(self._max_digits):
            nibble = (arr >> np.uint64(i * 4)) & np.uint64(0xF)
            bad = nibble > 9
            if np.any(bad):
                if self._errors == "raise":
                    bad_idx = np.where(bad)[0][0]
                    raise ValueError(
                        f"Invalid BCD nibble {nibble[bad_idx]} at position {i} "
                        f"in DN {arr[bad_idx]}"
                    )
                invalid |= bad
            result += nibble * multiplier
            multiplier *= np.uint64(10)

        if self._errors == "nan":
            out = result.astype(np.float64)
            out[invalid] = np.nan
            return out

        if np.any(invalid):
            result[invalid] = np.uint64(self._errors)

        return result.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, self._max_bcd_value)

    def __repr__(self) -> str:
        extras = f", errors={self._errors!r}" if self._errors != "raise" else ""
        return f"BCD(bit_width={self.bit_width}, max_digits={self._max_digits}{extras})"
