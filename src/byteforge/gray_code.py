import os

import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register

_HAS_C = False
if not os.environ.get("BYTEFORGE_NO_C"):
    try:
        from byteforge._c.ufunc import gray_decode as _c_gray_decode

        _HAS_C = True
    except ImportError:
        pass


@register("gray_code")
@register("gray")
class GrayCode(Encoding):
    """Gray code encoding.

    Adjacent values differ by exactly one bit.
    """

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
        if np.isdtype(arr.dtype, "integral"):
            clamped = np.where(arr < 0, 0, arr).astype(np.uint64)
            n = np.clip(clamped, np.uint64(0), np.uint64(self.max_unsigned))
        else:
            upper = float(self.max_unsigned)
            if int(upper) > self.max_unsigned:
                upper = np.nextafter(upper, 0.0)
            n = np.clip(np.round(arr.astype(np.float64)), 0, upper).astype(np.uint64)
        result = (n ^ (n >> 1)).astype(self._dn_dtype)
        return self._apply_encode_overflow(arr, 0, self.max_unsigned, result)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)

        if _HAS_C:
            return _c_gray_decode(  # type: ignore[possibly-undefined]
                arr, np.uint8(self.bit_width)
            ).astype(self._dn_dtype)
        return self._decode_py(arr)

    def _decode_py(self, arr: np.ndarray) -> np.ndarray:
        """Pure-Python Gray code decode via iterative XOR-shift.

        Args:
            arr: Validated uint64 array of Gray-coded bit patterns.

        Returns:
            Decoded unsigned integer values with the target DN dtype.
        """
        n = arr.copy()
        shift = 1
        while shift < self.bit_width:
            n ^= n >> shift
            shift <<= 1
        return n.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, self.max_unsigned)

    @classmethod
    def from_range(cls, *, max_value: int) -> "GrayCode":
        """Construct from the maximum value that needs to be represented.

        Args:
            max_value: The largest value to encode.

        Returns:
            A GrayCode encoding with the minimum required bit width.

        Raises:
            ValueError: If ``max_value`` is negative.
        """
        if max_value < 0:
            raise ValueError(f"max_value must be >= 0, got {max_value}")
        bit_width = max(1, max_value.bit_length())
        return cls(bit_width)

    def __repr__(self) -> str:
        return f"GrayCode(bit_width={self.bit_width})"
