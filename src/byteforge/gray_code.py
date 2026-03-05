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

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
        if np.isdtype(arr.dtype, "integral"):
            n = np.clip(arr, 0, self.max_unsigned).astype(np.uint64)
        else:
            upper = float(self.max_unsigned)
            if int(upper) > self.max_unsigned:
                upper = np.nextafter(upper, 0.0)
            n = np.clip(np.round(arr.astype(np.float64)), 0, upper).astype(np.uint64)
        return (n ^ (n >> 1)).astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)

        if _HAS_C:
            return _c_gray_decode(  # type: ignore[possibly-undefined]
                arr, np.uint8(self.bit_width)
            ).astype(self._dn_dtype)
        return self._decode_py(arr)

    def _decode_py(self, arr: np.ndarray) -> np.ndarray:
        n = arr.copy()
        shift = 1
        while shift < self.bit_width:
            n ^= n >> shift
            shift <<= 1
        return n.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, self.max_unsigned)

    def __repr__(self) -> str:
        return f"GrayCode(bit_width={self.bit_width})"
