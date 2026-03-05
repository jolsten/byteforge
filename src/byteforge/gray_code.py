import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register


@register("gray_code")
@register("gray")
class GrayCode(Encoding):
    """Gray code encoding.

    Adjacent values differ by exactly one bit.
    """

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.integer):
            n = np.clip(arr, 0, self.max_unsigned).astype(np.uint64)
        else:
            upper = np.float64(self.max_unsigned)
            if int(upper) > self.max_unsigned:
                upper = np.nextafter(upper, np.float64(0))
            n = np.clip(np.round(arr.astype(np.float64)), 0, upper).astype(np.uint64)
        return (n ^ (n >> np.uint64(1))).astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)
        n = arr.copy()
        shift = 1
        while shift < self.bit_width:
            n ^= n >> np.uint64(shift)
            shift <<= 1
        return n.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, self.max_unsigned)

    def __repr__(self) -> str:
        return f"GrayCode(bit_width={self.bit_width})"
