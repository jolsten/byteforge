import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register


@register("unsigned")
@register("u")
class Unsigned(Encoding):
    """Encodes values as unsigned integers, clamped to [0, 2^bit_width - 1]."""

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.integer):
            clamped = np.clip(arr, 0, self.max_unsigned).astype(np.uint64)
        else:
            # float64 can't exactly represent large uint64 values;
            # e.g. float64(2^64-1) rounds to 2^64, overflowing uint64 on cast.
            # Cap to the largest float64 below 2^bit_width.
            upper = np.float64(self.max_unsigned)
            if int(upper) > self.max_unsigned:
                upper = np.nextafter(upper, np.float64(0))
            clamped = np.clip(np.round(arr.astype(np.float64)), 0, upper).astype(np.uint64)
        return clamped.astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)
        return arr.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, self.max_unsigned)

    @classmethod
    def from_range(cls, *, max_value: int) -> "Unsigned":
        """Construct from the maximum value that needs to be represented."""
        if max_value < 0:
            raise ValueError(f"max_value must be >= 0, got {max_value}")
        bit_width = max(1, max_value.bit_length())
        return cls(bit_width)
