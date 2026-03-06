import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register


@register("unsigned")
@register("u")
class Unsigned(Encoding):
    """Encodes values as unsigned integers, clamped to [0, 2^bit_width - 1]."""

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
        self._check_overflow(arr, 0, self.max_unsigned)
        if np.isdtype(arr.dtype, "integral"):
            # Clamp negatives to 0 before uint64 cast to avoid silent wrap
            arr = np.where(arr < 0, 0, arr).astype(np.uint64)
            clamped = np.clip(arr, np.uint64(0), np.uint64(self.max_unsigned))
        else:
            # float64 can't exactly represent large uint64 values;
            # e.g. float64(2^64-1) rounds to 2^64, overflowing uint64 on cast.
            # Cap to the largest float64 below 2^bit_width.
            upper = float(self.max_unsigned)
            if int(upper) > self.max_unsigned:
                upper = np.nextafter(upper, 0.0)
            clamped = np.clip(np.round(arr.astype(np.float64)), 0, upper).astype(np.uint64)
        return clamped.astype(self._dn_dtype)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)
        return arr.astype(self._dn_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (0, self.max_unsigned)

    @classmethod
    def from_range(cls, *, max_value: int) -> "Unsigned":
        """Construct from the maximum value that needs to be represented.

        Args:
            max_value: The largest value to encode.

        Returns:
            An Unsigned encoding with the minimum required bit width.

        Raises:
            ValueError: If ``max_value`` is negative.
        """
        if max_value < 0:
            raise ValueError(f"max_value must be >= 0, got {max_value}")
        bit_width = max(1, max_value.bit_length())
        return cls(bit_width)
