import numpy as np
import numpy.typing as npt

from ._base import Encoding, _min_int_dtype
from ._registry import register


@register("offset_binary")
class OffsetBinary(Encoding):
    """Offset Binary encoding (excess-N).

    Maps signed values to unsigned by adding a zero offset of 2^(N-1).
    """

    def __init__(self, bit_width: int) -> None:
        super().__init__(bit_width)
        self._zero_offset = 1 << (bit_width - 1)
        self._min_value = -(1 << (bit_width - 1))
        self._max_value = (1 << (bit_width - 1)) - 1
        self._int_dtype: type[np.signedinteger] = _min_int_dtype(bit_width)

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.integer):
            clamped = np.clip(arr, self._min_value, self._max_value).astype(np.int64)
        else:
            clamped = np.clip(
                np.round(arr.astype(np.float64)),
                self._min_value,
                self._max_value,
            ).astype(np.int64)
        return (clamped + self._zero_offset).astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)
        return (arr.astype(np.int64) - self._zero_offset).astype(self._int_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (self._min_value, self._max_value)

    def __repr__(self) -> str:
        return (
            f"OffsetBinary(bit_width={self.bit_width}, "
            f"range=[{self._min_value}, {self._max_value}])"
        )
