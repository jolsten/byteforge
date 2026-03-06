from typing import Union

import numpy as np
import numpy.typing as npt

from ._base import Encoding, _min_int_dtype
from ._registry import register


@register("offset_binary")
class OffsetBinary(Encoding):
    """Offset Binary encoding (excess-N).

    Maps signed values to unsigned by adding a zero offset of 2^(N-1).
    """

    def __init__(
        self, bit_width: int, *, encode_errors: Union[str, int, float] = "clamp"
    ) -> None:
        super().__init__(bit_width, encode_errors=encode_errors)
        self._zero_offset = 1 << (bit_width - 1)
        self._min_value = -(1 << (bit_width - 1))
        self._max_value = (1 << (bit_width - 1)) - 1
        self._int_dtype: type[np.signedinteger] = _min_int_dtype(bit_width)

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
        if np.isdtype(arr.dtype, "integral"):
            clamped = np.clip(arr, self._min_value, self._max_value).astype(np.int64)
        else:
            clamped = np.clip(
                np.round(arr.astype(np.float64)),
                self._min_value,
                self._max_value,
            ).astype(np.int64)
        result = (clamped + self._zero_offset).astype(self._dn_dtype)
        return self._apply_encode_overflow(arr, self._min_value, self._max_value, result)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)
        return (arr.astype(np.int64) - self._zero_offset).astype(self._int_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (self._min_value, self._max_value)

    @classmethod
    def from_range(cls, *, min_value: int, max_value: int) -> "OffsetBinary":
        """Construct from the signed range that needs to be represented.

        Offset binary has the same range as two's complement:
        ``[-2^(N-1), 2^(N-1)-1]``.

        Args:
            min_value: Most negative value to encode.
            max_value: Most positive value to encode.

        Returns:
            An OffsetBinary encoding with the minimum required bit width.
        """
        if min_value >= 0:
            bit_width = max_value.bit_length() + 1
        else:
            neg_bits = ((-min_value) - 1).bit_length() + 1
            pos_bits = max_value.bit_length() + 1 if max_value > 0 else 1
            bit_width = max(neg_bits, pos_bits)
        return cls(max(1, bit_width))

    def __repr__(self) -> str:
        return (
            f"OffsetBinary(bit_width={self.bit_width}, "
            f"range=[{self._min_value}, {self._max_value}])"
        )
