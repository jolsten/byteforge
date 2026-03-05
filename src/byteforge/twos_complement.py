import numpy as np
import numpy.typing as npt

from ._base import Encoding, _min_int_dtype
from ._registry import register


@register("twos_complement")
@register("2c")
class TwosComplement(Encoding):
    """Encodes signed values as two's complement integers."""

    def __init__(self, bit_width: int) -> None:
        super().__init__(bit_width)
        self._min_signed = -(1 << (bit_width - 1))
        self._max_signed = (1 << (bit_width - 1)) - 1
        self._int_dtype: type[np.signedinteger] = _min_int_dtype(bit_width)

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        # For bit_width <= 53, float64 round-trip is fine.
        # For bit_width > 53, we accept int-typed arrays directly to avoid precision loss.
        arr = np.asarray(values)
        if np.issubdtype(arr.dtype, np.integer):
            clamped = np.clip(arr, self._min_signed, self._max_signed).astype(np.int64)
        else:
            clamped = np.clip(
                np.round(arr.astype(np.float64)),
                self._min_signed,
                self._max_signed,
            ).astype(np.int64)
        # Two's complement: negative values become unsigned via wrapping.
        # Use numpy uint64 arithmetic to avoid Python int overflow at bit_width=64.
        result = clamped.view(np.uint64)
        if self.bit_width < 64:
            mask = np.uint64((1 << self.bit_width) - 1)
            result = result & mask
        return result.astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)
        if self.bit_width == 64:
            # uint64 -> int64 view is already two's complement
            return arr.view(np.int64).astype(self._int_dtype)
        # Sign-extend: set all bits above bit_width to 1 for negative values
        sign_bit = np.uint64(1 << (self.bit_width - 1))
        is_negative = (arr & sign_bit) != 0
        # Create sign-extension mask: all 1s above bit_width
        extend_mask = np.uint64(np.iinfo(np.uint64).max) - np.uint64((1 << self.bit_width) - 1)
        extended = np.where(is_negative, arr | extend_mask, arr)
        return extended.view(np.int64).astype(self._int_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (self._min_signed, self._max_signed)

    def __repr__(self) -> str:
        return (
            f"TwosComplement(bit_width={self.bit_width}, "
            f"range=[{self._min_signed}, {self._max_signed}])"
        )
