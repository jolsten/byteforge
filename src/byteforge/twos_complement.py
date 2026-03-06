import numpy as np
import numpy.typing as npt

from ._base import Encoding, _min_int_dtype
from ._registry import register


@register("twos_complement")
@register("2c")
class TwosComplement(Encoding):
    """Encodes signed values as two's complement integers."""

    def __init__(self, bit_width: int, *, errors: str = "clamp") -> None:
        super().__init__(bit_width, errors=errors)
        self._min_signed = -(1 << (bit_width - 1))
        self._max_signed = (1 << (bit_width - 1)) - 1
        self._int_dtype: type[np.signedinteger] = _min_int_dtype(bit_width)

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        # For bit_width <= 53, float64 round-trip is fine.
        # For bit_width > 53, we accept int-typed arrays directly to avoid precision loss.
        arr = np.asarray(values)
        self._check_overflow(arr, self._min_signed, self._max_signed)
        if np.isdtype(arr.dtype, "integral"):
            clamped = np.clip(arr, self._min_signed, self._max_signed).astype(np.int64)
        else:
            clamped = np.clip(
                np.round(arr.astype(np.float64)),
                self._min_signed,
                self._max_signed,
            ).astype(np.int64)
        # Two's complement: negative values become unsigned via wrapping.
        result = clamped.view(np.uint64)
        if self.bit_width < 64:
            result = result & ((1 << self.bit_width) - 1)
        return result.astype(self._dn_dtype)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)
        if self.bit_width == 64:
            # uint64 -> int64 view is already two's complement
            return arr.view(np.int64).astype(self._int_dtype)
        # Sign-extend: set all bits above bit_width to 1 for negative values
        sign_bit = 1 << (self.bit_width - 1)
        is_negative = (arr & sign_bit) != 0
        # Create sign-extension mask: all 1s above bit_width
        extend_mask = np.iinfo(np.uint64).max - ((1 << self.bit_width) - 1)
        extended = np.where(is_negative, arr | extend_mask, arr)
        return extended.view(np.int64).astype(self._int_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (self._min_signed, self._max_signed)

    @classmethod
    def from_range(cls, *, min_value: int, max_value: int) -> "TwosComplement":
        """Construct from the signed range that needs to be represented.

        Args:
            min_value: Most negative value to encode.
            max_value: Most positive value to encode.

        Returns:
            A TwosComplement encoding with the minimum required bit width.
        """
        if min_value >= 0:
            # Need at least 1 sign bit + magnitude bits
            bit_width = max_value.bit_length() + 1
        else:
            neg_bits = ((-min_value) - 1).bit_length() + 1  # includes sign bit
            pos_bits = max_value.bit_length() + 1 if max_value > 0 else 1
            bit_width = max(neg_bits, pos_bits)
        return cls(max(1, bit_width))

    def __repr__(self) -> str:
        return (
            f"TwosComplement(bit_width={self.bit_width}, "
            f"range=[{self._min_signed}, {self._max_signed}])"
        )
