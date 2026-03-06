import numpy as np
import numpy.typing as npt

from ._base import Encoding, _min_int_dtype
from ._registry import register


@register("ones_complement")
@register("1c")
class OnesComplement(Encoding):
    """Encodes signed values as one's complement integers.

    Negative values are represented by inverting all bits of the magnitude.
    The range is symmetric: [-(2^(N-1) - 1), +(2^(N-1) - 1)].

    Negative zero (all bits set) decodes to 0.
    """

    def __init__(self, bit_width: int, *, errors: str = "clamp") -> None:
        super().__init__(bit_width, errors=errors)
        self._mask = (1 << bit_width) - 1
        self._min_signed = -((1 << (bit_width - 1)) - 1)
        self._max_signed = (1 << (bit_width - 1)) - 1
        self._int_dtype: type[np.signedinteger] = _min_int_dtype(bit_width)

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
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
        negative = clamped < 0
        result = np.where(
            negative,
            self._mask - (-clamped).astype(np.uint64),
            clamped.astype(np.uint64),
        )
        return result.astype(self._dn_dtype)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)
        sign_bit = 1 << (self.bit_width - 1)
        is_negative = (arr & sign_bit) != 0
        neg_val: np.ndarray = -((self._mask - arr).astype(np.int64))
        pos_val: np.ndarray = arr.astype(np.int64)
        result = np.where(is_negative, neg_val, pos_val)
        return result.astype(self._int_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (self._min_signed, self._max_signed)

    @classmethod
    def from_range(cls, *, min_value: int, max_value: int) -> "OnesComplement":
        """Construct from the signed range that needs to be represented.

        One's complement has a symmetric range: ``[-(2^(N-1)-1), 2^(N-1)-1]``.

        Args:
            min_value: Most negative value to encode.
            max_value: Most positive value to encode.

        Returns:
            An OnesComplement encoding with the minimum required bit width.
        """
        magnitude = max(abs(min_value), abs(max_value))
        bit_width = magnitude.bit_length() + 1  # +1 for sign bit
        return cls(max(2, bit_width))

    def __repr__(self) -> str:
        return (
            f"OnesComplement(bit_width={self.bit_width}, "
            f"range=[{self._min_signed}, {self._max_signed}])"
        )
