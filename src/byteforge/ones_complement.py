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

    def __init__(self, bit_width: int) -> None:
        super().__init__(bit_width)
        self._mask = (1 << bit_width) - 1
        self._min_signed = -((1 << (bit_width - 1)) - 1)
        self._max_signed = (1 << (bit_width - 1)) - 1
        self._int_dtype: type[np.signedinteger] = _min_int_dtype(bit_width)

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(values)
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

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)
        sign_bit = 1 << (self.bit_width - 1)
        is_negative = (arr & sign_bit) != 0
        neg_val = -((self._mask - arr).astype(np.int64))
        pos_val = arr.astype(np.int64)
        result = np.where(is_negative, neg_val, pos_val)
        return result.astype(self._int_dtype)

    @property
    def value_range(self) -> tuple[int, int]:
        return (self._min_signed, self._max_signed)

    def __repr__(self) -> str:
        return (
            f"OnesComplement(bit_width={self.bit_width}, "
            f"range=[{self._min_signed}, {self._max_signed}])"
        )
