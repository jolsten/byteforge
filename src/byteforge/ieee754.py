import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register

_VALID_WIDTHS = frozenset((16, 32, 64))
_FLOAT_DTYPES = {16: np.float16, 32: np.float32, 64: np.float64}
_UINT_DTYPES = {16: np.uint16, 32: np.uint32, 64: np.uint64}


@register("ieee754")
@register("ieee16", bit_width=16)
@register("ieee32", bit_width=32)
@register("ieee64", bit_width=64)
class IEEE754(Encoding):
    """Encodes float values in IEEE 754 format (16, 32, or 64-bit).

    Uses ``np.view()`` for zero-copy bit reinterpretation.

    Unlike other float encodings, IEEE754 does not accept an ``encode_errors``
    parameter because NumPy natively handles all IEEE 754 special values
    (NaN, ±Inf, subnormals) — there is no out-of-range condition to detect.
    """

    def __init__(self, bit_width: int) -> None:
        if bit_width not in _VALID_WIDTHS:
            raise ValueError(f"IEEE754 bit_width must be 16, 32, or 64, got {bit_width}")
        super().__init__(bit_width)
        self._float_dtype = _FLOAT_DTYPES[bit_width]
        self._uint_dtype = _UINT_DTYPES[bit_width]

    def _encode(self, values: npt.ArrayLike) -> np.ndarray:
        float_arr = np.asarray(values, dtype=self._float_dtype)
        return float_arr.view(self._uint_dtype)

    def _decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = self._validate_dns(dns)
        native_uint = arr.astype(self._uint_dtype)
        return native_uint.view(self._float_dtype)

    @property
    def value_range(self) -> tuple[float, float]:
        info: np.finfo = np.finfo(self._float_dtype)  # type: ignore[type-arg,arg-type]
        return (float(info.min), float(info.max))

    def __repr__(self) -> str:
        label = {16: "half", 32: "single", 64: "double"}[self.bit_width]
        return f"IEEE754(bit_width={self.bit_width}, precision={label})"
