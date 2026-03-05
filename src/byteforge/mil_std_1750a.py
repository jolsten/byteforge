import numpy as np
import numpy.typing as npt

from ._base import Encoding
from ._registry import register

_VALID_WIDTHS = frozenset((32, 48))


@register("mil_std_1750a")
@register("1750a32", bit_width=32)
@register("1750a48", bit_width=48)
class MilStd1750A(Encoding):
    """MIL-STD-1750A floating point encoding (32-bit or 48-bit).

    32-bit layout (MSB -> LSB): [mantissa 24 bits][exponent 8 bits]
    48-bit layout (MSB -> LSB): [mantissa_high 24 bits][exponent 8 bits][mantissa_low 16 bits]

    Value = M * 2^(E - (mantissa_bits - 1))

    Where mantissa_bits = 24 (32-bit) or 40 (48-bit).
    """

    def __init__(self, bit_width: int) -> None:
        if bit_width not in _VALID_WIDTHS:
            raise ValueError(f"MilStd1750A bit_width must be 32 or 48, got {bit_width}")
        super().__init__(bit_width)
        self._mantissa_bits = bit_width - 8

    def _pack(self, M_u: np.ndarray, E_u: np.ndarray) -> np.ndarray:
        """Pack mantissa and exponent into the word format."""
        if self.bit_width == 32:
            return (M_u << np.uint64(8)) | E_u
        else:
            # 48-bit: [M_high_24 | E_8 | M_low_16]
            M_high = (M_u >> np.uint64(16)) & np.uint64(0xFFFFFF)
            M_low = M_u & np.uint64(0xFFFF)
            return (M_high << np.uint64(24)) | (E_u << np.uint64(16)) | M_low

    def _unpack(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unpack word into (M_u, E_u) unsigned arrays."""
        if self.bit_width == 32:
            E_u = arr & np.uint64(0xFF)
            M_u = (arr >> np.uint64(8)) & np.uint64((1 << 24) - 1)
        else:
            # 48-bit: [M_high_24 | E_8 | M_low_16]
            M_high = (arr >> np.uint64(24)) & np.uint64(0xFFFFFF)
            E_u = (arr >> np.uint64(16)) & np.uint64(0xFF)
            M_low = arr & np.uint64(0xFFFF)
            M_u = (M_high << np.uint64(16)) | M_low
        return M_u, E_u

    def encode(self, values: npt.ArrayLike) -> np.ndarray:
        fval = np.asarray(values, dtype=np.float64)
        mbits = self._mantissa_bits
        result = np.zeros(fval.shape, dtype=np.uint64)

        nonzero = fval != 0.0
        if not np.any(nonzero):
            return result

        abs_vals = np.abs(fval[nonzero])
        E = np.floor(np.log2(abs_vals)).astype(np.int64) + 1
        M = np.round(fval[nonzero] * np.exp2((mbits - 1 - E).astype(np.float64))).astype(
            np.int64
        )

        M_min = -(1 << (mbits - 1))
        M_max = (1 << (mbits - 1)) - 1
        M = np.clip(M, M_min, M_max)
        E = np.clip(E, -128, 127)

        M_u = M.astype(np.uint64) & np.uint64((1 << mbits) - 1)
        E_u = E.astype(np.uint64) & np.uint64(0xFF)
        result[nonzero] = self._pack(M_u, E_u)

        return result.astype(self._dn_dtype)

    def decode(self, dns: npt.ArrayLike) -> np.ndarray:
        arr = np.asarray(dns, dtype=np.uint64)
        self._validate_dns(arr)
        mbits = self._mantissa_bits

        M_u, E_u = self._unpack(arr)

        E_i = E_u.astype(np.int64)
        E = np.where(E_i >= 128, E_i - 256, E_i)

        M_i = M_u.astype(np.int64)
        M = np.where(M_i >= (1 << (mbits - 1)), M_i - (1 << mbits), M_i)

        return M.astype(np.float64) * np.exp2(E.astype(np.float64) - (mbits - 1))

    def __repr__(self) -> str:
        return (
            f"MilStd1750A(bit_width={self.bit_width}, "
            f"mantissa={self._mantissa_bits}, exponent=8)"
        )
