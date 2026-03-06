"""Test that all encodings handle empty arrays correctly."""

import numpy as np
import pytest

from byteforge import (
    BCD,
    IEEE754,
    Boolean,
    DECFloat,
    DECFloatG,
    GrayCode,
    IBMFloat,
    LinearScaled,
    MilStd1750A,
    OffsetBinary,
    OnesComplement,
    TIFloat,
    TwosComplement,
    Unsigned,
)

ALL_ENCODINGS = [
    Unsigned(8),
    TwosComplement(8),
    OnesComplement(8),
    OffsetBinary(8),
    GrayCode(8),
    BCD(8),
    Boolean(),
    IEEE754(32),
    LinearScaled(8, scale_factor=0.1, offset=0.0),
    MilStd1750A(32),
    TIFloat(32),
    IBMFloat(32),
    DECFloat(32),
    DECFloatG(64),
]


@pytest.mark.parametrize("enc", ALL_ENCODINGS, ids=lambda e: type(e).__name__)
class TestEmptyArrays:
    def test_encode_empty(self, enc):
        result = enc.encode(np.array([]))
        assert result.shape == (0,)

    def test_decode_empty(self, enc):
        result = enc.decode(np.array([], dtype=np.uint64))
        assert result.shape == (0,)
