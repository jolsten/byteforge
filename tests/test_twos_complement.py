"""Tests for TwosComplement encoding.

Reference vectors from type-convert/tests/test_twoscomp.py.
"""

import numpy as np
import pytest

from byteforge import TwosComplement

# Reference decode vectors: (bit_width, dn, expected_value)
# From type-convert/tests/test_twoscomp.py
DECODE_VECTORS: dict[int, list[tuple[int, int]]] = {
    3: [
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -4),
        (0b101, -3),
        (0b110, -2),
        (0b111, -1),
    ],
    8: [
        (0b00000000, 0),
        (0b00000001, 1),
        (0b00000010, 2),
        (0b01111110, 126),
        (0b01111111, 127),
        (0b10000000, -128),
        (0b10000001, -127),
        (0b10000010, -126),
        (0b11111110, -2),
        (0b11111111, -1),
    ],
    16: [
        (0x0000, 0),
        (0x7FFF, 2**15 - 1),
        (0x8000, -(2**15)),
        (0xFFFF, -1),
    ],
    24: [
        (0x000000, 0),
        (0x7FFFFF, 2**23 - 1),
        (0x800000, -(2**23)),
        (0xFFFFFF, -1),
    ],
    32: [
        (0x00000000, 0),
        (0x7FFFFFFF, 2**31 - 1),
        (0x80000000, -(2**31)),
        (0xFFFFFFFF, -1),
    ],
    48: [
        (0x000000000000, 0),
        (0x7FFFFFFFFFFF, 2**47 - 1),
        (0x800000000000, -(2**47)),
        (0xFFFFFFFFFFFF, -1),
    ],
    64: [
        (0x0000000000000000, 0),
        (0x7FFFFFFFFFFFFFFF, 2**63 - 1),
        (0x8000000000000000, -(2**63)),
        (0xFFFFFFFFFFFFFFFF, -1),
    ],
}

# Build parametrized list: (bit_width, dn, expected)
_decode_tests: list[tuple[int, int, int]] = []
for bw, cases in DECODE_VECTORS.items():
    for dn, val in cases:
        _decode_tests.append((bw, dn, val))

# Generated boundary cases for all widths 2-64
for size in range(2, 65):
    _decode_tests.append((size, 0, 0))  # zero
    _decode_tests.append((size, 1, 1))  # min positive
    _decode_tests.append((size, 2 ** (size - 1) - 1, 2 ** (size - 1) - 1))  # max positive
    _decode_tests.append((size, 2**size - 1, -1))  # -1
    _decode_tests.append((size, 1 << (size - 1), -(2 ** (size - 1))))  # max negative


@pytest.mark.parametrize("bit_width, dn, expected", _decode_tests)
def test_decode_reference_vectors(bit_width: int, dn: int, expected: int):
    enc = TwosComplement(bit_width)
    result = enc.decode(np.array([dn], dtype=np.uint64))
    assert int(result[0]) == expected


@pytest.mark.parametrize("bit_width, dn, expected", _decode_tests)
def test_encode_roundtrip(bit_width: int, dn: int, expected: int):
    """Encoding the decoded value should recover the original DN."""
    enc = TwosComplement(bit_width)
    re_encoded = enc.encode(np.array([expected]))
    assert int(re_encoded[0]) == dn


class TestTwosComplementEncode:
    def test_positive_value(self):
        result = TwosComplement(8).encode(np.array([10]))
        assert result[0] == 10

    def test_negative_value(self):
        result = TwosComplement(8).encode(np.array([-1]))
        assert result[0] == 255

    def test_negative_128(self):
        result = TwosComplement(8).encode(np.array([-128]))
        assert result[0] == 128

    def test_clamp_positive(self):
        result = TwosComplement(8).encode(np.array([200]))
        assert result[0] == 127

    def test_clamp_negative(self):
        result = TwosComplement(8).encode(np.array([-200]))
        assert result[0] == 128

    def test_encode_dtype(self):
        assert TwosComplement(8).encode(np.array([0])).dtype == np.uint8
        assert TwosComplement(16).encode(np.array([0])).dtype == np.uint16
        assert TwosComplement(32).encode(np.array([0])).dtype == np.uint32
        assert TwosComplement(64).encode(np.array([0])).dtype == np.uint64


class TestTwosComplementDecode:
    def test_decode_validates_dn(self):
        with pytest.raises(ValueError, match="out of range"):
            TwosComplement(8).decode(np.array([256], dtype=np.uint64))


class TestTwosComplementMisc:
    def test_value_range(self):
        assert TwosComplement(8).value_range == (-128, 127)
        assert TwosComplement(16).value_range == (-32768, 32767)

    def test_repr(self):
        r = repr(TwosComplement(8))
        assert "TwosComplement" in r
        assert "8" in r
