"""Tests for OnesComplement encoding."""

import numpy as np
import pytest

from byteforge import OnesComplement

# Reference decode vectors: (bit_width, dn, expected_value)
# One's complement: negative = bitwise NOT of magnitude
# Negative zero (all 1s) decodes to 0
DECODE_VECTORS: dict[int, list[tuple[int, int]]] = {
    3: [
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -3),
        (0b101, -2),
        (0b110, -1),
        (0b111, 0),  # negative zero
    ],
    8: [
        (0x00, 0),
        (0x01, 1),
        (0x7E, 126),
        (0x7F, 127),
        (0x80, -127),
        (0x81, -126),
        (0xFE, -1),
        (0xFF, 0),  # negative zero
    ],
    16: [
        (0x0000, 0),
        (0x7FFF, 2**15 - 1),
        (0x8000, -(2**15 - 1)),
        (0xFFFE, -1),
        (0xFFFF, 0),  # negative zero
    ],
    32: [
        (0x00000000, 0),
        (0x7FFFFFFF, 2**31 - 1),
        (0x80000000, -(2**31 - 1)),
        (0xFFFFFFFE, -1),
        (0xFFFFFFFF, 0),  # negative zero
    ],
    64: [
        (0x0000000000000000, 0),
        (0x7FFFFFFFFFFFFFFF, 2**63 - 1),
        (0x8000000000000000, -(2**63 - 1)),
        (0xFFFFFFFFFFFFFFFE, -1),
        (0xFFFFFFFFFFFFFFFF, 0),  # negative zero
    ],
}

_decode_tests: list[tuple[int, int, int]] = []
for bw, cases in DECODE_VECTORS.items():
    for dn, val in cases:
        _decode_tests.append((bw, dn, val))

# Generated boundary cases for all widths 2-64
for size in range(2, 65):
    mask = (1 << size) - 1
    max_pos = (1 << (size - 1)) - 1
    _decode_tests.append((size, 0, 0))  # zero
    _decode_tests.append((size, 1, 1))  # min positive
    _decode_tests.append((size, max_pos, max_pos))  # max positive
    _decode_tests.append((size, mask - 1, -1))  # -1
    _decode_tests.append((size, 1 << (size - 1), -max_pos))  # most negative
    _decode_tests.append((size, mask, 0))  # negative zero


@pytest.mark.parametrize("bit_width, dn, expected", _decode_tests)
def test_decode_reference_vectors(bit_width: int, dn: int, expected: int):
    enc = OnesComplement(bit_width)
    result = enc.decode(np.array([dn], dtype=np.uint64))
    assert int(result[0]) == expected


@pytest.mark.parametrize("bit_width, dn, expected", _decode_tests)
def test_encode_roundtrip(bit_width: int, dn: int, expected: int):
    """Encoding the decoded value should recover the original DN.

    Negative zero maps to positive zero, so skip negative-zero DNs.
    """
    enc = OnesComplement(bit_width)
    mask = (1 << bit_width) - 1
    if dn == mask and expected == 0:
        # Negative zero: encode(0) produces 0, not mask
        assert int(enc.encode(np.array([0]))[0]) == 0
        return
    re_encoded = enc.encode(np.array([expected]))
    assert int(re_encoded[0]) == dn


class TestOnesComplementEncode:
    def test_positive_value(self):
        result = OnesComplement(8).encode(np.array([10]))
        assert result[0] == 10

    def test_negative_value(self):
        # -1 in 8-bit one's complement: 0xFF - 1 = 0xFE
        result = OnesComplement(8).encode(np.array([-1]))
        assert result[0] == 0xFE

    def test_negative_max(self):
        # -127 in 8-bit one's complement: 0xFF - 127 = 0x80
        result = OnesComplement(8).encode(np.array([-127]))
        assert result[0] == 0x80

    def test_clamp_positive(self):
        result = OnesComplement(8).encode(np.array([200]))
        assert result[0] == 127

    def test_clamp_negative(self):
        result = OnesComplement(8).encode(np.array([-200]))
        assert result[0] == 0x80  # -127

    def test_zero(self):
        assert OnesComplement(8).encode(np.array([0]))[0] == 0

    def test_encode_dtype(self):
        assert OnesComplement(8).encode(np.array([0])).dtype == np.uint8
        assert OnesComplement(16).encode(np.array([0])).dtype == np.uint16
        assert OnesComplement(32).encode(np.array([0])).dtype == np.uint32
        assert OnesComplement(64).encode(np.array([0])).dtype == np.uint64


class TestOnesComplementDecode:
    def test_decode_validates_dn(self):
        with pytest.raises(ValueError, match="out of range"):
            OnesComplement(8).decode(np.array([256], dtype=np.uint64))

    def test_negative_zero_is_zero(self):
        result = OnesComplement(8).decode(np.array([0xFF], dtype=np.uint64))
        assert int(result[0]) == 0

    def test_decode_dtype(self):
        assert OnesComplement(8).decode(np.array([0], dtype=np.uint64)).dtype == np.int8
        assert OnesComplement(16).decode(np.array([0], dtype=np.uint64)).dtype == np.int16
        assert OnesComplement(32).decode(np.array([0], dtype=np.uint64)).dtype == np.int32
        assert OnesComplement(64).decode(np.array([0], dtype=np.uint64)).dtype == np.int64


class TestOnesComplementMisc:
    def test_value_range(self):
        assert OnesComplement(8).value_range == (-127, 127)
        assert OnesComplement(16).value_range == (-32767, 32767)

    def test_repr(self):
        r = repr(OnesComplement(8))
        assert "OnesComplement" in r
        assert "8" in r
