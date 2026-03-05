"""Tests for Encoding.to_bytes / from_bytes."""

import numpy as np
import pytest

from byteforge import BCD, IEEE754, MilStd1750A, TwosComplement, Unsigned


class TestToBytes:
    def test_8bit_single_value(self):
        enc = Unsigned(8)
        dns = enc.encode(np.array([0xAB]))
        raw = enc.to_bytes(dns)
        assert raw.shape == (1, 1)
        assert raw.dtype == np.uint8
        assert raw[0, 0] == 0xAB

    def test_16bit_big_endian(self):
        enc = Unsigned(16)
        dns = enc.encode(np.array([0x1234]))
        raw = enc.to_bytes(dns)
        assert raw.shape == (1, 2)
        assert list(raw[0]) == [0x12, 0x34]

    def test_16bit_little_endian(self):
        enc = Unsigned(16)
        dns = enc.encode(np.array([0x1234]))
        raw = enc.to_bytes(dns, byteorder="little")
        assert list(raw[0]) == [0x34, 0x12]

    def test_32bit(self):
        enc = Unsigned(32)
        dns = enc.encode(np.array([0xDEADBEEF]))
        raw = enc.to_bytes(dns)
        assert list(raw[0]) == [0xDE, 0xAD, 0xBE, 0xEF]

    def test_48bit(self):
        enc = MilStd1750A(48)
        dns = enc.encode(np.array([1.0]))
        raw = enc.to_bytes(dns)
        assert raw.shape == (1, 6)
        # Reconstruct and verify round-trip
        reconstructed = enc.from_bytes(raw)
        np.testing.assert_array_equal(dns, reconstructed)

    def test_12bit_uses_2_bytes(self):
        enc = BCD(12)
        dns = enc.encode(np.array([123]))
        raw = enc.to_bytes(dns)
        assert raw.shape == (1, 2)
        # 0x123 = 0x01, 0x23
        assert list(raw[0]) == [0x01, 0x23]

    def test_batch(self):
        enc = Unsigned(16)
        dns = enc.encode(np.array([0x0001, 0x00FF, 0x1234]))
        raw = enc.to_bytes(dns)
        assert raw.shape == (3, 2)
        assert list(raw[0]) == [0x00, 0x01]
        assert list(raw[1]) == [0x00, 0xFF]
        assert list(raw[2]) == [0x12, 0x34]

    def test_invalid_byteorder(self):
        enc = Unsigned(8)
        with pytest.raises(ValueError, match="byteorder"):
            enc.to_bytes(np.array([0]), byteorder="middle")


class TestFromBytes:
    def test_8bit(self):
        enc = Unsigned(8)
        raw = np.array([[0xAB]], dtype=np.uint8)
        result = enc.from_bytes(raw)
        assert result[0] == 0xAB
        assert result.dtype == np.uint8

    def test_16bit_big_endian(self):
        enc = Unsigned(16)
        raw = np.array([[0x12, 0x34]], dtype=np.uint8)
        result = enc.from_bytes(raw)
        assert result[0] == 0x1234
        assert result.dtype == np.uint16

    def test_16bit_little_endian(self):
        enc = Unsigned(16)
        raw = np.array([[0x34, 0x12]], dtype=np.uint8)
        result = enc.from_bytes(raw, byteorder="little")
        assert result[0] == 0x1234

    def test_wrong_byte_count(self):
        enc = Unsigned(16)
        raw = np.array([[0x12, 0x34, 0x56]], dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2 bytes"):
            enc.from_bytes(raw)

    def test_batch(self):
        enc = Unsigned(16)
        raw = np.array([[0x00, 0x01], [0x12, 0x34]], dtype=np.uint8)
        result = enc.from_bytes(raw)
        assert list(result) == [1, 0x1234]

    def test_invalid_byteorder(self):
        enc = Unsigned(8)
        with pytest.raises(ValueError, match="byteorder"):
            enc.from_bytes(np.array([[0]], dtype=np.uint8), byteorder="middle")


class TestRoundTrip:
    @pytest.mark.parametrize("bit_width", [4, 8, 12, 16, 24, 32, 48, 64])
    def test_unsigned_roundtrip(self, bit_width: int):
        enc = Unsigned(bit_width)
        values = np.array([0, 1, enc.max_unsigned])
        dns = enc.encode(values)
        raw = enc.to_bytes(dns)
        assert raw.shape == (3, (bit_width + 7) // 8)
        recovered = enc.from_bytes(raw)
        np.testing.assert_array_equal(dns, recovered)

    @pytest.mark.parametrize("bit_width", [8, 16, 32, 64])
    def test_twos_complement_roundtrip(self, bit_width: int):
        enc = TwosComplement(bit_width)
        values = np.array([0, 1, -1])
        dns = enc.encode(values)
        raw = enc.to_bytes(dns)
        recovered = enc.from_bytes(raw)
        np.testing.assert_array_equal(dns, recovered)

    @pytest.mark.parametrize("bit_width", [16, 32, 64])
    def test_ieee754_roundtrip(self, bit_width: int):
        enc = IEEE754(bit_width)
        values = np.array([0.0, 1.0, -1.0])
        dns = enc.encode(values)
        raw = enc.to_bytes(dns)
        recovered = enc.from_bytes(raw)
        np.testing.assert_array_equal(dns, recovered)

    @pytest.mark.parametrize("byteorder", ["big", "little"])
    def test_byteorder_roundtrip(self, byteorder: str):
        enc = Unsigned(32)
        dns = enc.encode(np.array([0xDEADBEEF]))
        raw = enc.to_bytes(dns, byteorder=byteorder)
        recovered = enc.from_bytes(raw, byteorder=byteorder)
        np.testing.assert_array_equal(dns, recovered)
