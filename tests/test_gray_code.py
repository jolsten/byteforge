"""Tests for GrayCode encoding."""

import numpy as np
import pytest

from byteforge import GrayCode


class TestGrayCodeEncode:
    def test_encode_zero(self):
        assert GrayCode(8).encode(np.array([0]))[0] == 0

    def test_encode_one(self):
        assert GrayCode(8).encode(np.array([1]))[0] == 1

    def test_encode_two(self):
        # 2 (binary 10) -> 10 XOR 01 = 11 = 3
        assert GrayCode(8).encode(np.array([2]))[0] == 3

    def test_encode_three(self):
        # 3 (binary 11) -> 11 XOR 01 = 10 = 2
        assert GrayCode(8).encode(np.array([3]))[0] == 2

    def test_known_4bit_sequence(self):
        gc = GrayCode(4)
        expected = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
        result = gc.encode(np.arange(16, dtype=np.float64))
        np.testing.assert_array_equal(result, expected)

    def test_clamp_above_max(self):
        gc = GrayCode(4)
        assert gc.encode(np.array([20]))[0] == gc.encode(np.array([15]))[0]

    def test_clamp_below_zero(self):
        gc = GrayCode(4)
        assert gc.encode(np.array([-3]))[0] == gc.encode(np.array([0]))[0]

    def test_encode_dtype(self):
        assert GrayCode(8).encode(np.array([42])).dtype == np.uint8
        assert GrayCode(16).encode(np.array([42])).dtype == np.uint16
        assert GrayCode(32).encode(np.array([42])).dtype == np.uint32
        assert GrayCode(64).encode(np.array([42])).dtype == np.uint64


class TestGrayCodeDecode:
    def test_decode_zero(self):
        assert GrayCode(8).decode(np.array([0], dtype=np.uint64))[0] == 0

    def test_decode_one(self):
        assert GrayCode(8).decode(np.array([1], dtype=np.uint64))[0] == 1

    def test_decode_three(self):
        # Gray 3 (binary 11) -> decode -> 2
        assert GrayCode(8).decode(np.array([3], dtype=np.uint64))[0] == 2

    def test_roundtrip_all_8bit(self):
        gc = GrayCode(8)
        values = np.arange(256, dtype=np.float64)
        np.testing.assert_array_equal(gc.decode(gc.encode(values)), values.astype(np.uint64))

    def test_roundtrip_all_4bit(self):
        gc = GrayCode(4)
        values = np.arange(16, dtype=np.float64)
        np.testing.assert_array_equal(gc.decode(gc.encode(values)), values.astype(np.uint64))

    def test_adjacent_values_differ_by_one_bit(self):
        gc = GrayCode(8)
        encoded = gc.encode(np.arange(256, dtype=np.float64))
        for i in range(255):
            xor = int(encoded[i]) ^ int(encoded[i + 1])
            assert bin(xor).count("1") == 1

    def test_adjacent_16bit_spot_check(self):
        gc = GrayCode(16)
        for i in [0, 255, 256, 1023, 1024, 65534]:
            g1 = gc.encode(np.array([i]))[0]
            g2 = gc.encode(np.array([i + 1]))[0]
            assert bin(int(g1) ^ int(g2)).count("1") == 1

    def test_decode_validates_dn(self):
        gc = GrayCode(8)
        with pytest.raises(ValueError):
            gc.decode(np.array([256], dtype=np.uint64))


class TestGrayCodeMisc:
    def test_1bit(self):
        gc = GrayCode(1)
        assert gc.encode(np.array([0]))[0] == 0
        assert gc.encode(np.array([1]))[0] == 1
        assert gc.decode(np.array([0], dtype=np.uint64))[0] == 0
        assert gc.decode(np.array([1], dtype=np.uint64))[0] == 1

    def test_value_range(self):
        assert GrayCode(8).value_range == (0, 255)

    def test_repr(self):
        r = repr(GrayCode(8))
        assert "GrayCode" in r
        assert "8" in r
