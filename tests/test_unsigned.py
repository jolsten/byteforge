import numpy as np
import pytest

from byteforge import Unsigned


class TestUnsignedEncode:
    def test_encode_zero(self):
        result = Unsigned(8).encode(np.array([0]))
        assert result[0] == 0

    def test_encode_max(self):
        result = Unsigned(8).encode(np.array([255]))
        assert result[0] == 255

    def test_clamp_above_max(self):
        result = Unsigned(8).encode(np.array([300]))
        assert result[0] == 255

    def test_clamp_below_zero(self):
        result = Unsigned(8).encode(np.array([-5]))
        assert result[0] == 0

    def test_rounds_float(self):
        result = Unsigned(8).encode(np.array([3.7, 3.2]))
        assert result[0] == 4
        assert result[1] == 3

    def test_4bit_max(self):
        result = Unsigned(4).encode(np.array([15, 16]))
        assert result[0] == 15
        assert result[1] == 15

    def test_encode_dtype(self):
        assert Unsigned(8).encode(np.array([42])).dtype == np.uint8
        assert Unsigned(16).encode(np.array([42])).dtype == np.uint16
        assert Unsigned(32).encode(np.array([42])).dtype == np.uint32
        assert Unsigned(64).encode(np.array([42])).dtype == np.uint64

    def test_array_shape_preserved(self):
        values = np.array([0, 100, 200, 255])
        result = Unsigned(8).encode(values)
        assert result.shape == values.shape


class TestUnsignedDecode:
    def test_decode_zero(self):
        result = Unsigned(8).decode(np.array([0], dtype=np.uint64))
        assert result[0] == 0

    def test_decode_max(self):
        result = Unsigned(8).decode(np.array([255], dtype=np.uint64))
        assert result[0] == 255

    def test_decode_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            Unsigned(8).decode(np.array([256], dtype=np.uint64))

    def test_roundtrip(self):
        enc = Unsigned(8)
        values = np.array([0, 42, 127, 255])
        np.testing.assert_array_equal(enc.decode(enc.encode(values)), values)


class TestUnsignedMisc:
    def test_invalid_bit_width(self):
        with pytest.raises(ValueError):
            Unsigned(0)
        with pytest.raises(ValueError):
            Unsigned(65)

    def test_value_range(self):
        assert Unsigned(8).value_range == (0, 255)
        assert Unsigned(16).value_range == (0, 65535)

    def test_from_range(self):
        enc = Unsigned.from_range(max_value=255)
        assert enc.bit_width == 8

    def test_repr(self):
        assert "Unsigned" in repr(Unsigned(8))
        assert "8" in repr(Unsigned(8))
