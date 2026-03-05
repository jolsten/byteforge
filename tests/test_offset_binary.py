"""Tests for OffsetBinary encoding."""

import numpy as np
import pytest

from byteforge import OffsetBinary


class TestOffsetBinaryEncode:
    def test_encode_zero(self):
        assert OffsetBinary(8).encode(np.array([0]))[0] == 128

    def test_encode_positive(self):
        ob = OffsetBinary(8)
        assert ob.encode(np.array([1]))[0] == 129
        assert ob.encode(np.array([127]))[0] == 255

    def test_encode_negative(self):
        ob = OffsetBinary(8)
        assert ob.encode(np.array([-1]))[0] == 127
        assert ob.encode(np.array([-128]))[0] == 0

    def test_clamp_above_max(self):
        ob = OffsetBinary(8)
        assert ob.encode(np.array([200]))[0] == ob.encode(np.array([127]))[0]

    def test_clamp_below_min(self):
        ob = OffsetBinary(8)
        assert ob.encode(np.array([-200]))[0] == ob.encode(np.array([-128]))[0]

    def test_rounds_float(self):
        ob = OffsetBinary(8)
        assert ob.encode(np.array([1.6]))[0] == ob.encode(np.array([2]))[0]
        assert ob.encode(np.array([1.4]))[0] == ob.encode(np.array([1]))[0]

    def test_encode_dtype(self):
        assert OffsetBinary(8).encode(np.array([0])).dtype == np.uint8
        assert OffsetBinary(16).encode(np.array([0])).dtype == np.uint16
        assert OffsetBinary(32).encode(np.array([0])).dtype == np.uint32


class TestOffsetBinaryDecode:
    def test_decode_zero_offset(self):
        assert OffsetBinary(8).decode(np.array([128], dtype=np.uint64))[0] == 0

    def test_decode_max(self):
        assert OffsetBinary(8).decode(np.array([255], dtype=np.uint64))[0] == 127

    def test_decode_min(self):
        assert OffsetBinary(8).decode(np.array([0], dtype=np.uint64))[0] == -128

    def test_roundtrip(self):
        ob = OffsetBinary(8)
        values = np.arange(-128, 128, dtype=np.float64)
        np.testing.assert_array_equal(ob.decode(ob.encode(values)), values.astype(np.int64))

    def test_decode_validates_dn(self):
        ob = OffsetBinary(8)
        with pytest.raises(ValueError):
            ob.decode(np.array([256], dtype=np.uint64))


class TestOffsetBinaryMisc:
    def test_value_range(self):
        assert OffsetBinary(8).value_range == (-128, 127)
        assert OffsetBinary(16).value_range == (-32768, 32767)

    def test_repr(self):
        r = repr(OffsetBinary(8))
        assert "OffsetBinary" in r
        assert "8" in r
