"""Tests for Boolean encoding."""

import numpy as np
import pytest

from byteforge import Boolean


class TestBooleanEncode:
    def test_encode_true(self):
        b = Boolean()
        assert b.encode(np.array([1]))[0] == 1
        assert b.encode(np.array([42]))[0] == 1

    def test_encode_false(self):
        assert Boolean().encode(np.array([0]))[0] == 0

    def test_encode_float_truthy(self):
        assert Boolean().encode(np.array([3.14]))[0] == 1

    def test_encode_float_zero(self):
        assert Boolean().encode(np.array([0.0]))[0] == 0

    def test_inverted_encode_true(self):
        b = Boolean(inverted=True)
        assert b.encode(np.array([1]))[0] == 0
        assert b.encode(np.array([42]))[0] == 0

    def test_inverted_encode_false(self):
        assert Boolean(inverted=True).encode(np.array([0]))[0] == 1

    def test_encode_dtype(self):
        assert Boolean().encode(np.array([1])).dtype == np.uint8


class TestBooleanDecode:
    def test_decode_one(self):
        assert Boolean().decode(np.array([1], dtype=np.uint64))[0] == 1

    def test_decode_zero(self):
        assert Boolean().decode(np.array([0], dtype=np.uint64))[0] == 0

    def test_inverted_decode_one(self):
        assert Boolean(inverted=True).decode(np.array([1], dtype=np.uint64))[0] == 0

    def test_inverted_decode_zero(self):
        assert Boolean(inverted=True).decode(np.array([0], dtype=np.uint64))[0] == 1

    def test_roundtrip_normal(self):
        b = Boolean()
        assert b.decode(b.encode(np.array([1])))[0] == 1
        assert b.decode(b.encode(np.array([0])))[0] == 0

    def test_roundtrip_inverted(self):
        b = Boolean(inverted=True)
        assert b.decode(b.encode(np.array([1])))[0] == 1
        assert b.decode(b.encode(np.array([0])))[0] == 0

    def test_decode_validates_dn(self):
        with pytest.raises(ValueError):
            Boolean().decode(np.array([2], dtype=np.uint64))


class TestBooleanMisc:
    def test_bit_width_is_one(self):
        assert Boolean().bit_width == 1

    def test_invalid_bit_width(self):
        with pytest.raises(ValueError, match="must be 1"):
            Boolean(bit_width=2)

    def test_value_range(self):
        assert Boolean().value_range == (0, 1)
        assert Boolean(inverted=True).value_range == (0, 1)

    def test_repr(self):
        assert "Boolean" in repr(Boolean())
        assert "inverted=False" in repr(Boolean())
