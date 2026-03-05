"""Tests for the encoding registry."""

import pytest

from byteforge import Unsigned, create_encoding
from byteforge._registry import _ENCODING_REGISTRY


def test_all_encodings_registered():
    expected = {
        "unsigned",
        "u",
        "twos_complement",
        "2c",
        "ones_complement",
        "1c",
        "ieee754",
        "ieee16",
        "ieee32",
        "ieee64",
        "linear_scaled",
        "mil_std_1750a",
        "1750a32",
        "1750a48",
        "bcd",
        "gray_code",
        "gray",
        "offset_binary",
        "boolean",
        "ti_float",
        "ti32",
        "ti40",
        "ibm_float",
        "ibm32",
        "ibm64",
        "dec_float",
        "dec32",
        "dec64",
        "dec_float_g",
        "dec64g",
    }
    assert set(_ENCODING_REGISTRY.keys()) == expected


def test_create_encoding_unsigned():
    enc = create_encoding("unsigned", 8)
    assert isinstance(enc, Unsigned)
    assert enc.bit_width == 8


def test_create_encoding_with_kwargs():
    enc = create_encoding("linear_scaled", 16, scale_factor=0.1, offset=10.0)
    assert enc.bit_width == 16


def test_create_encoding_unknown():
    with pytest.raises(ValueError, match="Unknown encoding type"):
        create_encoding("nonexistent", 8)


def test_create_alias_ieee32():
    enc = create_encoding("ieee32")
    assert enc.bit_width == 32


def test_create_alias_ieee64():
    enc = create_encoding("ieee64")
    assert enc.bit_width == 64


def test_create_alias_1750a32():
    enc = create_encoding("1750a32")
    assert enc.bit_width == 32


def test_create_alias_1750a48():
    enc = create_encoding("1750a48")
    assert enc.bit_width == 48


def test_create_alias_ti32():
    enc = create_encoding("ti32")
    assert enc.bit_width == 32


def test_create_alias_ti40():
    enc = create_encoding("ti40")
    assert enc.bit_width == 40


def test_create_alias_ibm32():
    enc = create_encoding("ibm32")
    assert enc.bit_width == 32


def test_create_alias_ibm64():
    enc = create_encoding("ibm64")
    assert enc.bit_width == 64


def test_create_alias_dec32():
    enc = create_encoding("dec32")
    assert enc.bit_width == 32


def test_create_alias_dec64():
    enc = create_encoding("dec64")
    assert enc.bit_width == 64


def test_create_alias_dec64g():
    enc = create_encoding("dec64g")
    assert enc.bit_width == 64


def test_create_encoding_missing_bit_width():
    with pytest.raises(TypeError, match="bit_width is required"):
        create_encoding("unsigned")


def test_create_encoding_accepts_enum_like():
    class FakeEnum:
        value = "unsigned"

    enc = create_encoding(FakeEnum(), 8)  # type: ignore[arg-type]
    assert isinstance(enc, Unsigned)
