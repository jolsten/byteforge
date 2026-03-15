# Byteforge Skill File

Encode and decode numbers as unsigned integer bit patterns (`np.ndarray`). Each encoding converts physical values (integers, floats) to/from fixed-width unsigned integers ("DNs" — data numbers).

## Installation

```bash
pip install byteforge
# or
uv add byteforge
```

Requires Python >=3.9, numpy >=2.0.

## Import Map

```python
from byteforge import (
    # Base & registry
    Encoding,              # abstract base class
    register,              # @register decorator for custom encodings
    create_encoding,       # factory: create_encoding("ieee32") -> IEEE754(32)

    # Integer encodings
    Unsigned,              # unsigned integer [0, 2^N-1]
    TwosComplement,        # two's complement signed [-2^(N-1), 2^(N-1)-1]
    OnesComplement,        # one's complement signed [-(2^(N-1)-1), 2^(N-1)-1]
    OffsetBinary,          # excess-N (add 2^(N-1) bias) [-2^(N-1), 2^(N-1)-1]
    GrayCode,              # Gray code [0, 2^N-1]
    BCD,                   # binary-coded decimal [0, 10^(N/4)-1]
    Boolean,               # single-bit boolean [0, 1]

    # Float encodings
    IEEE754,               # IEEE 754 (16/32/64-bit)
    LinearScaled,          # linear transfer function: EU = scale * DN + offset
    MilStd1750A,           # MIL-STD-1750A float (32/48-bit)
    TIFloat,               # Texas Instruments float (32/40-bit)
    IBMFloat,              # IBM hexadecimal float (32/64-bit)
    DECFloat,              # DEC VAX F4/D4 float (32/64-bit)
    DECFloatG,             # DEC VAX G4 float (64-bit only)
)
```

## Core API

Every encoding inherits from `Encoding` and shares these methods:

### Constructor

```python
Encoding(bit_width: int, *, encode_errors: Union[str, int, float] = "clamp")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bit_width` | int | **Required** | Number of bits, range [1, 64]. Some encodings restrict this further. Readable as `enc.bit_width` after construction. |
| `encode_errors` | str \| int \| float | `"clamp"` | Overflow policy for `encode()` (see table below) |

Other readable attributes: `enc.max_unsigned` → `2^bit_width - 1`, `enc.value_range` → `(min, max)` tuple.

**`encode_errors` policies:**

| Value | Behavior | Output dtype |
|-------|----------|-------------|
| `"clamp"` | Silently clamp to representable range | `_dn_dtype` (uint) |
| `"raise"` | Raise `OverflowError` if any value out of range | `_dn_dtype` (uint) |
| `"nan"` | Return `float64` with `np.nan` for out-of-range elements | float64 |
| numeric (int/float) | Substitute sentinel value for out-of-range elements | `_dn_dtype` (uint) |

### `encode(values) -> ndarray | scalar`

Encode physical values to unsigned integer bit patterns.

```python
enc = Unsigned(8)
enc.encode(42)                       # -> np.uint8(42)  (scalar in, scalar out)
enc.encode(np.array([0, 127, 255]))  # -> array([0, 127, 255], dtype=uint8)
enc.encode(np.array([300]))          # -> array([255], dtype=uint8)  (clamped)
```

- Accepts scalars, lists, or arrays
- Returns scalar if input is scalar, ndarray otherwise
- Output dtype: smallest unsigned integer dtype that fits `bit_width` (uint8/16/32/64)

### `decode(dns) -> ndarray | scalar`

Decode unsigned integer bit patterns back to physical values.

```python
enc = TwosComplement(8)
enc.decode(np.array([0, 127, 255], dtype=np.uint8))  # -> array([0, 127, -1], dtype=int8)
enc.decode(np.uint8(255))                             # -> np.int8(-1)  (scalar)
```

- **Raises** `TypeError` if input is float dtype
- **Raises** `ValueError` if any DN is negative or exceeds `2^bit_width - 1`

### `value_range` (property)

Returns `(min_value, max_value)` tuple of the representable physical range.

```python
TwosComplement(8).value_range  # -> (-128, 127)
Unsigned(16).value_range       # -> (0, 65535)
```

### `to_bytes(dns, byteorder="big") -> ndarray`

Convert encoded DNs to raw bytes.

```python
enc = Unsigned(16)
dns = enc.encode(np.array([0x1234, 0xABCD]))
raw = enc.to_bytes(dns)            # shape (2, 2), dtype uint8, big-endian
# raw = [[0x12, 0x34], [0xAB, 0xCD]]
raw_le = enc.to_bytes(dns, "little")  # [[0x34, 0x12], [0xCD, 0xAB]]
```

- Output shape: `(*dns.shape, n_bytes)` where `n_bytes = ceil(bit_width / 8)`
- For non-byte-aligned widths (e.g., 12-bit), MSBs are zero-padded
- **Raises** `ValueError` if `byteorder` not in `("big", "little")`

### `from_bytes(raw, byteorder="big") -> ndarray`

Reconstruct encoded DNs from raw bytes.

```python
enc = Unsigned(16)
raw = np.array([[0x12, 0x34], [0xAB, 0xCD]], dtype=np.uint8)
dns = enc.from_bytes(raw)  # -> array([0x1234, 0xABCD], dtype=uint16)
```

- Input must have shape `(..., n_bytes)` with dtype uint8
- **Raises** `ValueError` if `raw.shape[-1] != ceil(bit_width / 8)`
- **Raises** `ValueError` if `byteorder` not in `("big", "little")`

## Factory Function

```python
create_encoding(encoding_type: str, bit_width: int | None = None, **kwargs) -> Encoding
```

Create an encoding by registered name. Some aliases have built-in bit widths. All `**kwargs` are forwarded directly to the encoding class constructor.

```python
enc = create_encoding("ieee32")                                  # IEEE754(32)
enc = create_encoding("unsigned", 8)                             # Unsigned(8)
enc = create_encoding("linear_scaled", 16, scale_factor=0.1)    # LinearScaled(16, scale_factor=0.1)
enc = create_encoding("bcd", 12, decode_errors="nan")           # BCD(12, decode_errors="nan")
enc = create_encoding("linear_scaled", 16, scale_factor=0.5, offset=10.0, signed=True, encode_errors="raise")
```

**Raises** `ValueError` if `encoding_type` not registered.

## Custom Encodings with `@register`

```python
register(name: str, *, bit_width: int | None = None) -> Callable[[type], type]
```

Class decorator that registers an encoding under a name for use with `create_encoding()`.

```python
from byteforge import Encoding, register, create_encoding

@register("my_enc")
class MyEncoding(Encoding):
    def __init__(self, bit_width: int, *, encode_errors="clamp"):
        super().__init__(bit_width, encode_errors=encode_errors)

    def _encode(self, values):
        ...  # return uint ndarray of bit patterns

    def _decode(self, dns):
        ...  # return ndarray of physical values

    @property
    def value_range(self):
        return (0, self.max_unsigned)

# Now usable via factory:
enc = create_encoding("my_enc", 8)
```

**Parameters:**
- `name` (str): Registry key. Must be unique — **raises `ValueError`** if already registered.
- `bit_width` (int, optional): Default bit width for this alias. If provided, `create_encoding(name)` can omit the `bit_width` argument.

**Multiple aliases:** Decorate the same class multiple times:

```python
@register("foo")
@register("bar", bit_width=16)  # "bar" defaults to 16-bit
class FooEncoding(Encoding): ...

create_encoding("foo", 8)   # FooEncoding(8)
create_encoding("bar")      # FooEncoding(16) — uses default bit_width
```

**Subclass requirements:** Must implement `_encode(self, values) -> np.ndarray`, `_decode(self, dns) -> np.ndarray`, and the `value_range` property. The base class handles scalar/array dispatch, input validation (`decode` rejects floats and out-of-range DNs), and `encode_errors` overflow policy. Your `_encode`/`_decode` receive and return `np.ndarray` (at least 1-D). `_encode` should return dtype `self._dn_dtype` (smallest uint for the bit width). Use `self._dn_dtype`, `self.bit_width`, and `self.max_unsigned` in your implementations.

## Encoding Reference

### Integer Encodings

| Encoding | Registry Names | Bit Widths | Value Range (N bits) | Output Dtype (decode) |
|----------|---------------|------------|---------------------|-----------------------|
| `Unsigned` | `"unsigned"`, `"u"` | 1–64 | `[0, 2^N - 1]` | uint8/16/32/64 |
| `TwosComplement` | `"twos_complement"`, `"2c"` | 1–64 | `[-2^(N-1), 2^(N-1) - 1]` | int8/16/32/64 |
| `OnesComplement` | `"ones_complement"`, `"1c"` | 1–64 | `[-(2^(N-1) - 1), 2^(N-1) - 1]` | int8/16/32/64 |
| `OffsetBinary` | `"offset_binary"` | 1–64 | `[-2^(N-1), 2^(N-1) - 1]` | int8/16/32/64 |
| `GrayCode` | `"gray_code"`, `"gray"` | 1–64 | `[0, 2^N - 1]` | uint8/16/32/64 |
| `BCD` | `"bcd"` | 4–64 (multiples of 4) | `[0, 10^(N/4) - 1]` | uint8/16/32/64 |
| `Boolean` | `"boolean"` | 1 only | `[0, 1]` | uint8 |

### Float Encodings

| Encoding | Registry Names | Bit Widths | Exponent Bits | Mantissa Bits |
|----------|---------------|------------|--------------|---------------|
| `IEEE754` | `"ieee754"`, `"ieee16"`, `"ieee32"`, `"ieee64"` | 16, 32, 64 | 5/8/11 | 10/23/52 |
| `LinearScaled` | `"linear_scaled"` | 1–64 | N/A | N/A |
| `MilStd1750A` | `"mil_std_1750a"`, `"1750a32"`, `"1750a48"` | 32, 48 | 8 | 24 / 40 |
| `TIFloat` | `"ti_float"`, `"ti32"`, `"ti40"` | 32, 40 | 8 | 23 / 31 |
| `IBMFloat` | `"ibm_float"`, `"ibm32"`, `"ibm64"` | 32, 64 | 7 | 24 / 56 |
| `DECFloat` | `"dec_float"`, `"dec32"`, `"dec64"` | 32, 64 | 8 | 23 / 55 |
| `DECFloatG` | `"dec_float_g"`, `"dec64g"` | 64 only | 11 | 52 |

All float encodings decode to `float64`. All float encodings encode output is the smallest uint dtype (uint32/64).

### Per-Encoding Details

#### Unsigned

```python
enc = Unsigned(8)
enc.encode(np.array([0, 127, 255]))  # -> array([0, 127, 255], dtype=uint8)
enc.decode(np.array([0, 127, 255], dtype=np.uint8))  # -> array([0, 127, 255], dtype=uint8)
```

**Class method:** `Unsigned.from_range(max_value=1023)` — creates encoding with minimum bit width. Raises `ValueError` if `max_value < 0`.

#### TwosComplement

```python
enc = TwosComplement(8)
enc.encode(np.array([-128, 0, 127]))   # -> array([128, 0, 127], dtype=uint8)
enc.decode(np.array([128, 0, 127], dtype=np.uint8))  # -> array([-128, 0, 127], dtype=int8)
```

**Class method:** `TwosComplement.from_range(min_value=-1000, max_value=1000)` — creates encoding with minimum bit width. Raises `ValueError` if `min_value >= max_value`.

#### OnesComplement

```python
enc = OnesComplement(8)
enc.encode(np.array([-127, 0, 127]))   # -> array([128, 0, 127], dtype=uint8)
enc.decode(np.array([128, 0, 127], dtype=np.uint8))  # -> array([-127, 0, 127], dtype=int8)
```

Symmetric range: no representation for -(2^(N-1)). Negative zero (all bits set) decodes to 0.

**Class method:** `OnesComplement.from_range(min_value=-100, max_value=100)` — minimum bit width.

#### OffsetBinary

```python
enc = OffsetBinary(8)
enc.encode(np.array([-128, 0, 127]))   # -> array([0, 128, 255], dtype=uint8)
enc.decode(np.array([0, 128, 255], dtype=np.uint8))  # -> array([-128, 0, 127], dtype=int8)
```

Maps signed values to unsigned by adding bias of `2^(N-1)`.

**Class method:** `OffsetBinary.from_range(min_value=-500, max_value=500)` — minimum bit width.

#### GrayCode

```python
enc = GrayCode(4)
enc.encode(np.array([0, 1, 2, 3]))  # -> array([0, 1, 3, 2], dtype=uint8)
enc.decode(np.array([0, 1, 3, 2], dtype=np.uint8))  # -> array([0, 1, 2, 3], dtype=uint8)
```

Adjacent values differ by exactly one bit.

**Class method:** `GrayCode.from_range(max_value=255)` — minimum bit width.

#### BCD (Binary-Coded Decimal)

```python
enc = BCD(12)  # 3 digits (12/4), max value 999
enc.encode(np.array([0, 123, 999]))
enc.decode(enc.encode(np.array([0, 123, 999])))  # -> array([0, 123, 999])
```

**Constructor:**
```python
BCD(
    bit_width: int,            # Must be multiple of 4, range [4, 64]
    *,
    encode_errors = "clamp",   # Overflow policy
    decode_errors = "raise"    # Invalid nibble policy: "raise", "nan", or numeric sentinel
)
```

**`decode_errors`** handles invalid BCD nibbles (digit >=10 in a 4-bit group):

| Value | Behavior |
|-------|----------|
| `"raise"` | Raise `ValueError` (default) |
| `"nan"` | Return `float64` with `np.nan` for invalid elements |
| numeric | Substitute sentinel value |

**Raises** `ValueError` if `bit_width` is not a multiple of 4 or outside [4, 64].

**Class method:** `BCD.from_range(max_value=9999, decode_errors="nan")` — minimum bit width (multiple of 4).

#### Boolean

```python
enc = Boolean()              # bit_width defaults to 1
enc.encode(np.array([0, 1, True, False, 42]))  # -> array([0, 1, 1, 0, 1], dtype=uint8)

enc_inv = Boolean(inverted=True)
enc_inv.encode(np.array([True, False]))  # -> array([0, 1], dtype=uint8)
```

**Constructor:** `Boolean(bit_width=1, *, inverted=False)`
- `bit_width` must be 1. **Raises** `ValueError` if not.
- `inverted`: if True, truthy encodes to 0, falsy encodes to 1.
- Does NOT accept `encode_errors`.

#### IEEE754

```python
enc = IEEE754(32)
enc.encode(np.array([1.0, -1.0, 3.14]))  # -> uint32 bit patterns
enc.decode(np.array([0x3F800000], dtype=np.uint32))  # -> array([1.0])
```

**Constructor:** `IEEE754(bit_width: int)` — Does NOT accept `encode_errors`.

**Supported bit widths:** 16, 32, 64 only. **Raises** `ValueError` otherwise.

Uses `np.view()` for zero-copy bit reinterpretation. Handles all IEEE 754 special values (NaN, ±Inf, subnormals).

#### LinearScaled

```python
enc = LinearScaled(16, scale_factor=0.01, offset=100.0)
enc.encode(np.array([100.0, 100.5, 200.0]))
enc.decode(enc.encode(np.array([100.0, 100.5, 200.0])))  # ≈ [100.0, 100.5, 200.0]
```

**Constructor:**
```python
LinearScaled(
    bit_width: int,
    *,
    scale_factor: float,     # **Required**. Must be non-zero. Raises ValueError if 0.
    offset: float = 0.0,     # Physical zero point
    signed: bool = False,    # If True, DN range uses two's complement wrapping
    encode_errors = "clamp"
)
```

**Transfer function:**
```
encode:  DN = clamp(round((EU - offset) / scale_factor), min_dn, max_dn)
decode:  EU = scale_factor * DN + offset
```

**DN range:**
- `signed=False`: `[0, 2^N - 1]`
- `signed=True`: `[-2^(N-1), 2^(N-1) - 1]` (stored as unsigned via two's complement)

**Class method:**
```python
LinearScaled.from_range(bit_width, *, physical_min, physical_max, signed=False)
```
Derives `scale_factor` and `offset` from physical range. **Raises** `ValueError` if `physical_min >= physical_max`.

#### MilStd1750A

```python
enc = MilStd1750A(32)
enc.encode(np.array([1.0, -1.0, 0.0]))
enc.decode(enc.encode(np.array([1.0, -1.0, 0.0])))  # ≈ [1.0, -1.0, 0.0]
```

**Supported bit widths:** 32, 48 only. **Raises** `ValueError` otherwise.

**Bit layout:**
- 32-bit: `[Mantissa_24 | Exponent_8]`
- 48-bit: `[Mantissa_high_24 | Exponent_8 | Mantissa_low_16]` — exponent between mantissa halves

Has C ufunc acceleration (falls back to Python if unavailable).

#### TIFloat

```python
enc = TIFloat(32)
enc.encode(np.array([1.0, -2.5]))
enc.decode(enc.encode(np.array([1.0, -2.5])))  # ≈ [1.0, -2.5]
```

**Supported bit widths:** 32, 40 only. **Raises** `ValueError` otherwise.

**Bit layout:** `[Exponent_8 | Sign_1 | Mantissa_23/31]`

Has C ufunc acceleration.

#### IBMFloat

```python
enc = IBMFloat(32)
enc.encode(np.array([1.0, -0.5]))
enc.decode(enc.encode(np.array([1.0, -0.5])))  # ≈ [1.0, -0.5]
```

**Supported bit widths:** 32, 64 only. **Raises** `ValueError` otherwise.

**Bit layout:** `[Sign_1 | Exponent_7 | Mantissa_24/56]` — base-16 exponent with bias 64.

Has C ufunc acceleration.

#### DECFloat / DECFloatG

```python
enc = DECFloat(32)   # F4 format
enc64 = DECFloat(64) # D4 format
enc_g = DECFloatG()  # G4 format (64-bit only, bit_width defaults to 64)

enc.encode(np.array([1.0]))
enc.decode(enc.encode(np.array([1.0])))  # ≈ [1.0]
```

**DECFloat supported bit widths:** 32, 64 only. **Raises** `ValueError` otherwise.

**DECFloatG:** 64-bit only (defaults to 64). **Raises** `ValueError` if `bit_width != 64`.

**DECFloat bit layout:** `[Sign_1 | Exponent_8 | Mantissa_23/55]` — hidden bit at 0.5, bias 128.

**DECFloatG bit layout:** `[Sign_1 | Exponent_11 | Mantissa_52]` — hidden bit at 0.5, bias 1024.

Both have C ufunc acceleration.

## Registry Aliases with Default Bit Widths

These aliases can be used with `create_encoding()` without specifying `bit_width`:

| Alias | Class | Default bit_width |
|-------|-------|-------------------|
| `"ieee16"` | IEEE754 | 16 |
| `"ieee32"` | IEEE754 | 32 |
| `"ieee64"` | IEEE754 | 64 |
| `"1750a32"` | MilStd1750A | 32 |
| `"1750a48"` | MilStd1750A | 48 |
| `"ti32"` | TIFloat | 32 |
| `"ti40"` | TIFloat | 40 |
| `"ibm32"` | IBMFloat | 32 |
| `"ibm64"` | IBMFloat | 64 |
| `"dec32"` | DECFloat | 32 |
| `"dec64"` | DECFloat | 64 |
| `"dec64g"` | DECFloatG | 64 |

Aliases without defaults (must provide `bit_width`): `"unsigned"`, `"u"`, `"twos_complement"`, `"2c"`, `"ones_complement"`, `"1c"`, `"offset_binary"`, `"gray_code"`, `"gray"`, `"bcd"`, `"boolean"`, `"ieee754"`, `"linear_scaled"`, `"mil_std_1750a"`, `"ti_float"`, `"ibm_float"`, `"dec_float"`, `"dec_float_g"`.

## Output Dtype Rules

| Encoding type | `encode()` output | `decode()` output |
|---------------|-------------------|-------------------|
| Unsigned, GrayCode, BCD | uint8/16/32/64 | uint8/16/32/64 |
| TwosComplement, OnesComplement, OffsetBinary | uint8/16/32/64 | int8/16/32/64 |
| Boolean | uint8 | uint8 |
| IEEE754 | uint16/32/64 | float16/32/64 |
| All other floats (LinearScaled, MilStd1750A, TIFloat, IBMFloat, DECFloat, DECFloatG) | uint32/64 | float64 |

The unsigned output dtype is always the smallest that fits `bit_width`: 1–8 bits → uint8, 9–16 → uint16, 17–32 → uint32, 33–64 → uint64. Similarly for signed: int8/16/32/64.

## Error Handling Summary

| Situation | Exception | When |
|-----------|-----------|------|
| `bit_width` not in [1, 64] | `ValueError` | Constructor |
| `bit_width` not valid for encoding (e.g., IEEE754 with 24) | `ValueError` | Constructor |
| BCD `bit_width` not multiple of 4 | `ValueError` | Constructor |
| Boolean `bit_width != 1` | `ValueError` | Constructor |
| `scale_factor == 0` in LinearScaled | `ValueError` | Constructor |
| Invalid `encode_errors` string | `ValueError` | Constructor |
| Invalid `encode_errors` type | `TypeError` | Constructor |
| `encode()` with out-of-range values and `encode_errors="raise"` | `OverflowError` | `encode()` |
| `decode()` with float input | `TypeError` | `decode()` |
| `decode()` with negative DN or DN > max_unsigned | `ValueError` | `decode()` |
| BCD `decode()` with invalid nibble and `decode_errors="raise"` | `ValueError` | `decode()` |
| `to_bytes`/`from_bytes` with invalid byteorder | `ValueError` | `to_bytes()`/`from_bytes()` |
| `from_bytes` with wrong byte count per element | `ValueError` | `from_bytes()` |

## C Acceleration

Optional C ufuncs accelerate encodings with per-element loops. Set `BYTEFORGE_NO_C=1` to force pure Python fallback.

**Has C ufuncs:** BCD, GrayCode (decode only), MilStd1750A, TIFloat, IBMFloat, DECFloat, DECFloatG.

**No C ufuncs (already vectorized):** Unsigned, TwosComplement, OnesComplement, IEEE754, OffsetBinary, Boolean, LinearScaled, GrayCode encode.

## Complete Examples

### Roundtrip encode/decode

```python
import numpy as np
from byteforge import TwosComplement

enc = TwosComplement(16)
values = np.array([-1000, 0, 1000, 32767])
dns = enc.encode(values)           # uint16 bit patterns
recovered = enc.decode(dns)        # int16 physical values
np.testing.assert_array_equal(recovered, values)
```

### Byte serialization roundtrip

```python
import numpy as np
from byteforge import Unsigned

enc = Unsigned(12)
dns = enc.encode(np.array([0, 2048, 4095]))
raw = enc.to_bytes(dns, byteorder="big")   # shape (3, 2), dtype uint8
recovered = enc.from_bytes(raw, byteorder="big")
np.testing.assert_array_equal(dns, recovered)
```

### Overflow handling

```python
import numpy as np
from byteforge import Unsigned

# Clamp (default)
enc = Unsigned(8)
enc.encode(np.array([300]))  # -> array([255])

# Raise
enc = Unsigned(8, encode_errors="raise")
try:
    enc.encode(np.array([300]))
except OverflowError:
    print("Out of range!")

# NaN sentinel
enc = Unsigned(8, encode_errors="nan")
result = enc.encode(np.array([300]))  # -> array([nan])  (float64)

# Numeric sentinel
enc = Unsigned(8, encode_errors=0)
result = enc.encode(np.array([300]))  # -> array([0])
```

### LinearScaled with from_range

```python
import numpy as np
from byteforge import LinearScaled

# Map physical range [0, 100] to 10-bit unsigned
enc = LinearScaled.from_range(10, physical_min=0.0, physical_max=100.0)
dns = enc.encode(np.array([0.0, 50.0, 100.0]))
recovered = enc.decode(dns)  # ≈ [0.0, 50.0, 100.0] (quantization error)
```

### Factory usage

```python
from byteforge import create_encoding

enc = create_encoding("ieee32")
enc = create_encoding("2c", 16)
enc = create_encoding("bcd", 16, decode_errors="nan")
enc = create_encoding("linear_scaled", 12, scale_factor=0.5, offset=10.0)
```

### BCD with invalid nibble handling

```python
import numpy as np
from byteforge import BCD

# Default: raise on invalid
enc = BCD(8)
try:
    enc.decode(np.array([0xAB], dtype=np.uint8))  # A and B are invalid BCD
except ValueError:
    print("Invalid BCD!")

# NaN for invalid
enc = BCD(8, decode_errors="nan")
result = enc.decode(np.array([0xAB], dtype=np.uint8))  # -> array([nan])
```

### Scalar I/O

```python
from byteforge import Unsigned

enc = Unsigned(8)
result = enc.encode(42)       # returns np.uint8(42), not an array
result = enc.decode(result)   # returns np.uint8(42)
```
