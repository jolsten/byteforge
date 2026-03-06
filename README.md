# Byteforge

Encode and decode numbers as unsigned integer bit patterns using NumPy arrays.

Requires Python >= 3.9 and NumPy >= 2.0.

## Installation

```
pip install byteforge
```

## Usage

Every encoding exposes the same interface: `encode()` converts physical values to bit patterns, `decode()` converts back.

```python
import numpy as np
from byteforge import TwosComplement, IEEE754, Unsigned

# Integer encodings
tc = TwosComplement(16)
dns = tc.encode(np.array([-1, 0, 127]))   # array([65535, 0, 127], dtype=uint16)
tc.decode(dns)                              # array([-1, 0, 127], dtype=int16)

# Floating-point encodings
ieee = IEEE754(32)
dns = ieee.encode(np.array([3.14]))         # uint32 bit pattern
ieee.decode(dns)                            # array([3.14], dtype=float32)

# Unsigned with automatic bit width
u = Unsigned.from_range(max_value=1000)     # Unsigned(10)
```

### Registry factory

Create encodings by name using the registry:

```python
from byteforge import create_encoding

enc = create_encoding("twos_complement", 8)
enc = create_encoding("ieee32")            # aliases have default bit widths
enc = create_encoding("linear_scaled", 16, scale_factor=0.1, offset=10.0)
```

### Byte serialization

Convert encoded bit patterns to/from raw bytes:

```python
enc = Unsigned(16)
dns = enc.encode(np.array([256, 512]))
raw = enc.to_bytes(dns, byteorder="big")    # uint8 array, shape (2, 2)
enc.decode(enc.from_bytes(raw, byteorder="big"))  # roundtrip
```

For non-byte-aligned widths (e.g. 12-bit), values are zero-padded in the most
significant bits to fill whole bytes (2 bytes for 12-bit).

## Encodings

| Encoding | Registry names | Bit widths |
|---|---|---|
| `Unsigned` | `unsigned`, `u` | 1-64 |
| `TwosComplement` | `twos_complement`, `2c` | 2-64 |
| `OnesComplement` | `ones_complement`, `1c` | 2-64 |
| `OffsetBinary` | `offset_binary` | 1-64 |
| `GrayCode` | `gray_code`, `gray` | 1-64 |
| `BCD` | `bcd` | 4-64 (multiples of 4) |
| `Boolean` | `boolean` | 1 |
| `IEEE754` | `ieee754`, `ieee16`, `ieee32`, `ieee64` | 16, 32, 64 |
| `LinearScaled` | `linear_scaled` | 1-64 |
| `MilStd1750A` | `mil_std_1750a`, `1750a32`, `1750a48` | 32, 48 |
| `TIFloat` | `ti_float`, `ti32`, `ti40` | 32, 40 |
| `IBMFloat` | `ibm_float`, `ibm32`, `ibm64` | 32, 64 |
| `DECFloat` | `dec_float`, `dec32`, `dec64` | 32, 64 |
| `DECFloatG` | `dec_float_g`, `dec64g` | 64 |

## C Extension

Optional NumPy C ufuncs accelerate encodings that would otherwise require per-element loops
or multi-pass numpy operations. The C extension is compiled automatically during install and
falls back to pure Python when unavailable. Set `BYTEFORGE_NO_C=1` to force the pure-Python
path.

Encodings with C ufuncs: BCD, GrayCode (decode), MilStd1750A, TIFloat, IBMFloat,
DECFloat, and DECFloatG.

## Development

```
uv run pytest tests/ -v       # tests
uv run ruff check src/ tests/ # lint
uv run mypy                   # type check
```
