# Changelog

## [0.0.6] - 2026-03-08

### Added
- `from_range()` now validates that `min_value < max_value` on TwosComplement,
  OnesComplement, and OffsetBinary (raises `ValueError`).
- `decode()` now rejects float-dtype arrays with a clear `TypeError` instead of
  silently casting.
- Numeric `encode_errors` sentinels are now validated to fit in the output dtype
  at encode time (raises `ValueError`).
- `OverflowError` messages from `encode_errors="raise"` now include the
  offending values.

### Changed
- Error messages in `_validate_dns()` now show values as a Python list
  (truncated to first 5 elements) for consistency.

### Docs
- Expanded `LinearScaled` docstring to explain signed mode DN wrapping with an
  example.
- Documented why `IEEE754` does not accept an `encode_errors` parameter.
- Added docstrings to `BCD._decode_c()` and `DECFloat._init_masks()`.

## [0.0.4] - 2026-03-06

### Added
- Unified `encode_errors` parameter on all encodings (except IEEE754 and Boolean):
  `"clamp"` (default), `"raise"`, `"nan"`, or numeric sentinel.
- `BCD` now accepts `decode_errors` for invalid-nibble handling during decode.
- Float encodings (MilStd1750A, TIFloat, IBMFloat, DECFloat, DECFloatG) now
  support `encode_errors` for out-of-range detection.
- pytest-benchmark benchmarks for all encoding types.
- `__version__` attribute powered by setuptools-scm.
- Docstrings on all `_encode_py()`/`_decode_py()` private helpers.

### Changed
- **BREAKING**: `errors` parameter renamed to `encode_errors` on all encodings.
- **BREAKING**: `BCD(errors=...)` renamed to `BCD(decode_errors=...)`.
- Version management switched from hardcoded to setuptools-scm.

### Fixed
- Fixed mypy type annotation errors in `_base.py`, `twos_complement.py`,
  `ieee754.py`, and `bcd.py`.
