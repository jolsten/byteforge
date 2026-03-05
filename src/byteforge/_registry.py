from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ._base import Encoding

_ENCODING_REGISTRY: dict[str, tuple[type, Optional[int]]] = {}


def register(name: str, *, bit_width: Optional[int] = None):
    """Decorator that registers an Encoding class under the given name.

    If *bit_width* is provided, it becomes the default for ``create_encoding``
    so callers can omit it (e.g. ``create_encoding("ieee32")``).
    """

    def decorator(cls):  # type: ignore[no-untyped-def]
        _ENCODING_REGISTRY[name] = (cls, bit_width)
        return cls

    return decorator


def create_encoding(
    encoding_type: str, bit_width: Optional[int] = None, **kwargs: Any
) -> "Encoding":
    """Factory: create an Encoding by its registered name.

    *bit_width* can be omitted for aliases that have a built-in default
    (e.g. ``"ieee32"``, ``"1750a32"``).

    Extra keyword arguments are forwarded to the constructor.
    """
    if hasattr(encoding_type, "value"):
        encoding_type = encoding_type.value
    if encoding_type not in _ENCODING_REGISTRY:
        raise ValueError(
            f"Unknown encoding type: {encoding_type!r}. "
            f"Available: {sorted(_ENCODING_REGISTRY)}"
        )
    cls, default_bw = _ENCODING_REGISTRY[encoding_type]
    actual_bw = bit_width if bit_width is not None else default_bw
    if actual_bw is None:
        raise TypeError(
            f"bit_width is required for encoding type {encoding_type!r}"
        )
    return cls(bit_width=actual_bw, **kwargs)
