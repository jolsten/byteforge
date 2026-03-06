from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from ._base import Encoding

_ENCODING_REGISTRY: dict[str, tuple[type, Optional[int]]] = {}


def register(name: str, *, bit_width: Optional[int] = None) -> Callable[[type], type]:
    """Decorator that registers an Encoding class under the given name.

    If ``bit_width`` is provided, it becomes the default for
    ``create_encoding`` so callers can omit it
    (e.g. ``create_encoding("ieee32")``).

    Args:
        name: Registry key for the encoding.
        bit_width: Default bit width for this alias.

    Returns:
        Class decorator that registers the encoding.
    """

    def decorator(cls: type) -> type:
        if name in _ENCODING_REGISTRY:
            existing_cls, _ = _ENCODING_REGISTRY[name]
            raise ValueError(
                f"Encoding name {name!r} is already registered to "
                f"{existing_cls.__name__}"
            )
        _ENCODING_REGISTRY[name] = (cls, bit_width)
        return cls

    return decorator


def create_encoding(
    encoding_type: str, bit_width: Optional[int] = None, **kwargs: Any
) -> "Encoding":
    """Create an Encoding by its registered name.

    Args:
        encoding_type: Registry key (e.g. ``"ieee32"``, ``"bcd"``).
        bit_width: Number of bits. Can be omitted for aliases with a
            built-in default (e.g. ``"ieee32"``, ``"1750a32"``).
        **kwargs: Extra keyword arguments forwarded to the constructor.

    Returns:
        An Encoding instance.

    Raises:
        ValueError: If ``encoding_type`` is not registered.
        TypeError: If ``bit_width`` is required but not provided.
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
