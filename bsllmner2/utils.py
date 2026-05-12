"""Small, dependency-free utilities shared across the package.

Living here (rather than in :mod:`bsllmner2.pipeline` or :mod:`bsllmner2.config`)
keeps the import graph acyclic: anything in the package may freely import from
here.
"""

from datetime import datetime, timezone

_TRUTHY = frozenset({"true", "1", "yes", "on"})
_FALSY = frozenset({"false", "0", "no", "off"})


def get_now() -> datetime:
    """Return the current UTC time as a timezone-aware ``datetime``."""
    return datetime.now(timezone.utc)


def parse_bool(value: str | bool, *, strict: bool = False) -> bool:
    """Parse a string or bool as a boolean.

    The recognised truthy values are ``true``/``1``/``yes``/``on`` and the
    recognised falsy values are ``false``/``0``/``no``/``off`` (all
    case-insensitive, surrounding whitespace stripped).

    Args:
        value: The value to parse. Booleans are passed through unchanged.
        strict: When ``True``, unrecognised strings raise ``ValueError``.
            When ``False`` (the default), unrecognised strings return ``False``
            so environment-variable parsing does not crash on user typos.

    Raises:
        ValueError: When ``strict`` is ``True`` and *value* is not recognised.

    """
    if isinstance(value, bool):
        return value
    lower = value.strip().lower()
    if lower in _TRUTHY:
        return True
    if lower in _FALSY:
        return False
    if strict:
        raise ValueError(f"Invalid boolean value: {value!r}")
    return False
