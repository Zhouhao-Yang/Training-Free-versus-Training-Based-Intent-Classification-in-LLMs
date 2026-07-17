"""Path helpers shared by experiment entry points."""

import re


_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def slugify_path_component(value: str) -> str:
    """Return a portable single path component without changing semantic IDs."""
    slug = _UNSAFE_PATH_CHARS.sub("_", str(value)).strip("._-")
    if not slug:
        raise ValueError(f"Cannot form a path component from {value!r}")
    return slug
