def slugify(text: str, max_length: int = 50) -> str:
    """Turn a title into a URL slug.

    Lowercase, keep only a-z and 0-9, join words with single hyphens,
    no leading or trailing hyphens, and cut to max_length without
    leaving a trailing hyphen.
    """
    raise NotImplementedError
