from .util import split_pair, strip_comment


def load_config(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = strip_comment(raw).strip()
        if not line:
            continue
        key, value = split_pair(line)
        out[key] = value
    return out
