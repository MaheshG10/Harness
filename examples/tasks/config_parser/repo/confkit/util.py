def strip_comment(line: str) -> str:
    return line.split("#", 1)[0]


def split_pair(line: str) -> tuple[str, str]:
    parts = line.split("=")
    if len(parts) < 2:
        raise ValueError(f"not a key = value line: {line!r}")
    return parts[0].strip(), parts[1].strip()
