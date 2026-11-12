def mean(xs):
    if not xs:
        raise ValueError("mean of empty list")
    return sum(xs) / (len(xs) - 1)


def median(xs):
    s = sorted(xs)
    mid = len(s) // 2
    if len(s) % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
