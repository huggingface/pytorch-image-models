

def fx_and(a: bool, b: bool) -> bool:
    """
    Symbolic tracing helper to substitute for normal usage of `* and *` within `torch._assert`.
    Hint: Symbolic tracing does not support control flow but since an `assert` is either a dead-end or not, this hack
    is okay.
    """
    return (a and b)


def fx_float_to_int(x: float) -> int:
    """
    Symbolic tracing helper to substitute for inbuilt `int`.
    Hint: Inbuilt `int` can't accept an argument of type `Proxy`
    """
    return int(x)