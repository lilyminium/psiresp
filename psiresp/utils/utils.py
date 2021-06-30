
def is_iterable(obj: Any) -> bool:
    """Returns ``True`` if `obj` can be iterated over and is *not* a string
    nor a :class:`NamedStream`

    Adapted from MDAnalysis.lib.util.iterable
    """
    if isinstance(obj, str):
        return False
    if hasattr(obj, "__next__") or hasattr(obj, "__iter__"):
        return True
    try:
        len(obj)
    except (TypeError, AttributeError):
        return False
    return True


def as_iterable(obj: Any) -> Iterable:
    """Returns `obj` so that it can be iterated over.

    A string is *not* considered an iterable and is wrapped into a
    :class:`list` with a single element.

    See Also
    --------
    is_iterable
    """
    if not is_iterable(obj):
        obj = [obj]
    return obj
