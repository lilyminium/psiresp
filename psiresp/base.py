

from .utils import cached, datafile, prefix_file

class CachedMeta(type):
    def __init__(cls, name, bases, classdict):
        items = [(x, y) for x, y in classdict.items() if x.startswith("get_")]

        type.__init__(type, name, bases, classdict)
        for name, method in items:
            key = name[4:]
            if key in classdict:
                continue
            setattr(cls, key, cached(method))


class CachedBase(metaclass=CachedMeta):
    name = ""
    verbose = False
    force = False

    def __init__(self, force=False, verbose=False, name=None):
        self.verbose = verbose
        self.force = force
        self.name = name
        self._cache = {}
    
    
