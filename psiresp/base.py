from collections import defaultdict
from typing import Dict, List

from pydantic import BaseModel

from . import utils


class ModelMeta(type):

    def __new__(cls, name, bases, clsdict):
        docstring = clsdict.get("__doc__", "")
        for base in bases:
            docstring = utils.extend_docstring_with_base(docstring, base)
        clsdict["__doc__"] = docstring
        return type.__new__(cls, name, bases, clsdict)


class Model(BaseModel, metaclass=ModelMeta):
    """Base class that all option-containing classes should subclass.

    This mostly contains the convenience methods:
        * :meth:`psiresp.base.Model.__post_init__`
            This is called after `__init__`
        * :meth:`psiresp.base.Model.from_model`
            This constructs an instance from compatible attributes of another object
    """

    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.__post_init__()

    def __post_init__(self):
        pass

    @ classmethod
    def from_model(cls, object, **kwargs) -> "Model":
        """Construct an instance from compatible attributes of
        ``object`` and ``kwargs``"""
        default_kwargs = {}
        for key in cls.__fields__:
            try:
                default_kwargs[key] = getattr(object, key)
            except AttributeError:
                pass
        default_kwargs.update(kwargs)
        return cls(**default_kwargs)

    def to_kwargs(self, **kwargs):
        new = self.copy().dict()
        new.update(kwargs)
        return new
