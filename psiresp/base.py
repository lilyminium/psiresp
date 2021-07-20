import copy

from pydantic import BaseModel
from pydantic.main import ModelMetaclass

from .utils import meta as mtutils


class ModelMeta(ModelMetaclass):
    """This hacks classes to append the Parameters and Attributes
    docstrings in parent classes to the subclass, because life is
    too short to maintain docstrings"""

    def __new__(cls, name, bases, clsdict):
        docstring = clsdict.get("__doc__", "")
        for base in bases:
            docstring = mtutils.extend_docstring_with_base(docstring, base)
        clsdict["__doc__"] = docstring
        return ModelMetaclass.__new__(cls, name, bases, clsdict)

    def __init__(self, name, bases, clsdict):
        self.__doc__ = clsdict["__doc__"]


class Model(BaseModel, metaclass=ModelMeta):
    """Base class that all option-containing classes should subclass.

    This mostly contains the convenience methods:
        * :meth:`psiresp.base.Model.__post_init__`
            This is called after `__init__`
        * :meth:`psiresp.base.Model.from_model`
            This constructs an instance from compatible attributes of another object
    """

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = False
        validate_assignment = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self_fields = set(self.__fields__)
        for name, field in self.__fields__.items():
            if not field.default_factory:
                continue
            fieldcls = field.default_factory
            if (name.endswith("options")
                and name not in self.__fields_set__
                and isinstance(fieldcls, ModelMeta)):
                keywords = self_fields & set(fieldcls.__fields__)
                values = {k: getattr(self, k) for k in keywords}
                value = fieldcls(**values)
                setattr(self, name, value)

    @classmethod
    def from_model(cls, obj, **kwargs) -> "Model":
        """Construct an instance from compatible attributes of
        ``object`` and ``kwargs``"""
        default_kwargs = {}
        for key in cls.__fields__:
            if key == "psi4mol" and hasattr(obj, "psi4mol"):
                default_kwargs["psi4mol"] = obj.psi4mol.clone()
                continue
            try:
                default_kwargs[key] = copy.deepcopy(getattr(obj, key))
            except AttributeError:
                pass
        default_kwargs.update(kwargs)
        return cls(**default_kwargs)

    def to_kwargs(self, **kwargs):
        original = self.copy().dict()
        new = {k: getattr(self, k) for k in original}
        new.update(kwargs)

        return new
