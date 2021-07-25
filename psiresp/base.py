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
        schema_parts = mtutils.schema_to_docstring_sections(clsdict)

        # ordered highest to lowest priority
        sections = [schema_parts]
        for base in bases:
            sections.append(mtutils.get_cls_docstring_sections(base))

        prioritize = ["psi4mol", "conformer"]
        docstring = mtutils.create_docstring_from_sections(docstring, sections,
                                                           order_first=prioritize)
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
        # extra = "allow"

    def __init__(self, *args, **kwargs):
        extra = {k: v for k, v in kwargs.items() if k not in self.__fields__}
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
        for k, v in extra.items():
            setattr(self, k, v)

    __getattr__ = object.__getattribute__

    def __setattr__(self, attrname, value):
        # if it's a property, just allow
        try:
            super().__setattr__(attrname, value)
        except ValueError:
            if attrname in dir(self):
                object.__setattr__(self, attrname, value)
            else:
                raise

    def __setstate__(self, state):
        super().__setstate__(state)
        for k, v in self.__fields__.items():
            # validate
            setattr(self, k, getattr(self, k))
        self._init_private_attributes()

    # def _fields_from_kwargs(self, **kwargs):
    #     return {k: v for k, v in kwargs.items() if k in self.__fields__}

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
