from typing import Type
from dataclasses import dataclass
from collections import UserDict

def create_field_property(field_name: str) -> property:
    """Helper function to create field property from name

    The property gets the value with cls.__getitem__; it
    sets the value with cls.__setitem__.
    """
    def getter(self):
        return self[field_name]
    
    def setter(self, value):
        self[field_name] = value
        
    return property(getter, setter, None)


def options(cls: Type) -> Type:
    """Decorator for options classes.
    
    It first decorates the class with the dataclass decorator,
    and then replaces the class attribute defaults with properties.
    """
    datacls = dataclass(cls)
    for k in datacls.__dataclass_fields__:
        setattr(cls, k, create_field_property(k))
    return cls


# TODO: is this just roundabout pydantic?
class AttrDict(UserDict):
    """Dictionary where keys are accessible as attributes.

    This should behave exactly as a normal dictionary,
    except that you can add keys and change values using
    attribute syntax.

    Examples
    --------

    ::

        >>> AttrDict()
        {}
        >>> AttrDict({"key": "value"})
        {"key": "value"}
        >>> AttrDict(key="value")
        {"key": "value"}
        >>> adct = AttrDict()
        >>> adct.key = "value"
        >>> adct
        {"key": "value"}
        >>> dict(**adct)
        {"key": "value"}
        >>> AttrDict(**adct)
        {"key": "value"}
    """
    
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"No attribute {attr}")

    def __setattr__(self, attr, value):
        if "data" not in self.__dict__:
            UserDict.__setattr__(self, "data", {})
        self[attr] = value

    def __getstate__(self):
        return dict(self)

    def __setitem__(self, key, item):
        try:
            assert key in type(self).__dataclass_fields__
        except AssertionError:
            # TODO: raise error?
            return
        except AttributeError:
            pass
        UserDict.__setitem__(self, key, item)
        
    def __setstate__(self, state):
        self.__init__(state)
    
    def __iadd__(self, other):
        self.update(other)
        return self
    
    def __radd__(self, other):
        if not other:
            return type(self)(**self)
        return other.__add__(self)


class OptionsBase(AttrDict):

    """Base class for options"""
    
    def __setitem__(self, key, item):
        fields = type(self).__dict__.get("__dataclass_fields__", {})
        if key not in fields:
            raise KeyError(f"{key} is not a keyword in {type(self).__name__}")
        super(OptionsBase, self).__setitem__(key, item)
