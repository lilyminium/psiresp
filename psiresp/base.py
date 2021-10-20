import inspect
from typing import Any, Optional, Union, no_type_check

import numpy as np
from pydantic import BaseModel

def _is_settable(member):
    return isinstance(member, property) and member.fset is not None


class Model(BaseModel):
    """Base class that all option-containing classes should subclass.

    This mostly contains the convenience methods:
        * :meth:`psiresp.base.Model.__post_init__`
            This is called after `__init__`
        * :meth:`psiresp.base.Model.from_model`
            This constructs an instance from compatible attributes of another object
    """

    class Config:
        arbitrary_types_allowed = True
        underscore_attrs_are_private = True
        validate_assignment = True
        json_encoders = {np.ndarray: lambda x: x.tolist()}

    @property
    def _clsname(self):
        return type(self).__name__

    def __setattr__(self, attr, value):
        try:
            super().__setattr__(attr, value)
        except ValueError as e:
            setters = inspect.getmembers(self.__class__, predicate=_is_settable)
            for propname, setter in setters:
                if propname == attr:
                    return object.__setattr__(self, propname, value)
            else:
                raise e

    @classmethod
    @no_type_check
    def _get_value(
        cls,
        v: Any,
        to_dict: bool,
        by_alias: bool,
        include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']],
        exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']],
        exclude_unset: bool,
        exclude_defaults: bool,
        exclude_none: bool,
    ) -> Any:

        if isinstance(v, set):
            v = list(v)
        return super()._get_value(v, to_dict, by_alias, include, exclude, exclude_unset, exclude_defaults, exclude_none)

