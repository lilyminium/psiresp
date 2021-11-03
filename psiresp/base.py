import inspect
import hashlib
from typing import Any, Optional, Union, no_type_check

import numpy as np
from pydantic import BaseModel


def _is_settable(member):
    return isinstance(member, property) and member.fset is not None


def _to_immutable(obj):
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if isinstance(obj, set):
        return frozenset(obj)
    elif isinstance(obj, list):
        return tuple(_to_immutable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple((k, _to_immutable(v)) for k, v in sorted(obj.items()))
    return obj


class Model(BaseModel):
    """Base class that all classes should subclass.
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
            for propname, _ in setters:
                if propname == attr:
                    return object.__setattr__(self, propname, value)
            raise e

    def get_hash(self):
        mash = hashlib.sha1()
        mash.update(self.json().encode("utf-8"))
        return mash.hexdigest()

    def __hash__(self):
        return hash(self.get_hash())

    def __eq__(self, other):
        try:
            other_hash = hash(other)
        except TypeError:
            other_hash = hash(_to_immutable(other))
        return hash(self) == other_hash

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
