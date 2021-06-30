import abc
import os
from dataclasses import dataclass, field, Field
from collections import UserDict
import pathlib
import tempfile
import contextlib

import psi4

from .base import OptionsBase
from .utils import (split_docstring_around_parameters,
                    get_parameter_docstrings,
                    extend_new_class_parameters,
                    add_option_to_init,
                    create_option_property)

class ContainsOptionsMeta(abc.ABCMeta):

    def __init__(cls, name, bases, clsdict):
        docstring = cls.__doc__
        init = cls.__init__
        before, params, after = split_docstring_around_parameters(docstring)
        for name, value in clsdict.items():
            if isinstance(value, Field) and issubclass(value.default_factory, OptionsBase):
                base = value.default_factory
                if not hasattr(base, "__dataclass_fields__"):
                    continue
                base_name = base.__name__
                if not base_name.endswith("Options"):
                    continue
                docstrings = get_parameter_docstrings(base)
                extend_new_class_parameters(base, params)
                base_argname = base_name[:-7].lower() + "_" + "options"
                init = add_option_to_init(base_argname, base)(init)
                for field in base.__dataclass_fields__:
                    prop = create_option_property(field, base_argname, docstrings)
                    setattr(cls, field, prop)
        cls.__doc__ = "\n".join([before, "", "\n".join(params), after])
        cls.__init__ = init


class ContainsOptionsBase(metaclass=ContainsOptionsMeta):
    
    def to_dict(self):
        fields = self.__dict__.get("__dataclass_fields__", {})
        state = {}
        for k in fields:
            v = getattr(self, k)
            if isinstance(v, OptionsBase):
                v = type(v)(**v)
            state[k] = v
        return state

