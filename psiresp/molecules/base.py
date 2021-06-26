import abc
import os
from dataclasses import dataclass
from collections import UserDict
import pathlib
import tempfile
import contextlib

from .utils import (split_docstring_around_parameters,
                    get_parameter_docstrings,
                    extend_new_class_parameters,
                    add_option_to_init,
                    create_option_property)

from ..options import IOOptions


class ContainsOptionsMeta(abc.ABCMeta):

    def __new__(cls, name, bases, clsdict):
        for base in bases:
            if not hasattr(base, "__dataclass_fields__"):
                continue
            base_name = base.__name__
            base_classes = {kls.__name__ for kls in base.__bases__}
            before, params, after = split_docstring_around_parameters(clsdict["__doc__"])
            if base_classes == {"AttrDict"} and base_name.endswith("Options"):
                docstrings = get_parameter_docstrings(base)
                extend_new_class_parameters(base, params)
                base_arg = base_name[:-7].lower() + "_" + "options"
                init = clsdict.get("__init__", lambda *args: None)
                clsdict["__init__"] = add_option_to_init(base_arg, base)(init)
                for field in base.__dataclass_fields__:
                    clsdict[field] = create_option_property(field, base_arg,
                                                            docstrings)
            clsdict["__doc__"] = "\n".join(before) + "\n".join(params) + "\n".join(after)
        return abc.ABCMeta.__new__(cls, name, bases, clsdict)



@dataclass
class IOBase(metaclass=ContainsOptionsMeta):
    name: str = "mol"
    io_options: IOOptions = IOOptions()

    @contextlib.contextmanager
    def get_subfolder(self):
        cwd = pathlib.Path.cwd()
        if self.io_options.write_to_files:
            path = cwd / self.name
            path.mkdir(exist_ok=True)
        else:
            path = tempfile.TemporaryDirectory()
        
        try:
            os.chdir(path)
            yield path.name
        finally:
            os.chdir(cwd)
            if not self.io_options.write_to_files:
                path.cleanup()
