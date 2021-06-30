
import io
import functools
import pathlib
import tempfile
import contextlib
import os
from dataclasses import dataclass, field, Field

from pydantic import BaseModel

import psi4
import numpy as np

from .mixins import IOMixin

from . import constants, psi4utils

class Model(BaseModel):
    
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__(*args, **kwargs)
        self.__post_init__()

    def __post_init__(self):
        pass

    @classmethod
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

