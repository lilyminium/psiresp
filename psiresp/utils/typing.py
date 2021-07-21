from typing import Generic, TypeVar

import numpy as np

DType = TypeVar("DType")


class TypedArray(np.ndarray, Generic[DType]):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field):
        dtype = field.sub_fields[0]
