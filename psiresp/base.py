from collections import defaultdict
from typing import Dict, List
import re
import abc

from pydantic import BaseModel
from pydantic.main import ModelMetaclass

from . import utils


def split_docstring_into_parts(docstring: str) -> Dict[str, List[str]]:
    """Split docstring around headings"""
    parts = defaultdict(list)
    heading_pattern = "[ ]{4}[A-Z][a-z]+\s*\n[ ]{4}[-]{4}[-]+\s*\n"
    directive_pattern = "[ ]{4}\.\. [a-z]+:: .+\n"
    pattern = re.compile("(" + heading_pattern + "|" + directive_pattern + ")")
    sections = re.split(pattern, docstring)
    parts["base"] = sections.pop(0)
    while sections:
        heading_match, section = sections[:2]
        sub_pattern = "([A-Z][a-z]+|[ ]{4}\.\. [a-z]+:: .+\n)"
        heading = re.search(sub_pattern, heading_match).groups()[0]
        section = heading_match + section
        parts[heading] = section.split("\n")
        sections = sections[2:]
    return parts


def join_split_docstring(parts: Dict[str, List[str]]) -> str:
    """Join split docstring back into one string"""
    docstring = parts.pop("base", "")
    headings = ("Parameters", "Attributes", "Examples")
    for heading in headings:
        section = parts.pop(heading, [])
        docstring += "\n".join(section)
    for section in parts.values():
        docstring += "\n".join(section)
    return docstring


def extend_docstring_with_base(docstring: str, base_class: type) -> str:
    """Extend docstring with the parameters in `base_class`"""
    doc_parts = split_docstring_into_parts(docstring)
    base_parts = split_docstring_into_parts(base_class.__doc__)
    headings = ("Parameters", "Attributes", "Examples")
    for k in headings:
        if k in base_parts:
            section = base_parts.pop(k)
            if doc_parts.get(k):
                section = section[2:]
            doc_parts[k].extend(section)

    for k, lines in base_parts.items():
        if k != "base" and k in doc_parts:
            doc_parts[k].extend(lines[2:])

    return join_split_docstring(doc_parts)


class ModelMeta(ModelMetaclass):

    def __new__(cls, name, bases, clsdict):
        docstring = clsdict.get("__doc__", "")
        for base in bases:
            docstring = extend_docstring_with_base(docstring, base)
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
    def from_model(cls, obj, **kwargs) -> "Model":
        """Construct an instance from compatible attributes of
        ``object`` and ``kwargs``"""
        obj = obj.copy(deep=True)
        default_kwargs = {}
        for key in cls.__fields__:
            try:
                default_kwargs[key] = getattr(obj, key)
            except AttributeError:
                pass
        default_kwargs.update(kwargs)
        return cls(**default_kwargs)

    def to_kwargs(self, **kwargs):
        new = self.copy().dict()
        new.update(kwargs)
        return new
