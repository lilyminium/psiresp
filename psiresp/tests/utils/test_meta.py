import pytest

from pydantic import BaseModel, Field

from psiresp.base import Model
from psiresp.utils import meta as mtutils

WRITTEN_DOCSTRING = """Class that contains a Psi4 molecule and a name

    .. note::

        Interesting note!

    Parameters
    ----------
    name: str
        My multiline
        description of
        a
        name
        with trailing spaces   
    psi4mol: psi4.core.Molecule
        Psi4 molecule

    Attributes
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule
    name: str
        Name

    Examples
    --------
    ::
        mol = MoleculeMixin()
    
    .. versionchanged:: 1.0.0

"""


@pytest.fixture()
def written_sections():
    return mtutils.split_docstring_into_parts(WRITTEN_DOCSTRING)


def test_split_docstring_into_parts(written_sections):
    assert len(written_sections) == 3
    assert "Parameters" in written_sections
    assert "Attributes" in written_sections
    assert "Examples" in written_sections

    # merged sections should be dict
    parameters = written_sections["Parameters"]
    assert list(parameters.keys()) == ["name", "psi4mol"]
    assert parameters["psi4mol"] == ["psi4mol : psi4.core.Molecule",
                                     "    Psi4 molecule"]
    assert parameters["name"] == ["name : str",
                                  "    My multiline",
                                  "    description of",
                                  "    a",
                                  "    name",
                                  "    with trailing spaces"]

    # examples should be string
    examples = written_sections["Examples"]
    assert isinstance(examples, str)


class Base(BaseModel):
    """
    Blahdiblah

    Parameters
    ----------
    repeated: int
        this should show up

    Examples
    --------
    this shouldn't show up

    """

    test: str = Field(description="whatever")
    repeated: int = Field(description="this should be overwritten")


def test_schema_to_docstring_sections():
    sections = mtutils.schema_to_docstring_sections(Base.__fields__)
    assert len(sections) == 2
    assert "Parameters" in sections
    assert "Attributes" in sections

    parameters = sections["Parameters"]
    assert len(parameters) == 2
    assert parameters["repeated"][1].strip() == "this should be overwritten"
    assert parameters["test"][1] == "    whatever"

    attributes = sections["Attributes"]
    assert attributes == parameters


def test_get_cls_docstring_sections_priority():
    sections = mtutils.get_cls_docstring_sections(Base)
    assert len(sections) == 3
    assert list(sections.keys()) == ["Parameters", "Attributes", "Examples"]

    parameters = sections["Parameters"]
    assert len(parameters) == 2
    assert parameters["repeated"][1] == "    this should show up"
    assert parameters["test"][1] == "    whatever"

    attributes = sections["Attributes"]
    assert len(attributes) == 2
    assert attributes["repeated"][1] == "    this should be overwritten"


EXTENDED_DOCSTRING = """Class that contains a Psi4 molecule and a name

    .. note::

        Interesting note!

    Parameters
    ----------
    psi4mol : psi4.core.Molecule
        Psi4 molecule
    test : str
        whatever
    repeated : int
        this should show up
    name : str
        My multiline
        description of
        a
        name
        with trailing spaces

    Attributes
    ----------
    psi4mol : psi4.core.Molecule
        Psi4 molecule
    test : str
        whatever
    repeated : int
        this should be overwritten
    name : str
        Name

    Examples
    --------
    ::
        mol = MoleculeMixin()
    
    .. versionchanged:: 1.0.0

"""


def test_model_meta():
    class Subclass(Model, Base):
        __doc__ = WRITTEN_DOCSTRING

    generated = Subclass.__doc__
    assert generated == EXTENDED_DOCSTRING
