import pytest

from psiresp.utils import meta as mtutils

DOCSTRING = """Class that contains a Psi4 molecule and a name

    .. note::

        Interesting note!

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule
    name: str
        Name

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


def test_split_docstring_into_parts():
    parts = mtutils.split_docstring_into_parts(DOCSTRING)
    # assert parts[""] == "Class that contains a Psi4 molecule and a name\n\n"
    assert len(parts) == 4
    assert len(parts["Parameters"]) == 6
    assert len(parts["Attributes"]) == 6
    assert len(parts["Examples"]) == 5


def test_join_split_docstring():
    parts = mtutils.split_docstring_into_parts(DOCSTRING)
    joined = mtutils.join_split_docstring(parts)
    assert joined == DOCSTRING


class Base:
    """
    Blahdiblah

    Parameters
    ----------
    test: str
        whatever

    Attributes
    ----------
    myval: int

    Examples
    --------
    this shouldn't show up

    """


EXTENDED = """Class that contains a Psi4 molecule and a name

    .. note::

        Interesting note!

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule
    name: str
        Name
    test: str
        whatever

    Attributes
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule
    name: str
        Name
    myval: int

    Examples
    --------
    ::
        mol = MoleculeMixin()
    .. versionchanged:: 1.0.0

"""


def test_extend_docstring_with_base():
    docstring = mtutils.extend_docstring_with_base(DOCSTRING, Base)
    assert docstring == EXTENDED
