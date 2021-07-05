from typing import List, Tuple

from pydantic import Field
from .io import IOOptions


class OrientationOptions(IOOptions):
    pass


class ConformerOptions(IOOptions):
    """Options for Conformers

    Parameters
    ----------
    optimize_geometry: bool
        Whether to optimize the geometry with Psi4
    weight: float
        How much to weight this conformer in the resp calculation
    orientation_options: OrientationOptions or dict
        Options for creating new Orientations
    n_reorientations: int
        Number of rigid-body reorientations to generate
    reorientations: list of tuples of 3 atom IDs
        Specific rigid-body reorientations to generate. Each
        number in the tuple represents an atom. The first atom
        in the tuple becomes the new origin; the second defines
        the x-axis; and the third defines the xy plane. This is
        indexed from 1, such that (3, 1, 4) refers to the third,
        first and fourth atoms respectively.
    n_rotations: int
        Number of rotations to generate
    rotations: list of tuples of 3 atom IDs
        Specific rigid-body rotations to generate. Each
        number in the tuple represents an atom. The first and
        second atoms in the tuple define a vector parallel to the
        x-axis, and the third atom defines a plane parallel to the
        xy plane. This is indexed from 1,
        such that (3, 1, 4) refers to the third,
        first and fourth atoms respectively.
    n_translations: int
        Number of translations to generate
    translations: list of tuples of 3 floats
        Specific translations to generate. Each item is a tuple of
        (x, y, z) coordinates. The entire molecule is translated
        by these coordinates.
    keep_original_conformer_geometry: bool
        Whether to keep the original conformation in the conformer.
    orientation_name_template: str
        Template to generate new orientation names
    """
    optimize_geometry: bool = False
    weight: float = 1
    orientation_options: OrientationOptions = Field(default_factory=OrientationOptions)

    n_reorientations: int = 0
    reorientations: List[Tuple[int, int, int]] = []
    n_translations: int = 0
    translations: List[Tuple[float, float, float]] = []
    n_rotations: int = 0
    rotations: List[Tuple[int, int, int]] = []
    keep_original_conformer_geometry: bool = False
    orientation_name_template: str = "{conformer.name}_o{counter:03d}"
