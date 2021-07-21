from typing import List, Tuple

from pydantic import Field, validator
from .io import IOOptions


class OrientationOptions(IOOptions):
    pass


class ConformerOptions(IOOptions):
    """Options for Conformers"""

    n_reorientations: int = Field(
        default=0,
        description="Number of rigid-body reorientations to generate")
    n_rotations: int = Field(
        default=0,
        description="Number of rotations to generate")
    n_translations: int = Field(
        default=0,
        description="Number of translations to generate")
    reorientations: List[Tuple[int, int, int]] = Field(
        default_factory=list,
        description=("Specific rigid-body reorientations to generate. Each"
                     "number in the tuple represents an atom. The first atom"
                     "in the tuple becomes the new origin; the second defines"
                     "the x-axis; and the third defines the xy plane. This is"
                     "indexed from 1, such that (3, 1, 4) refers to the third,"
                     "first and fourth atoms respectively."))
    translations: List[Tuple[float, float, float]] = Field(
        default_factory=list,
        description=("Specific translations to generate. Each item is a tuple of "
                     "(x, y, z) coordinates. The entire molecule is translated "
                     "by these coordinates. ")
    )
    rotations: List[Tuple[int, int, int]] = Field(
        default_factory=list,
        description=("Specific rigid-body rotations to generate. Each"
                     "number in the tuple represents an atom. The first and"
                     "second atoms in the tuple define a vector parallel to the"
                     "x-axis, and the third atom defines a plane parallel to the"
                     "xy plane. This is indexed from 1,"
                     "such that (3, 1, 4) refers to the third,"
                     "first and fourth atoms respectively.")
    )
    keep_original_conformer_geometry: bool = Field(
        default=False,
        description="Whether to keep the original conformation in the conformer.")
    orientation_name_template: str = Field(
        default="{conformer.name}_o{counter:03d}",
        description="Template to generate new orientation names")

    optimize_geometry: bool = Field(
        default=False,
        description="Whether to optimize the geometry")
    weight: float = Field(
        default=1,
        description="Weight of this conformer in the resp calculation")
    orientation_options: OrientationOptions = Field(
        default_factory=OrientationOptions,
        description="Options for creating new Orientations"
    )
