from collections import defaultdict
from typing import List, Dict, Tuple

from pydantic import Field
from .io import IOOptions


class OrientationOptions(IOOptions):
    pass


class ConformerOptions(IOOptions):
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
