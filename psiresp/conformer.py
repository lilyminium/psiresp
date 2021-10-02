
from typing import Optional, List

from psiresp import base
from psiresp.orientation import Orientation
from psiresp.moleculebase import BaseMolecule

class Conformer(BaseMolecule):
    orientations: List[Orientation] = []
    is_optimized: bool = False

    def add_orientation_with_coordinates(self, coordinates):
        qcmol = self.qcmol_with_coordinates(coordinates)
        self.orientations.append(Orientation(qcmol=qcmol))
