
from typing import Optional

from . import base
from .orientation import Orientation


class Conformer(base.Model):
    qcmol: "qcelemental.models.Molecule"
    orientations: List[Orientation] = []
    is_optimized: bool = False

    def add_orientation_with_coordinates(self, coordinates):
        qcmol = self.qcmol_with_coordinates(coordinates)
        self.orientations.append(Orientation(qcmol=qcmol))
