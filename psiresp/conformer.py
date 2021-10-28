
from typing import Optional, List

from psiresp.orientation import Orientation
from psiresp.moleculebase import BaseMolecule


class Conformer(BaseMolecule):
    """Class to manage one conformer of a molecule.

    It must hold at least one orientation.
    """
    orientations: List[Orientation] = []
    is_optimized: bool = False
    _qc_id: Optional[int] = None

    def add_orientation_with_coordinates(self, coordinates):
        qcmol = self.qcmol_with_coordinates(coordinates)
        self.orientations.append(Orientation(qcmol=qcmol))
