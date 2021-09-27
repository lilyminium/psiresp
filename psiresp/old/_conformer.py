from typing import Optional

import qcelemental as qcel


class Conformer:
    qcmol: qcel.models.Molecule
    weight: Optional[float] = 1
    optimize: bool = True
