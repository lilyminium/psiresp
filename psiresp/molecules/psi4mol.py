from dataclasses import dataclass

import psi4

from .base import ContainsOptionsBase
from ..options import IOOptions

@dataclass
class MoleculeBase(ContainsOptionsBase):

    psi4mol: psi4.core.Molecule