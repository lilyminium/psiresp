from typing import List
from dataclasses import field

import numpy.typing as npt
import psi4
import rdkit

from .. import rdutils
from .base import options, OptionsBase

@options
class ConformerOptions(OptionsBase):
    conformer_geometries: npt.ArrayLike = field(default_factory=list)
    max_generated_conformers: int = 0
    min_conformer_rmsd: float = 1.5
    minimize_conformer_geometries: bool = False
    minimize_max_iter: int = 2000
    keep_original_resp_geometry: bool = True


    def generate_conformer_geometries(self, psi4mol: psi4.core.Molecule):
        rdmol = rdutils.rdmol_from_psi4mol(psi4mol)
        self._generate_conformers_from_rdmol(rdmol)

    def _generate_conformers_from_rdmol(self, rdmol: rdkit.Chem.Mol):
        if not self.keep_original_resp_geometry:
            rdmol.RemoveAllConformers()
        
        for coordinates in self.conformer_geometries:
            rdutils.add_conformer_from_coordinates(rdmol, coordinates)

        rdutils.generate_conformers(rdmol,
                                    n_conformers=self.max_generated_conformers,
                                    rmsd_threshold=self.min_conformer_rmsd)
        if self.minimize_conformer_geometries:
            rdutils.minimize_conformer_geometries(rdmol,
                                                  self.minimize_max_iter)
        self.conformer_geometries = rdutils.get_conformer_coordinates(rdmol)

    
