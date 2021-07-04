import concurrent.futures
from typing import Optional, Dict, List

import numpy as np
from pydantic import PrivateAttr, BaseModel, Field

from .conformer import Conformer
from . import psi4utils, rdutils
from .mixins import RespMoleculeOptions, IOMixin, MoleculeMixin, RespMixin
from .utils.io import datafile


class Resp(RespMoleculeOptions, RespMixin, IOMixin, MoleculeMixin):

    _resp: Optional["MultiResp"] = None
    _conformers: List[Conformer] = PrivateAttr(default_factory=list)

    @property
    def conformers(self):
        return self._conformers

    @conformers.setter
    def conformers(self, value):
        self._conformers = value

    @property
    def resp(self):
        if self._resp is None:
            return self
        return self._resp

    @resp.setter
    def resp(self, value):
        self._resp = value

    def generate_conformers(self):
        """Generate conformers from settings in conformer_generator.

        If no conformers result from those settings, the geometry of the
        input Psi4 molecule to RESP is used.
        """
        # self._conformers = []
        all_coordinates = self.generate_conformer_coordinates()
        if len(all_coordinates) > len(self.conformers):
            self._conformers = []
            for coordinates in all_coordinates:
                self.add_conformer(coordinates)
            if not self.conformers:
                self.add_conformer(self.psi4mol)

    def add_conformer(self,
                      coordinates_or_psi4mol: psi4utils.CoordinateInputs,
                      name: Optional[str] = None,
                      **kwargs) -> Conformer:
        """Create Conformer from Psi4 molecule or coordinates and add

        Parameters
        ----------
        coordinates_or_psi4mol: numpy.ndarray of coordinates or psi4.core.Molecule
            An array of coordinates or a Psi4 Molecule. If this is a molecule,
            the molecule is copied before creating the Conformer.
        name: str (optional)
            Name of the conformer. If not provided, one will be generated
            from the name template in the conformer_generator
        **kwargs:
            Arguments used to construct the Conformer.
            If not provided, the default specification given in
            :attr:`psiresp.resp.Resp.conformer_options`
            will be used.

        Returns
        -------
        conformer: Conformer
        """
        if name is None:
            counter = len(self.conformers) + 1
            name = self.conformer_name_template.format(resp=self,
                                                       counter=counter)
        mol = psi4utils.psi4mol_with_coordinates(self.psi4mol,
                                                 coordinates_or_psi4mol,
                                                 name=name)
        default_kwargs = self.conformer_options.to_kwargs(**kwargs)
        conf = Conformer(resp=self.resp, psi4mol=mol, name=name, **default_kwargs)
        self._conformers.append(conf)
        return conf

    def to_mda(self):
        """Create a MDAnalysis.Universe with charges

        Returns
        -------
        MDAnalysis.Universe
        """
        u = super().to_mda()
        if self.charges is not None:
            u.add_TopologyAttr("charges", self.charges)
        return u

    def get_sp3_ch_ids(self) -> Dict[int, List[int]]:
        """Get dictionary of sp3 carbon atom number to bonded hydrogen numbers.

        These atom numbers are indexed from 1. Each key is the number of an
        sp3 carbon. The value is the list of bonded hydrogen numbers.

        Returns
        -------
        c_h_dict: dict of {int: list of ints}
        """
        return psi4utils.get_sp3_ch_ids(self.psi4mol)

    @datafile(filename="conformer_coordinates.npy")
    def generate_conformer_coordinates(self):
        rdmol = rdutils.rdmol_from_psi4mol(self.psi4mol)
        if not self.keep_original_resp_geometry:
            rdmol.RemoveAllConformers()

        rdutils.generate_conformers(rdmol,
                                    n_conformers=self.max_generated_conformers,
                                    rmsd_threshold=self.min_conformer_rmsd)
        if self.minimize_conformer_geometries:
            rdutils.minimize_conformer_geometries(rdmol,
                                                  self.minimize_max_iter)
        return np.asarray(rdutils.get_conformer_coordinates(rdmol))
