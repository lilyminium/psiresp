import concurrent.futures

import numpy as np

from .conformer import Conformer
from . import psi4utils, mixins


class Resp(mixins.RespMixin, mixins.MoleculeMixin):
    resp: Optional["MultiResp"] = None

    def __post_init__(self):
        super().__post_init__()
        self.conformers = []
        if self.resp is None:
            self.resp = self

    def generate_conformers(self):
        """Generate conformers from settings in conformer_generator.

        If no conformers result from those settings, the geometry of the
        input Psi4 molecule to RESP is used.
        """
        self.conformers = []
        self.conformer_generator.generate_conformer_geometries(self.psi4mol)
        for coordinates in self.conformer_generator.conformer_geometries:
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
            name = self.conformer_options.format_name(resp=self,
                                                      counter=counter)
        mol = psi4utils.psi4mol_with_coordinates(self.psi4mol,
                                                 coordinates_or_psi4mol,
                                                 name=name)
        default_kwargs = self.conformer_options.to_kwargs(**kwargs)
        conf = Conformer(resp=self, psi4mol=mol, name=name, **default_kwargs)
        self.conformers.append(conf)
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
