from typing import Optional  # , TYPE_CHECKING

import qcelemental as qcel
from rdkit import Chem

from . import base, rdutils
from .orutils import generate_atom_combinations


class BaseMolecule(base.Model):
    qcmol: qcel.models.Molecule
    _rdmol: Optional[Chem.Mol] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rdmol = rdutils.rdmol_from_qcelemental(self.qcmol)
        extras = self.qcmol.__dict__["extras"]
        if extras is None:
            extras = {}
        extras[rdutils.OFF_SMILES_ATTRIBUTE] = rdutils.rdmol_to_smiles(self._rdmol)
        self.qcmol.__dict__["extras"] = extras

    def qcmol_with_coordinates(self, coordinates, units="angstrom"):
        dct = self.qcmol.dict()
        dct["geometry"] = coordinates * qcel.constants.conversion_factor(units, "bohr")
        return qcel.models.Molecule(**dct)

    @property
    def n_atoms(self):
        return self.coordinates.shape[0]

    @property
    def coordinates(self):
        return self.qcmol.geometry * qcel.constants.conversion_factor("bohr", "angstrom")

    # def __hash__(self):
    #     dct = self.dict()
    #     dct.pop("qcmol")
    #     return hash((self.qcmol.get_hash(), base._to_immutable(dct)))

    # def __eq__(self, other):
    #     return hash(self) == hash(other)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash(self.qcmol.get_hash())

    def _get_qcmol_repr(self):
        qcmol_attrs = [f"{x}={getattr(self.qcmol, x)}" for x in ["name"]]
        return ", ".join(qcmol_attrs)

    def generate_atom_combinations(self, n_combinations=None):
        atoms = generate_atom_combinations(self.qcmol.symbols)
        if n_combinations is None or n_combinations < 0:
            return atoms

        return [next(atoms) for i in range(n_combinations)]

    def get_smarts_matches(self, smiles):
        if self._rdmol is None:
            raise ValueError("Could not create a valid RDKit molecule from QCElemental molecule")
        rdmol = Chem.Mol(self._rdmol)
        Chem.SanitizeMol(rdmol, Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY)
        Chem.SetAromaticity(rdmol, Chem.AromaticityModel.AROMATICITY_MDL)

        query = Chem.MolFromSmarts(smiles)
        if query is None:
            raise ValueError(f"RDKit could not parse SMARTS {smiles}")

        index_to_map = {i: atom.GetAtomMapNum()
                        for i, atom in enumerate(query.GetAtoms())
                        if atom.GetAtomMapNum()}

        map_list = sorted(index_to_map, key=index_to_map.get)
        full_matches = rdmol.GetSubstructMatches(query, uniquify=True,
                                                 useChirality=True)
        if not map_list:
            return full_matches
        else:
            matches = [tuple(match[i] for i in map_list) for match in full_matches]
            return matches

    def to_smiles(self, mapped=True):
        return rdutils.rdmol_to_smiles(self._rdmol, mapped=mapped)
