from typing import List

import psi4
import numpy as np
import rdkit
from rdkit import Chem

from . import psi4utils

ACCEPTED_FILE_FORMATS = {
    "pdb": Chem.MolFromPDBFile,
    "mol2": Chem.MolFromMol2File,
    "mol": Chem.MolFromMolFile,
}

ACCEPTED_STRING_PARSERS = (
    Chem.MolFromMol2Block,
    Chem.MolFromMolBlock,
    Chem.MolFromPDBBlock,
    Chem.MolFromRDKitSVG,
)


def rdmol_to_psi4mols(rdmol: rdkit.Chem.Mol,
                      name: str = "mol",
                      conformer_name_template: str = "{name}_c{i:03d}",
                      ) -> List[psi4.core.Molecule]:
    """Convert all conformers of an RDKit molecule to Psi4 molecules

    Parameters
    ----------
    rdmol: rdkit.Chem.Mol
        RDKit Molecule
    name: str
        Name of the overall molecule
    conformer_name_template: str
        Template used to format the conformer name

    Returns
    -------
    list of psi4.core.Molecules
    """
    mols = []
    for i in range(rdmol.GetNumConformers()):
        xyz = Chem.MolToXYZBlock(rdmol, confId=i)
        mol = psi4utils.psi4mol_from_xyz_string(xyz)
        mol.set_name(conformer_name_template.format(name=name, i=i+1))
        mol.activate_all_fragments()
        mols.append(mol)
    return mols


def generate_conformers(rdmol: rdkit.Chem.Mol,
                        n_conformers: int = 0,
                        rmsd_threshold: float = 1.5):
    """Generate conformers for an RDKit molecule.

    This does not clear existing conformers.

    Parameters
    ----------
    rdmol: rdkit.Chem.Mol
        RDKit Molecule
    n_conformers: int (optional)
        Maximum number of conformers to generate
    rmsd_threshold: float (optional)
        RMSD threshold used to prune new conformers
    """
    from rdkit.Chem import AllChem

    AllChem.EmbedMultipleConfs(rdmol, numConfs=n_conformers,
                               pruneRmsThresh=rmsd_threshold,
                               useRandomCoords=True,
                               clearConfs=False,
                               ignoreSmoothingFailures=True)


def minimize_conformer_geometries(rdmol: rdkit.Chem.Mol,
                                  minimize_max_iter: int = 2000):
    """Minimize conformer geometries of an RDKit molecule

    Parameters
    ----------
    minimize_max_iter: int (optional)
        Maximum number of iterations for minimization
    """
    from rdkit.Chem import AllChem
    # TODO: is UFF good?
    AllChem.UFFOptimizeMoleculeConfs(rdmol, numThreads=0,
                                     maxIters=minimize_max_iter)


def generate_conformer_coordinates(psi4mol: psi4.core.Molecule,
                                   max_generated_conformers: int = 0,
                                   min_conformer_rmsd: float = 1.5,
                                   **kwargs) -> np.ndarray:
    """Use RDKit to generate conformer coordinates from Psi4 molecule

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule
    max_generated_conformers: int (optional)
        Maximum number of conformers to generate
    min_conformer_rmsd: float (optional)
        RMSD threshold used to prune new conformers

    Returns
    -------
    coordinates: numpy.ndarray
        This has shape (n_conformers, n_atoms, 3)
    """
    rdmol = rdmol_from_psi4mol(psi4mol)
    generate_conformers(rdmol)
    return get_conformer_coordinates(rdmol)


def get_conformer_coordinates(self, rdmol: rdkit.Chem.Mol) -> np.ndarray:
    """Get conformer coordinates from RDKit molecule

    Parameters
    ----------
    rdmol: rdkit.Chem.Mol

    Returns
    -------
    coordinates: numpy.ndarray
        This has shape (n_conformers, n_atoms, 3)
    """
    n_conformers = rdmol.GetNumConformers()
    n_atoms = rdmol.GetNumAtoms()
    coordinates = np.zeros((n_conformers, n_atoms, 3))
    for i in range(n_conformers):
        coordinates[i] = rdmol.GetConformer(i).GetPositions()
    return coordinates


def add_conformer_from_coordinates(self,
                                   rdmol: rdkit.Chem.Mol,
                                   coordinates: np.ndarray):
    """Add conformer to RDKit from coordinate array

    Parameters
    ----------
    rdmol: rdkit.Chem.Mol
        RDKit Molecule
    coordinates: numpy.ndarray of floats
        Coordinates in angstrom
    """
    from rdkit import Geometry
    n_atoms = rdmol.GetNumAtoms()
    if coordinates.shape != (n_atoms, 3):
        raise ValueError("Shape of coordinates must be (n_atoms, 3)")
    conformer = Chem.Conformer(n_atoms)
    for i, (x, y, z) in enumerate(coordinates):
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))


def rdmol_from_psi4mol(psi4mol: psi4.core.Molecule) -> rdkit.Chem.Mol:
    """"Create RDKit molecule from Psi4 molecule

    Parameters
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule

    Returns
    -------
    rdmol: rdkit.Chem.Mol
        RDKit Molecule
    """
    mol = psi4utils.psi4mol_to_mol2_string(psi4mol)
    return Chem.MolFromMol2Block(mol)


def rdmol_from_file_or_string(string: str) -> rdkit.Chem.Mol:
    """Create an RDKit molecule from string or file

    Parameters
    ----------
    string: str
        Input string or filename. Accepted file formats include
        PDB, MOL2, MOL. Accepted string formats include
        PDB, MOL2, MOL, and RDKit SVGs. The file must end with
        the typical "pdb", "mol" or "mol2" suffix to be parsed.

    Returns
    -------
    rdmol: rdkit.Chem.Mol
        RDKit Molecule
    """
    suffix = string.split(".")[-1]
    if suffix in ACCEPTED_FILE_FORMATS:
        parser = ACCEPTED_FILE_FORMATS[suffix]
        return parser(string, removeHs=False, sanitize=True)

    for parser in ACCEPTED_STRING_PARSERS:
        rdmol = parser(string, removeHs=False, sanitize=True)
        if rdmol is not None:
            return rdmol

    import MDAnalysis as mda
    u = mda.Universe(string)
    return u.atoms.convert_to("RDKIT")
