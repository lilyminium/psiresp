from typing import List

import psi4
import numpy as np
import rdkit
from rdkit import Chem


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

ACCEPTED_STRING_OUTPUT = {
    "pdb": Chem.MolToPDBBlock,
    "xyz": Chem.MolToXYZBlock,
}


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
    rdmol = psi4mol_to_rdmol(psi4mol)
    generate_conformers(rdmol)
    return get_conformer_coordinates(rdmol)


def get_conformer_coordinates(rdmol: rdkit.Chem.Mol) -> np.ndarray:
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


def add_conformer_from_coordinates(rdmol: rdkit.Chem.Mol,
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
    for i, xyz in enumerate(coordinates):
        x, y, z = map(float, xyz)
        conformer.SetAtomPosition(i, Geometry.Point3D(x, y, z))
    conformer.SetId(rdmol.GetNumConformers())
    rdmol.AddConformer(conformer)


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
    try:
        return rdmol_from_file(string)
    except ValueError:
        return rdmol_from_string(string)


def rdmol_from_file(filename: str) -> rdkit.Chem.Mol:
    suffix = filename.split(".")[-1]
    if suffix in ACCEPTED_FILE_FORMATS:
        parser = ACCEPTED_FILE_FORMATS[suffix]
        return parser(filename, removeHs=False, sanitize=True)
    file_formats = "'." + "', '".join(ACCEPTED_FILE_FORMATS) + "'"
    raise ValueError("File type cannot be read by RDKit. "
                     f"Please provide one of {file_formats}")


def rdmol_from_string(string: str) -> rdkit.Chem.Mol:
    for parser in ACCEPTED_STRING_PARSERS:
        try:
            rdmol = parser(string, removeHs=False, sanitize=True)
        except RuntimeError:
            continue
        if rdmol is not None:
            return rdmol

    import MDAnalysis as mda
    from MDAnalysis.topology.guessers import guess_atom_element
    u = mda.Universe(string)
    if not hasattr(u.atoms, "elements"):
        elements = [guess_atom_element(atom.type) for atom in u.atoms]
        u.add_TopologyAttr("elements", elements)
    return u.atoms.convert_to("RDKIT")


def rdmol_to_string(rdmol: rdkit.Chem.Mol, dtype="xyz", conf_id=0):
    return ACCEPTED_STRING_OUTPUT[dtype](rdmol, confId=conf_id)
