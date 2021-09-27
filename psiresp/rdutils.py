
import itertools
from typing import TYPE_CHECKING, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

if TYPE_CHECKING:
    import rdkit
    import qcelemental

BONDTYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}


def rdmol_from_qcelemental(qcmol: "qcelemental.models.Molecule"):
    rwmol = Chem.RWMol()
    for symbol in qcmol.symbols:
        rwmol.AddAtom(symbol)
    for i, j, d in qcmol.connectivity:
        if np.isclose(d, 1.5):
            bondtype = Chem.BondType.AROMATIC
        else:
            bondtype = BONDTYPES.get(int(d), Chem.BondType.UNSPECIFIED)
        rwmol.AddBond(i, j, bondtype)
    Chem.SanitizeMol(rwmol)
    add_conformer_from_coordinates(rwmol, qcmol.geometry)
    return rwmol


def add_conformer_from_coordinates(rdmol: "rdkit.Chem.Mol",
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


def generate_conformers(rdmol: "rdkit.Chem.Mol",
                        n_conformers: int = 0,
                        rms_tolerance: float = 1.5):
    """Generate conformers for an RDKit molecule.

    This does not clear existing conformers.

    Parameters
    ----------
    rdmol: rdkit.Chem.Mol
        RDKit Molecule
    n_conformers: int (optional)
        Maximum number of conformers to generate
    rms_tolerance: float (optional)
        RMSD threshold used to prune new conformers
    """
    AllChem.EmbedMultipleConfs(rdmol, numConfs=n_conformers,
                               pruneRmsThresh=rms_tolerance,
                               useRandomCoords=True,
                               clearConfs=False,
                               ignoreSmoothingFailures=True)


def minimize_conformer_geometries(rdmol: "rdkit.Chem.Mol",
                                  minimize_max_iter: int = 2000):
    """Minimize conformer geometries of an RDKit molecule

    Parameters
    ----------
    minimize_max_iter: int (optional)
        Maximum number of iterations for minimization
    """

    # TODO: is UFF good?
    AllChem.UFFOptimizeMoleculeConfs(rdmol, numThreads=0,
                                     maxIters=minimize_max_iter)


def compute_mmff_charges(rdmol: "rdkit.Chem.Mol",
                         forcefield: Literal["MMFF94", "MMFF94s"] = "MMFF94",
                         normalize_partial_charges: bool = True):
    mps = AllChem.MMFFGetMoleculeProperties(rdmol, forcefield)
    n_atoms = rdmol.GetNumAtoms()
    charges = np.array([mps.GetMMFFPartialCharge(i) for i in range(n_atoms)])

    if normalize_partial_charges:
        total_charge = rdmol.GetFormalCharge()
        partial_charges = charges.sum()
        offset = (total_charge - partial_charges) / n_atoms
        charges += offset
    return charges


def get_exclusions(rdmol: "rdkit.Chem.Mol") -> Set[Tuple[int, int]]:
    exclusions = set()
    for i, atom in enumerate(rdmol.GetAtoms()):
        partners = [b.GetOtherAtomIdx(i) for b in atom.GetBonds()]
        exclusions |= set((i, x) for x in partners)
        exclusions |= set(itertools.combinations(sorted(partners), 2))
    return exclusions


def compute_distance_matrix(coordinates):
    dist_sq = np.einsum('ijk,ilk->ijl', coordinates, coordinates)
    diag = np.einsum("ijj->ij", dist_sq)
    a, b = diag.shape
    dist_sq += dist_sq - diag.reshape((a, 1, b)) - diag.reshape((a, b, 1))
    diag[:] = -0.0
    return np.sqrt(-dist_sq)


def compute_electrostatic_energy(rdmol: "rdkit.Chem.Mol",
                                 forcefield: Literal["MMFF94", "MMFF94s"] = "MMFF94"):

    conformers = np.array([c.GetPositions() for c in rdmol.GetConformers()])
    distances = compute_distance_matrix(conformers)
    inverse_distances = np.reciprocal(distances,
                                      out=np.zeros_like(distances),
                                      where=~np.isclose(distances, 0))

    charges = compute_mmff_charges(rdmol, forcefield)
    charge_products = charges @ charges.T

    excl_i, excl_j = zip(*get_exclusions(rdmol))
    charge_products[excl_i, excl_j] = charge_products[excl_j, excl_i] = 0.0

    energies = inverse_distances * charge_products[np.newaxis, ...]
    return 0.5 * energies.sum(axis=(1, 2))


def compute_heavy_rms(rdmol: "rdkit.Chem.Mol",
                      conformer_ids=None):
    if conformer_ids is None:
        conformer_ids = [c.GetId() for c in rdmol.GetConformers()]

    rdmol = Chem.RemoveHs(rdmol)
    n_conformers = len(conformer_ids)

    rms = np.zeros((n_conformers, n_conformers))
    for i, j in itertools.combinations(conformer_ids, 2):
        rms[i, j] = rms[j, i] = AllChem.GetBestRMS(rdmol, rdmol, i, j)
    return rms


def select_elf_conformer_ids(rdmol: "rdkit.Chem.Mol",
                             energy_window: float = 15,
                             limit: int = 10,
                             rms_tolerance: float = 0.05,
                             ):

    n_conformers = rdmol.GetNumConformers()
    if n_conformers == 0:
        return

    energies = compute_electrostatic_energy(rdmol, forcefield="MMFF94")
    all_conformer_ids = np.array([c.GetId() for c in rdmol.GetConformers()])

    sorting = np.argsort(energies)
    upper_energy = energy_window + energies[sorting[0]]
    cutoff = np.searchsorted(energies[sorting], upper_energy)
    conformer_ids = all_conformer_ids[sorting[:cutoff]]
    rms_matrix = compute_heavy_rms(rdmol, conformer_ids)

    n_max_output = min(limit, rms_matrix.shape[0])
    selected_indices = [0]
    for i in range(n_max_output - 1):
        selected_rms = rms_matrix[selected_indices]
        any_too_close = np.any(selected_rms < rms_tolerance, axis=0)
        if np.all(any_too_close):
            break

        rmsdist = np.where(any_too_close, -np.inf, selected_rms.sum(axis=0))
        selected_indices.append(rmsdist.argmax())

    return conformer_ids[selected_indices]


def select_elf_conformer_coordinates(rdmol: "rdkit.Chem.Mol",
                                     energy_window: float = 15,
                                     limit: int = 10,
                                     rms_tolerance: float = 0.05,):
    ids = select_elf_conformer_ids(rdmol, energy_window=energy_window,
                                   limit=limit, rms_tolerance=rms_tolerance)
    return np.array([rdmol.GetConformer(i).GetPositions() for i in ids])


def generate_diverse_conformer_coordinates(molecule,
                                           n_conformer_pool: int = 4000,
                                           energy_window: float = 15,
                                           n_max_conformers: int = 10,
                                           rms_tolerance: float = 0.05,
                                           ):
    if not isinstance(molecule, Chem.Mol):
        molecule = rdmol_from_qcelemental(molecule)
    else:
        molecule = Chem.RWMol(molecule)

    generate_conformers(molecule, n_conformers=n_conformer_pool,
                        rms_tolerance=rms_tolerance)
    return select_elf_conformer_coordinates(molecule,
                                            energy_window=energy_window,
                                            limit=n_max_conformers,
                                            rms_tolerance=rms_tolerance)
