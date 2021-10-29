
import itertools
import logging
from typing import TYPE_CHECKING, Set, Tuple

from typing_extensions import Literal
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

logger = logging.getLogger(__name__)


def rdmol_from_qcelemental(qcmol: "qcelemental.models.Molecule",
                           guess_connectivity: bool = True):

    if guess_connectivity:
        import psi4
        psi4str = qcmol.to_string(dtype="psi4")
        psi4mol = psi4.core.Molecule.from_string(psi4str, dtype="psi4", fix_com=True, fix_orientation=True)
        rdmol = rdmol_from_psi4(psi4mol)
    else:
        rdmol = _rdmol_from_qcelemental(qcmol)
    Chem.SanitizeMol(rdmol)
    return rdmol


def rdmol_from_psi4(psi4mol):
    molstr = psi4mol.format_molecule_for_mol()
    rdmol = Chem.MolFromMolBlock(molstr, removeHs=False, sanitize=True)
    return rdmol


def _rdmol_from_qcelemental(qcmol: "qcelemental.models.Molecule"):
    import qcelemental as qcel
    rwmol = Chem.RWMol()
    for symbol in qcmol.symbols:
        rwmol.AddAtom(Chem.Atom(symbol))
    if qcmol.connectivity is not None:
        for i, j, d in qcmol.connectivity:
            if np.isclose(d, 1.5):
                bondtype = Chem.BondType.AROMATIC
            else:
                bondtype = BONDTYPES.get(int(d), Chem.BondType.UNSPECIFIED)
            rwmol.AddBond(i, j, bondtype)
    coordinates = qcmol.geometry * qcel.constants.conversion_factor("bohr", "angstrom")
    add_conformer_from_coordinates(rwmol, coordinates=coordinates)
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
                         normalize_partial_charges: bool = True,
                         ) -> np.ndarray:
    mps = AllChem.MMFFGetMoleculeProperties(rdmol, forcefield)
    if mps is None:
        molstr = Chem.MolToSmiles(rdmol)
        raise ValueError(f"MMFF charges could not be computed for {molstr}")
    n_atoms = rdmol.GetNumAtoms()
    charges = np.array([mps.GetMMFFPartialCharge(i) for i in range(n_atoms)])

    if normalize_partial_charges:
        total_charge = Chem.GetFormalCharge(rdmol)
        partial_charges = charges.sum()
        offset = (total_charge - partial_charges) / n_atoms
        charges += offset
    return charges


def get_exclusions(rdmol: "rdkit.Chem.Mol") -> Set[Tuple[int, int]]:
    exclusions = set()
    for i, atom in enumerate(rdmol.GetAtoms()):
        partners = [b.GetOtherAtomIdx(i) for b in atom.GetBonds()]
        exclusions |= set(tuple(sorted([i, x])) for x in partners)
        exclusions |= set(itertools.combinations(sorted(partners), 2))
    return exclusions


def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    dist_sq = np.einsum('ijk,ilk->ijl', coordinates, coordinates)
    diag = np.einsum("ijj->ij", dist_sq)
    a, b = diag.shape
    dist_sq += dist_sq - diag.reshape((a, 1, b)) - diag.reshape((a, b, 1))
    diag[:] = -0.0
    return np.sqrt(-dist_sq)


def compute_electrostatic_energy(rdmol: "rdkit.Chem.Mol",
                                 forcefield: Literal["MMFF94", "MMFF94s"] = "MMFF94",
                                 ) -> np.ndarray:

    conformers = np.array([c.GetPositions() for c in rdmol.GetConformers()])
    distances = compute_distance_matrix(conformers)
    print(distances)
    inverse_distances = np.reciprocal(distances,
                                      out=np.zeros_like(distances),
                                      where=~np.isclose(distances, 0))

    charges = np.abs(compute_mmff_charges(rdmol)).reshape(-1, 1)
    charge_products = charges @ charges.T
    print(charge_products)

    excl_i, excl_j = zip(*get_exclusions(rdmol))
    charge_products[(excl_i, excl_j)] = 0.0
    charge_products[(excl_j, excl_i)] = 0.0

    energies = inverse_distances * charge_products
    print(energies)
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
                             energy_window: float = 30,
                             limit: int = 10,
                             rms_tolerance: float = 0.05,
                             ):
    import qcelemental as qcel
    n_conformers = rdmol.GetNumConformers()
    if n_conformers == 0:
        return

    energies = compute_electrostatic_energy(rdmol, forcefield="MMFF94")
    all_conformer_ids = [c.GetId() for c in rdmol.GetConformers()]

    sorting = np.argsort(energies)
    window = qcel.constants.conversion_factor(
        "(4 * pi * electric_constant) * kcal",
        "e * e / angstrom",
    )
    window /= qcel.constants.get("avogadro constant")
    upper_energy = window + energies[sorting[0]]
    cutoff = np.searchsorted(energies[sorting], upper_energy)
    conformer_ids = [all_conformer_ids[i] for i in sorting[:cutoff]]
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

    return [conformer_ids[i] for i in selected_indices]


def select_elf_conformer_coordinates(rdmol: "rdkit.Chem.Mol",
                                     energy_window: float = 15,
                                     limit: int = 10,
                                     rms_tolerance: float = 0.05,):
    ids = select_elf_conformer_ids(rdmol, energy_window=energy_window,
                                   limit=limit, rms_tolerance=rms_tolerance)
    return np.array([rdmol.GetConformer(i).GetPositions() for i in ids])


def generate_diverse_conformer_coordinates(molecule,
                                           n_conformer_pool: int = 4000,
                                           energy_window: float = 30,
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
