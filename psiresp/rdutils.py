
import itertools
import logging
from typing import TYPE_CHECKING, Set, Tuple

from typing_extensions import Literal
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    raise ImportError("rdkit is a core dependency of PsiRESP. "
                      "Please install it with "
                      "`conda install -c conda-forge rdkit`")


if TYPE_CHECKING:
    import rdkit
    import qcelemental

BONDTYPES = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

logger = logging.getLogger(__name__)

OFF_SMILES_ATTRIBUTE = "canonical_isomeric_explicit_hydrogen_mapped_smiles"


def rdmol_from_smiles(smiles, order_by_map_number: bool = False):
    smiles_parser = Chem.rdmolfiles.SmilesParserParams()
    smiles_parser.removeHs = False
    rdmol = Chem.AddHs(Chem.MolFromSmiles(smiles, smiles_parser))
    if order_by_map_number:
        map_numbers = [atom.GetAtomMapNum() for atom in rdmol.GetAtoms()]
        if 0 not in map_numbers:
            new_order = tuple(map(int, np.argsort(map_numbers)))
            rdmol = Chem.RenumberAtoms(rdmol, new_order)
            for atom in rdmol.GetAtoms():
                atom.SetAtomMapNum(0)
    return rdmol


def rdmol_to_smiles(rdmol, mapped=True):
    rdmol = Chem.Mol(rdmol)
    if not any(at.GetAtomMapNum() for at in rdmol.GetAtoms()) and mapped:
        for i, at in enumerate(rdmol.GetAtoms(), 1):
            at.SetAtomMapNum(i)

    return Chem.MolToSmiles(rdmol, allBondsExplicit=True, allHsExplicit=True)


def assign_connectivity_from_proximity(rdmol, scale_factor=0.55):
    table = Chem.GetPeriodicTable()
    conformer = rdmol.GetConformer(0)
    n_atoms = rdmol.GetNumAtoms()
    for i, atom1 in enumerate(rdmol.GetAtoms()):
        xyz1 = np.array(conformer.GetAtomPosition(i))
        rad1 = table.GetRvdw(atom1.GetSymbol())
        for j in range(i + 1, n_atoms):
            atom2 = rdmol.GetAtomWithIdx(j)
            xyz2 = np.array(conformer.GetAtomPosition(j))
            rad2 = table.GetRvdw(atom2.GetSymbol())

            max_distance = (rad1 + rad2) * scale_factor
            distance = np.linalg.norm(xyz1 - xyz2)
            if distance < max_distance:
                rdmol.AddBond(i, j, Chem.BondType.SINGLE)


def n_unpaired_e(atom):
    table = Chem.GetPeriodicTable()
    current = atom.GetTotalValence() - atom.GetFormalCharge()
    valences = list(table.GetValenceList(atom.GetSymbol()))
    return [valence - current for valence in valences]


def assign_bond_order_and_charges_from_electrons(rdmol):
    """
    Modelled after Cedric Bouysset's method for assigning bond order and charges
    as detailed in https://cedric.bouysset.net/blog/2020/07/22/rdkit-converter-part2 .

    In short:
        - if atom has no unpaired electrons, no action
        - if atom has too many paired electrons, turn them into positive charge
        - if atom has unpaired electrons:
             - search for neighbors with unpaired electrons
             - set bond between them to smallest one that satisfies the lowest
               number of electrons in any state in either atom
             - repeat for all neighbors
             - if atom has remaining unpaired electrons, turn them into negative charge
    """

    # sort by # unpaired electrons to deal with unpaired electrons first
    for atom in sorted(rdmol.GetAtoms(), reverse=True, key=n_unpaired_e):
        n_unpaired = n_unpaired_e(atom)
        if len(n_unpaired) == 1:  # single valence state
            n = n_unpaired[0]
            # set excess electrons to positive charge
            if n < 0:
                atom.SetFormalCharge(-n)
                rdmol.UpdatePropertyCache(strict=False)
            if n <= 0:
                continue

        # deal with unpaired electrons
        n_unpaired = np.array(n_unpaired_e(atom))
        # compare with unpaired electrons of neighboring atoms
        # again, sort for the likeliest pairs first
        for neighbor in sorted(
            atom.GetNeighbors(),
            reverse=True,
            key=n_unpaired_e,
        ):
            neighbor_unpaired = np.array(n_unpaired_e(neighbor))
            # set bond between neighbor and self to smallest possible
            # to satisfy valences
            smallest_overlap = min(
                min(n_unpaired[n_unpaired >= 0], default=0),
                min(neighbor_unpaired[neighbor_unpaired >= 0], default=0),
            )
            if smallest_overlap == 0:
                continue
            bond = rdmol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            bond.SetBondType(BONDTYPES[smallest_overlap + 1])
            # update valence states
            rdmol.UpdatePropertyCache(strict=False)
            n_unpaired = np.array(n_unpaired_e(atom))

        # any residual unpaired electrons are turned into negative charge
        n_unpaired = n_unpaired_e(atom)[0]
        if n_unpaired > 0:
            atom.SetFormalCharge(-n)
            atom.SetNumRadicalElectrons(0)
            rdmol.UpdatePropertyCache(strict=False)
    Chem.Kekulize(rdmol)


def rdmol_from_qcelemental(qcmol: "qcelemental.models.Molecule",
                           guess_connectivity: bool = True):
    import qcelemental as qcel
    smiles = qcmol.dict().get("extras", {}).get(OFF_SMILES_ATTRIBUTE)
    if smiles:
        rdmol = rdmol_from_smiles(smiles, order_by_map_number=True)
        coordinates = qcmol.geometry * qcel.constants.conversion_factor("bohr", "angstrom")
        add_conformer_from_coordinates(rdmol, coordinates=coordinates)
    else:
        rdmol = Chem.RWMol(_rdmol_from_qcelemental(qcmol))
    if guess_connectivity:
        if not qcmol.connectivity:
            assign_connectivity_from_proximity(rdmol)
        rdmol.UpdatePropertyCache(strict=False)
        assign_bond_order_and_charges_from_electrons(rdmol)

    rdmol = Chem.Mol(rdmol)
    Chem.SanitizeMol(rdmol)
    return rdmol


def rdmol_from_psi4(psi4mol):
    molstr = psi4mol.format_molecule_for_mol()
    rdmol = Chem.MolFromMolBlock(molstr, removeHs=False, sanitize=False)
    return rdmol


def get_rdkit_connectivity(rdmol):
    return np.array([
        [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble()]
        for bond in rdmol.GetBonds()
    ])


def rdmol_to_qcelemental(rdmol, multiplicity=1, random_seed=-1):
    import qcelemental as qcel
    rdmol = Chem.AddHs(rdmol)
    if not rdmol.GetNumConformers():
        Chem.rdDistGeom.EmbedMolecule(rdmol, useRandomCoords=True,
                                      randomSeed=random_seed)

    connectivity = get_rdkit_connectivity(rdmol)

    for i, atom in enumerate(rdmol.GetAtoms(), 1):
        atom.SetAtomMapNum(i)
    extras = {}
    extras[OFF_SMILES_ATTRIBUTE] = Chem.MolToSmiles(rdmol,
                                                    allBondsExplicit=True,
                                                    allHsExplicit=True,
                                                    )
    geometry = np.array(rdmol.GetConformer(0).GetPositions())
    schema = dict(symbols=[a.GetSymbol() for a in rdmol.GetAtoms()],
                  geometry=geometry * qcel.constants.conversion_factor("angstrom", "bohr"),
                  connectivity=connectivity if len(connectivity) else None,
                  molecular_charge=Chem.GetFormalCharge(rdmol),
                  molecular_multiplicity=multiplicity,
                  extras=extras
                  )
    qcmol = qcel.models.Molecule.from_data(schema)
    return qcmol


def _rdmol_from_qcelemental(qcmol: "qcelemental.models.Molecule"):
    import qcelemental as qcel
    rwmol = Chem.RWMol()
    for symbol in qcmol.symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        rwmol.AddAtom(atom)
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
                        rms_tolerance: float = 1.5,
                        random_seed=-1):
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
    Chem.SanitizeMol(rdmol)
    AllChem.EmbedMultipleConfs(rdmol, numConfs=n_conformers,
                               pruneRmsThresh=rms_tolerance,
                               useRandomCoords=True,
                               clearConfs=False,
                               randomSeed=random_seed,
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
    inverse_distances = np.reciprocal(distances,
                                      out=np.zeros_like(distances),
                                      where=~np.isclose(distances, 0))

    charges = np.abs(compute_mmff_charges(rdmol)).reshape(-1, 1)
    charge_products = charges @ charges.T

    excl_i, excl_j = zip(*get_exclusions(rdmol))
    charge_products[(excl_i, excl_j)] = 0.0
    charge_products[(excl_j, excl_i)] = 0.0

    energies = inverse_distances * charge_products
    return 0.5 * energies.sum(axis=(1, 2))


def compute_heavy_rms(rdmol: "rdkit.Chem.Mol",
                      conformer_ids=None):
    if conformer_ids is None:
        conformer_ids = [c.GetId() for c in rdmol.GetConformers()]

    rdmol = Chem.RemoveHs(rdmol)
    n_conformers = len(conformer_ids)

    rms = np.zeros((n_conformers, n_conformers))
    for i, j in itertools.combinations(np.arange(n_conformers), 2):
        rms[i, j] = rms[j, i] = AllChem.GetBestRMS(rdmol, rdmol, conformer_ids[i], conformer_ids[j])
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
    conversion = qcel.constants.conversion_factor(
        "(4 * pi * electric_constant) * kcal",
        "e * e / angstrom",
    )
    window = energy_window * conversion / qcel.constants.get("avogadro constant")
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


def get_sp3_ch_indices(rdmol):
    symbols = np.array([at.GetSymbol() for at in rdmol.GetAtoms()])
    bonds = get_rdkit_connectivity(rdmol)
    single_bonds = np.isclose(bonds[:, 2], np.ones_like(bonds[:, 2]))

    groups = {}
    for i in np.where(symbols == "C")[0]:
        contains_index = np.any(bonds[:, :2] == i, axis=1)
        c_bonds = bonds[contains_index & single_bonds][:, :2]
        c_partners = (c_bonds[c_bonds != i]).astype(int)
        if len(c_partners) == 4:
            groups[i] = c_partners[symbols[c_partners] == "H"]
    return groups


def molecule_to_rdkit(molecule):
    if molecule._rdmol is None:
        rdmol = rdmol_from_qcelemental(molecule.qcmol,
                                       guess_connectivity=True)
    else:
        rdmol = Chem.Mol(molecule._rdmol)

    for conformer in molecule.conformers:
        add_conformer_from_coordinates(rdmol, conformer.coordinates)

    charges = molecule.charges
    if charges is not None:
        for charge, atom in zip(charges, rdmol.GetAtoms()):
            atom.SetDoubleProp("PartialCharge", charge)
        Chem.CreateAtomDoublePropertyList(rdmol, "PartialCharge")
    rdmol.UpdatePropertyCache(strict=False)
    return Chem.Mol(rdmol)


def molecule_from_rdkit(rdmol, molecule_cls, random_seed=-1, **kwargs):
    rdmol = Chem.AddHs(rdmol)
    qcmol = rdmol_to_qcelemental(rdmol, random_seed=random_seed)
    kwargs = dict(**kwargs)
    kwargs["qcmol"] = qcmol
    obj = molecule_cls(**kwargs)
    obj._rdmol = rdmol

    for conformer in rdmol.GetConformers():
        obj.add_conformer_with_coordinates(np.array(conformer.GetPositions()))
    return obj


def get_connectivity(molecule):
    return get_rdkit_connectivity(molecule._rdmol)
