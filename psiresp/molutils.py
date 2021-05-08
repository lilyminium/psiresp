"""
Molecule utilities for use with MDAnalysis, RDKit, and Psi4 objects.
"""

import io
import glob
import tempfile
import itertools

import rdkit
import MDAnalysis as mda
from rdkit.Chem import AllChem, rdFMCS
from rdkit import Chem
import psi4


def load_rdmol(filename: str):
    """Load an RDKit molecule from file.

    Parameters
    ----------
    filename : str
        Filename or string containing file contents or SMILES/SMARTS.

    Returns
    -------
    :class:`rdkit.Chem.Mol`

    Raises
    ------
    ValueError
        If file cannot be parsed
    """
    suffix = filename.split(".")[-1]
    FILE = {
        "pdb": Chem.MolFromPDBFile,
        "tpl": Chem.MolFromTPLFile,
        "mol2": Chem.MolFromMol2File,
        "mol": Chem.MolFromMolFile,
        "png": Chem.MolFromPNGFile,
    }
    STR = (
        Chem.MolFromSmiles,
        Chem.MolFromSmarts,
        Chem.MolFromFASTA,
        Chem.MolFromHELM,
        Chem.MolFromSequence,
        Chem.MolFromMol2Block,
        Chem.MolFromMolBlock,
        Chem.MolFromPDBBlock,
        Chem.MolFromPNGString,
        Chem.MolFromRDKitSVG,
        Chem.MolFromTPLBlock,
    )
    if suffix in FILE:
        mol = FILE[suffix](filename, sanitize=False)
    else:
        for parser in STR:
            try:
                mol = parser(filename, sanitize=True)
            except RuntimeError:
                pass
            except Exception as e:
                if not str(e).startswith("Python argument types in"):
                    raise e from None
            else:
                if mol is not None:
                    break
        else:
            raise ValueError(f"Could not parse {filename}")
    mol = Chem.AddHs(mol, addCoords=True)
    return mol


def rdmol_from_file(filename: str, name: str = ""):
    """Create an RDKit molecule from file or string.
    Loads into MDAnalysis and converts to an RDKit molecule
    if an existing RDKit function does not exist. Accepts
    a filename template that uses the molecule name.

    Hydrogens are added if they are implicit.

    Parameters
    ----------
    filename : str
        Filename or string containing file contents or SMILES/SMARTS.
        Accepts templates with the name keyword, e.g. "mol_{name}.pdb"
    name: str (optional)
        Gets substituted into the filename template.

    Returns
    -------
    :class:`rdkit.Chem.Mol`

    Raises
    ------
    ValueError
        If file cannot be parsed

    Example
    -------

    .. code-block::python

        get_rdmol("mol_{name}.pdb", name="alanine")

    """
    try:
        rdfile = glob.glob(filename.format(name=name))[0]
    except IndexError:
        rdfile = filename

    try:
        return load_rdmol(rdfile)
    except ValueError:
        pass

    mol = mda.Universe(rdfile)
    if not hasattr(mol.atoms, "elements"):
        elements = mda.topology.guessers.guess_types(mol.atoms.names)
        mol.add_TopologyAttr("elements", elements)
    mol = mol.atoms.convert_to("RDKIT")

    mol = Chem.AddHs(mol, addCoords=True)
    return mol


def psi4mol_from_file(filename: str):
    """Read Psi4 molecule from file

    Parameters
    ----------
    filename: str
        Filename

    Returns
    -------
    :class:`psi4.core.Molecule`
    """
    u = mda.Universe(filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        file = f"{tmpdir}/temp.xyz"
        u.atoms.write(file)
        with open(file, "r") as f:
            geom = f.read()
    mol = psi4.core.Molecule.from_string(geom, fix_com=True, fix_orientation=True)
    mol.update_geometry()
    return mol


def psi4mol_to_string(psi4mol: psi4.core.Molecule):
    """
    Create Psi4 string representaiton of molecule from Psi4 molecule

    Parameters
    ----------
    psi4mol: psi4.core.Molecule

    Returns
    -------
    str
    """
    mol = psi4mol.create_psi4_string_from_molecule()
    return f"molecule {psi4mol.name()} {{\n{mol}\n}}\n\n"


def rdmol_to_psi4mols(rdmol: rdkit.Chem.Mol, name: str = None):
    """Convert RDKit molecule to one or more Psi4 molecules,
    one for each conformer.

    Parameters
    ----------
    rdmol: rdkit.Chem.Mol
        RDKit molecule with at least one conformer
    name: str (optional)
        Molecule name

    Returns
    -------
    list of :class:`psi4.core.Molecule`
    """
    confs = rdmol.GetConformers()
    n_atoms = rdmol.GetNumAtoms()
    atoms = [rdmol.GetAtomWithIdx(i) for i in range(n_atoms)]
    symbols = [a.GetSymbol() for a in atoms]
    ATOM = "{sym} {x[0]} {x[1]} {x[2]}"

    if name is None:
        name = "Mol"

    mols = []
    for i, c in enumerate(confs, 1):
        pos = c.GetPositions()
        xyz = [ATOM.format(sym=a, x=x) for a, x in zip(symbols, pos)]
        txt = f"{n_atoms}\n0 1 {name}_c{i:03d}\n" + "\n".join(xyz)
        mol = psi4.core.Molecule.from_string(txt, dtype="xyz")
        mol.set_molecular_charge(0)
        mol.set_multiplicity(1)
        mols.append(mol)

    return mols


def log_to_xyz(logfile: str):
    """Convert Psi4 geometry optimization log file to string in XYZ format

    Parameters
    ----------
    logfile: str
        File name

    Returns
    -------
    str
        XYZ formatted molecule string
    """
    with open(logfile, "r") as f:
        contents = f.read()
    last_lines = contents.split("OPTKING Finished Execution")[-1].split("\n")
    symbols = []
    xs = []
    ys = []
    zs = []
    for line in last_lines:
        line = line.strip().split()
        if len(line) == 4:
            try:
                x = float(line[1])
                y = float(line[2])
                z = float(line[3])
            except ValueError:
                continue
            else:
                symbols.append(line[0])
                xs.append(x)
                ys.append(y)
                zs.append(z)

    ATOM = "{sym} {x} {y} {z}"
    name = logfile.strip(".log")
    lines = [len(symbols), name]
    for sym, x, y, z in zip(symbols, xs, ys, zs):
        lines.append(ATOM.format(sym=sym, x=x, y=y, z=z))
    txt = "\n".join(lines)
    return txt


def log_to_psi4mol(logfile: str):
    """Create a Psi4 molecule from a Psi4 geometry optimization log file

    Parameters
    ----------
    logfile: str
        File name

    Returns
    -------
    psi4.core.Molecule
    """
    txt = log_to_xyz(logfile)
    mol = psi4.core.Molecule.from_string(txt, dtype="xyz")
    return mol


def psi4mol_to_rdmol(psi4mol: psi4.core.Molecule):
    """
    Create RDKit molecule from Psi4 molecule

    Parameters
    ----------
    psi4mol: psi4.core.Molecule

    Returns
    -------
    rdkit.Chem.Mol
    """
    txt = psi4mol.format_molecule_for_mol()
    return Chem.MolFromMol2Block(txt)


def xyz_to_psi4mol(txt: str):
    """Create Psi4 molecule from XYZ formatted string

    Parameters
    ----------
    txt: str
        XYZ formatted text

    Returns
    -------
    psi4.core.Molecule
    """
    return psi4.core.Molecule.from_string(txt, dtype="xyz")


def psi4mol_to_xyz(psi4mol: psi4.core.Molecule):
    """Create XYZ formatted string from Psi4 molecule

    Parameters
    ----------
    psi4mol: psi4.core.Molecule

    Returns
    -------
    str
        XYZ-formatted text
    """
    return psi4mol.to_string(dtype="xyz")


def rdmols_to_inter_chrequiv(rdmols: list, n_atoms: int = 4):
    """Create intermolecular charge equivalence constraints
    from RDKit molecules

    Parameters
    ----------
    rdmols: list of rdkit.Chem.Mol
        List of RDKit molecules
    n_atoms: int (optional)
        Number of atoms to include in common substructure matching

    Returns
    -------
    list of charge equivalence constraints
    """
    matches = set()
    for pair in itertools.combinations(rdmols, 2):
        res = rdFMCS.FindMCS(
            pair,
            # TODO: AtomCompare.CompareIsotopes?
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=1,
        )
        if not res.canceled and res.numAtoms >= n_atoms:
            matches.add(res.smartsString)

    submols = [Chem.MolFromSmarts(x) for x in matches]
    chrequiv = []
    for ref in submols:
        sub = []
        for n in range(ref.GetNumAtoms()):
            sub.append(set())
        for j, mol in enumerate(rdmols):
            subs = mol.GetSubstructMatches(ref)
            for k, atoms in enumerate(subs):
                sub[k] |= set([(j, x) for x in atoms])
        for cmp in sub:
            for k, group in enumerate(chrequiv):
                if len(group and cmp):
                    group.update(cmp)
                    break
            else:
                chrequiv.append(cmp)

    res = [[list(y) for y in x] for x in chrequiv]
    return res
