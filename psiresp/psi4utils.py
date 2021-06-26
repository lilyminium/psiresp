import re

import psi4

def get_mol_spec(molecule: psi4.core.Molecule) -> str:
    """Create Psi4 molecule specification from Psi4 molecule

    Parameters
    ----------
    molecule: Psi4Mol

    Returns
    -------
    mol_spec: str
    """
    mol = molecule.create_psi4_string_from_molecule()
    # remove any clashing fragment charge/multiplicity
    pattern = r"--\n\s*\d \d\n"
    mol = re.sub(pattern, "", mol)
    return f"molecule {molecule.name()} {{\n{mol}\n}}\n\n"


