from typing import List, Dict

import psi4
import numpy as np
import qcelemental as qcel

from . import qcutils


psi4.core.be_quiet()


def psi4mol_from_qcmol(qcmol):
    return psi4.geometry(qcmol.to_string("psi4", "angstrom"))


def construct_psi4_wavefunction(qc_wavefunction):
    psi4mol = psi4mol_from_qcmol(qc_wavefunction.qcmol)
    psi4mol.reset_point_group("c1")

    qcdensity = qc_wavefunction.reconstruct_density()
    density = psi4.core.Matrix.from_array(qcdensity)
    psi4wfn = psi4.core.RHF(
        psi4.core.Wavefunction.build(psi4mol, qc_wavefunction.basis),
        psi4.core.SuperFunctional(),
    )
    psi4wfn.Da().copy(density)
    return psi4wfn


def compute_esp(qc_wavefunction, grid):
    psi4wfn = construct_psi4_wavefunction(qc_wavefunction)
    esp_calc = psi4.core.ESPPropCalc(psi4wfn)

    ANGSTROM_TO_BOHR = qcel.constants.conversion_factor("angstrom", "bohr")
    psi4grid = psi4.core.Matrix.from_array(grid)# * ANGSTROM_TO_BOHR)
    psi4esp = esp_calc.compute_esp_over_grid_in_memory(psi4grid)

    return np.array(psi4esp)


def get_connectivity(qcmol) -> List[List[int]]:
    psi4mol = psi4mol_from_qcmol(qcmol)
    return np.asarray(psi4.qcdb.parker._bond_profile(psi4mol))


def get_sp3_ch_indices(qcmol) -> Dict[int, List[int]]:
    symbols = np.asarray(qcmol.symbols)

    bonds = get_connectivity(qcmol)
    single_bonds = bonds[:, 2] == 1

    groups = {}
    for i in np.where(symbols == "C")[0]:
        contains_index = np.any(bonds[:, :2] == i, axis=1)
        c_bonds = bonds[contains_index & single_bonds][:, :2]
        c_partners = c_bonds[c_bonds != i]
        if len(c_partners) == 4:
            groups[i] = c_partners[symbols[c_partners] == "H"]
    return groups