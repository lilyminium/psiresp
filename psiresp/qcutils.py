import numpy as np
import qcelemental as qcel


def qcmol_from_rdkit(rdmol, confId=-1):
    symbols = [a.GetSymbol() for a in rdmol.GetAtoms()]
    if rdmol.GetNumConformers() == 0:
        validate = False
        geometry = np.zeros((len(symbols), 3))
    else:
        validate = None
        geometry = np.array(rdmol.GetConformer(confId).GetPositions())

    connectivity = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble())
        for b in rdmol.GetBonds()
    ]

    qcmol = qcel.models.Molecule(symbols=symbols,
                                 geometry=geometry,
                                 validate=validate,
                                 connectivity=connectivity,
                                 molecular_charge=Chem.GetFormalCharge(rdmol),
                                 )
    return qcmol


def reconstruct_wavefunction(qcrecord):
    from qcelemental.models.results import WavefunctionProperties

    WFN_PROPS = ["scf_eigenvalues_a", "scf_orbitals_a", "basis", "restricted"]
    qcwfn = WavefunctionProperties(**qcrecord.get_wavefunction(WFN_PROPS),
                                   **qcrecord.wavefunction["return_map"])
    return qcwfn


def get_density_ordering(wavefunction):
    # Re-order the density matrix to match the ordering expected by psi4.
    angular_momenta = {
        angular_momentum
        for atom in wavefunction.basis.atom_map
        for shell in wavefunction.basis.center_data[atom].electron_shells
        for angular_momentum in shell.angular_momentum
    }

    spherical_maps = {
        L: np.array(
            list(range(L * 2 - 1, 0, -2)) + [0] + list(range(2, L * 2 + 1, 2))
        )
        for L in angular_momenta
    }

    # Build a flat index that we can transform the AO quantities
    ao_map = []
    counter = 0

    for atom in wavefunction.basis.atom_map:

        center = wavefunction.basis.center_data[atom]
        for shell in center.electron_shells:

            if shell.harmonic_type == "cartesian":
                ao_map.append(np.arange(counter, counter + shell.nfunctions()))

            else:
                smap = spherical_maps[shell.angular_momentum[0]]
                ao_map.append(smap + counter)

            counter += shell.nfunctions()

    ao_map = np.hstack(ao_map)

    reverse_ao_map = {map_index: i for i, map_index in enumerate(ao_map)}
    reverse_ao_map = np.array([reverse_ao_map[i] for i in range(len(ao_map))])
    return reverse_ao_map


def reconstruct_density(qcrecord):
    qcwfn = reconstruct_wavefunction(qcrecord)
    reverse_ao_map = get_density_ordering(qcwfn)

    n_alpha = qcrecord.properties.calcinfo_nalpha
    orbitals = getattr(qcwfn, qcwfn.orbitals_a)[:, :n_alpha]
    density = np.dot(orbitals, orbitals.T)
    return density[reverse_ao_map[:, None], reverse_ao_map]


def qcmol_with_coordinates(qcmol, coordinates):
    dct = qcmol.dict()
    dct["geometry"] = coordinates
    return qcel.models.Molecule(**dct)
