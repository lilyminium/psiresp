import numpy as np
import qcelemental as qcel

from .moleculebase import BaseMolecule


class QCWaveFunction(BaseMolecule):
    qc_wavefunction: qcel.models.results.WavefunctionProperties
    n_alpha: int
    energy: float
    basis: str

    @classmethod
    def from_atomicresult(cls, result):
        return cls(qc_wavefunction=result.wavefunction,
                   qcmol=result.molecule,
                   n_alpha=result.properties.calcinfo_nalpha,
                   basis=result.model.basis,
                   energy=result.properties.return_energy)

    @classmethod
    def from_qcrecord(cls, qcrecord):
        WFN_PROPS = ["scf_eigenvalues_a", "scf_orbitals_a", "basis", "restricted"]
        dct = qcrecord.get_wavefunction(WFN_PROPS)
        dct.update(qcrecord.wavefunction["return_map"])
        qcwfn = qcel.models.results.WavefunctionProperties(**dct)

        return cls(qc_wavefunction=qcwfn, qcmol=qcrecord.get_molecule(),
                   n_alpha=qcrecord.properties.calcinfo_nalpha,
                   basis=qcrecord.basis,
                   energy=qcrecord.properties.return_energy)

    def get_density_ordering(self):
        # Re-order the density matrix to match the ordering expected by psi4.
        angular_momenta = {
            angular_momentum
            for atom in self.qc_wavefunction.basis.atom_map
            for shell in self.qc_wavefunction.basis.center_data[atom].electron_shells
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

        for atom in self.qc_wavefunction.basis.atom_map:

            center = self.qc_wavefunction.basis.center_data[atom]
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

    def reconstruct_density(self):
        reverse_ao_map = self.get_density_ordering()
        orbitals = getattr(self.qc_wavefunction, self.qc_wavefunction.orbitals_a)[:, :self.n_alpha]
        density = np.dot(orbitals, orbitals.T)
        return density[reverse_ao_map[:, None], reverse_ao_map]


def get_vdwradii(element):
    try:
        radius = qcel.vdwradii.get(element, units="bohr")
    except qcel.DataUnavailableError:
        raise ValueError(
            f"Cannot get VdW radius for element {element}, "
            "so cannot guess bonds"
        )
    return radius
