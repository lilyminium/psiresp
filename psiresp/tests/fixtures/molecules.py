import pytest

from psiresp.molecule import Molecule


@pytest.fixture
def dmso(dmso_qcmol):
    return Molecule(qcmol=dmso_qcmol, optimize_geometry=False)


@pytest.fixture
def methylammonium(methylammonium_qcmol):
    molecule = Molecule(qcmol=methylammonium_qcmol,
                        reorientations=[(0, 4, 6), (6, 4, 0)],
                        keep_original_orientation=False,
                        charge=1)
    molecule.generate_conformers()
    molecule.generate_orientations()
    return molecule


@pytest.fixture
def nme2ala2(nme2ala2_c1_opt_qcmol, nme2ala2_c2_opt_qcmol):
    reorientations = [(4, 17, 18), (18, 17, 4),
                      (5, 18, 19), (19, 18, 5)]
    molecule = Molecule(qcmol=nme2ala2_c1_opt_qcmol,
                        reorientations=reorientations,
                        keep_original_orientation=False)
    # molecule.generate_conformers()
    molecule.add_conformer_with_coordinates(nme2ala2_c1_opt_qcmol.geometry)
    molecule.add_conformer_with_coordinates(nme2ala2_c2_opt_qcmol.geometry)
    molecule.generate_orientations()
    return molecule
