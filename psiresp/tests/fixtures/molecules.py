import pytest

from psiresp.molecule import Molecule


@pytest.fixture
def dmso(dmso_qcmol):
    return Molecule(qcmol=dmso_qcmol, optimize_geometry=False,
                    keep_original_orientation=True)


@pytest.fixture
def methylammonium(methylammonium_qcmol):
    molecule = Molecule(qcmol=methylammonium_qcmol,
                        reorientations=[(0, 4, 6), (6, 4, 0)],
                        keep_original_orientation=False,
                        multiplicity=1,
                        charge=1)
    molecule.generate_conformers()
    molecule.generate_orientations()
    return molecule


@pytest.fixture
def nme2ala2(nme2ala2_c1_opt_qcmol, nme2ala2_c2_opt_qcmol):
    reorientations = [(4, 17, 18), (18, 17, 4),
                      (5, 18, 19), (19, 18, 5)]
    molecule = Molecule(qcmol=nme2ala2_c1_opt_qcmol,
                        multiplicity=1,
                        reorientations=reorientations,
                        keep_original_orientation=False)
    # molecule.generate_conformers()
    molecule.add_conformer_with_coordinates(nme2ala2_c1_opt_qcmol.geometry, units="bohr")
    molecule.add_conformer_with_coordinates(nme2ala2_c2_opt_qcmol.geometry, units="bohr")
    molecule.generate_orientations()
    return molecule


@pytest.fixture
def methylammonium_empty(methylammonium_qcmol):
    molecule = Molecule(qcmol=methylammonium_qcmol.copy(update={"connectivity": None}),
                        reorientations=[(0, 4, 6), (6, 4, 0)],
                        keep_original_orientation=False,
                        multiplicity=1,
                        charge=1)
    molecule.generate_conformers()
    molecule.generate_orientations()
    return molecule


@pytest.fixture
def nme2ala2_empty(nme2ala2_c1_opt_qcmol, nme2ala2_c2_opt_qcmol):
    reorientations = [(4, 17, 18), (18, 17, 4),
                      (5, 18, 19), (19, 18, 5)]
    molecule = Molecule(qcmol=nme2ala2_c1_opt_qcmol.copy(update={"connectivity": None}),
                        multiplicity=1,
                        reorientations=reorientations,
                        keep_original_orientation=False)
    # molecule.generate_conformers()
    molecule.add_conformer_with_coordinates(nme2ala2_c1_opt_qcmol.geometry, units="bohr")
    molecule.add_conformer_with_coordinates(nme2ala2_c2_opt_qcmol.geometry, units="bohr")
    molecule.generate_orientations()
    return molecule
