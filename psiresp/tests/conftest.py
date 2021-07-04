import pytest
import psiresp

from .base import coordinates_from_xyzfile, psi4mol_from_xyzfile, data_dir
from .datafiles import (DMSO, DMSO_O1, DMSO_O2, DMSO_O3, DMSO_O4,
                        NME2ALA2_C1, NME2ALA2_OPT_C1, NME2ALA2_OPT_C2,
                        METHYLAMMONIUM_OPT)


@pytest.fixture()
def dmso_coordinates():
    return coordinates_from_xyzfile(DMSO)


@pytest.fixture()
def dmso_psi4mol():
    mol = psi4mol_from_xyzfile(DMSO)
    mol.set_molecular_charge(0)
    mol.set_multiplicity(1)
    mol.update_geometry()
    return mol


@pytest.fixture()
def dmso_o1_coordinates():
    return coordinates_from_xyzfile(DMSO_O1)


@pytest.fixture()
def dmso_o1_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O1)


@pytest.fixture()
def dmso_o2_coordinates():
    return coordinates_from_xyzfile(DMSO_O2)


@pytest.fixture()
def dmso_o2_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O2)


@pytest.fixture()
def dmso_o3_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O3)


@pytest.fixture()
def dmso_o4_psi4mol():
    return psi4mol_from_xyzfile(DMSO_O4)


@pytest.fixture()
def dmso_orientation_psi4mols(dmso_o1_psi4mol, dmso_o2_psi4mol,
                              dmso_o3_psi4mol, dmso_o4_psi4mol):
    return [dmso_o1_psi4mol, dmso_o2_psi4mol,
            dmso_o3_psi4mol, dmso_o4_psi4mol]


@pytest.fixture()
def nme2ala2_c1_psi4mol():
    return psi4mol_from_xyzfile(NME2ALA2_C1)


@pytest.fixture()
def nme2ala2_opt_c1_psi4mol():
    return psi4mol_from_xyzfile(NME2ALA2_OPT_C1)


@pytest.fixture()
def nme2ala2_opt_c2_psi4mol():
    return psi4mol_from_xyzfile(NME2ALA2_OPT_C2)


@pytest.fixture()
def nme2ala2_opt_resp(nme2ala2_opt_c1_psi4mol, nme2ala2_opt_c2_psi4mol):
    reorientations = [(5, 18, 19), (19, 18, 5), (6, 19, 20), (20, 19, 6)]
    conformer_options = dict(reorientations=reorientations,
                             keep_original_conformer_geometry=False,
                             orientation_options=dict(load_input=True))
    resp = psiresp.Resp(nme2ala2_opt_c1_psi4mol, conformer_options=conformer_options,
                        name="nme2ala2", directory_path=data_dir("data/test_resp"),
                        load_input=True)
    resp.add_conformer(nme2ala2_opt_c1_psi4mol)
    resp.add_conformer(nme2ala2_opt_c2_psi4mol)
    resp.generate_orientations()
    resp.finalize_geometries()
    resp.compute_esps()
    return resp


@pytest.fixture()
def methylammonium_psi4mol():
    return psi4mol_from_xyzfile(METHYLAMMONIUM_OPT)


@pytest.fixture()
def methylammonium_resp(methylammonium_psi4mol):
    reorientations = [(1, 5, 7), (7, 5, 1)]
    conformer_options = dict(reorientations=reorientations,
                             keep_original_conformer_geometry=False,
                             orientation_options=dict(load_input=True))

    resp = psiresp.Resp(methylammonium_psi4mol, charge=1,
                        conformer_options=conformer_options,
                        name="methylammonium",
                        load_input=True,
                        directory_path=data_dir("data/test_multiresp"),)
    resp.add_conformer(methylammonium_psi4mol)
    resp.generate_orientations()
    resp.finalize_geometries()
    resp.compute_esps()
    return resp
