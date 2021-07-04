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
    return psi4mol_from_xyzfile(DMSO)


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


@pytest.fixture(scope="session")
def nme2ala2_opt_resp_original():
    reorientations = [(5, 18, 19), (19, 18, 5), (6, 19, 20), (20, 19, 6)]
    conf1 = psi4mol_from_xyzfile(NME2ALA2_OPT_C1)
    conf2 = psi4mol_from_xyzfile(NME2ALA2_OPT_C2)
    conformer_options = dict(reorientations=reorientations,
                             keep_original_conformer_geometry=False,
                             orientation_options=dict(load_input=True))
    resp = psiresp.Resp(conf1, conformer_options=conformer_options,
                        name="nme2ala2", directory_path=data_dir("data/test_resp"),
                        load_input=True)
    resp.add_conformer(conf1)
    resp.add_conformer(conf2)
    resp.generate_orientations()
    resp.finalize_geometries()
    resp.compute_esps()
    return resp


@pytest.fixture()
def nme2ala2_opt_resp(nme2ala2_opt_resp_original):
    return nme2ala2_opt_resp_original.copy()


@pytest.fixture()
def methylammonium_resp():
    reorientations = [(1, 5, 7), (7, 5, 1)]
    conformer_options = dict(reorientations=reorientations,
                             keep_original_conformer_geometry=False,
                             orientation_options=dict(load_input=True))

    mol = psi4mol_from_xyzfile(METHYLAMMONIUM_OPT)
    resp = psiresp.Resp(mol, charge=1,
                        conformer_options=conformer_options,
                        name="methylammonium",
                        load_input=True,
                        directory_path=data_dir("data/test_multiresp/methylammonium"),)
    resp.add_conformer(mol)
    resp.generate_orientations()
    resp.finalize_geometries()
    resp.compute_esps()
    return resp
