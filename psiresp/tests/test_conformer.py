import pytest
import numpy as np

from numpy.testing import assert_almost_equal

from .datafiles import NME2ALA2_C1_ABMAT
from .base import (conformer_from_psi4mol,
                   psi4mol_from_xyzfile)


def test_add_reorientations_with_original(dmso_psi4mol,
                                          dmso_coordinates,
                                          dmso_o1_coordinates,
                                          dmso_o2_coordinates):
    conformer = conformer_from_psi4mol(dmso_psi4mol)
    conformer.keep_original_conformer_geometry = True
    conformer.n_reorientations = 2
    conformer.generate_orientations()
    assert len(conformer.orientations) == 3
    reference = [dmso_coordinates, dmso_o1_coordinates, dmso_o2_coordinates]
    for orientation, ref in zip(conformer.orientations, reference):
        assert_almost_equal(orientation.coordinates, ref)


def test_add_reorientations_without_original(dmso_psi4mol,
                                             dmso_o1_coordinates,
                                             dmso_o2_coordinates):
    conformer = conformer_from_psi4mol(dmso_psi4mol)
    conformer.keep_original_conformer_geometry = False
    conformer.n_reorientations = 2
    conformer.generate_orientations()
    assert len(conformer.orientations) == 2
    reference = [dmso_o1_coordinates, dmso_o2_coordinates]
    for orientation, ref in zip(conformer.orientations, reference):
        assert_almost_equal(orientation.coordinates, ref)


def test_add_orientations_with_updated_options(dmso_psi4mol,
                                               dmso_o1_coordinates):
    conformer = conformer_from_psi4mol(dmso_psi4mol)
    conformer.keep_original_conformer_geometry = False
    conformer.orientation_options.save_output = False
    conformer.orientation_options.load_input = True
    conformer.orientation_name_template = "carrot_{counter:02d}"
    assert conformer.load_input == False
    assert conformer.save_output == False
    conformer.add_orientation(dmso_o1_coordinates)
    assert len(conformer.orientations) == 1
    orientation = conformer.orientations[0]
    assert orientation.name == "carrot_01"
    assert_almost_equal(orientation.coordinates, dmso_o1_coordinates)
    assert orientation.save_output == False
    assert orientation.load_input == True


@pytest.fixture()
def unweighted_ab():
    return np.loadtxt(NME2ALA2_C1_ABMAT)


@pytest.fixture()
def nme2ala2_conformer(nme2ala2_c1_psi4mol):
    conformer = conformer_from_psi4mol(nme2ala2_c1_psi4mol)
    conformer.generate_orientations()
    assert len(conformer.orientations) == 1
    return conformer


def test_get_unweighted_a(nme2ala2_conformer, unweighted_ab):
    actual = nme2ala2_conformer.unweighted_a_matrix
    assert_almost_equal(actual, unweighted_ab[:-1])


def test_get_unweighted_b(nme2ala2_conformer, unweighted_ab):
    actual = nme2ala2_conformer.unweighted_b_matrix
    assert_almost_equal(actual, unweighted_ab[-1])
