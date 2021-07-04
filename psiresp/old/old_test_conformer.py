import psiresp
import pytest
import os
import numpy as np

from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose)
from .utils import mol_from_file
from .datafiles import ABMAT


class TestConformer:
    @pytest.fixture()
    def psi4mol(self):
        return mol_from_file('nme2ala2_c1.xyz')

    @pytest.fixture()
    def opt_mol(self):
        return mol_from_file('nme2ala2_opt_c1.xyz')

    @pytest.fixture(scope='function')
    def conformer(self, psi4mol):
        return psiresp.Conformer(psi4mol)

    @pytest.mark.fast
    def test_init_conformer_defaults(self, conformer):
        assert conformer.name == 'conf'
        assert conformer.charge == 0
        assert conformer.psi4mol.molecular_charge() == 0
        assert conformer.multiplicity == 1
        assert conformer.psi4mol.multiplicity() == 1
        assert conformer.n_atoms == 25
        assert_equal(conformer.symbols, list('CHHHCONHCCHHHCHHHCONHCHHH'))
        assert len(conformer.orientations) == 1
        # assert len(conformer.orientations) == 1

    @pytest.mark.fast
    def test_init_conformer_options(self, psi4mol):
        orient = [(5, 18, 19), (19, 18, 5)]
        options = psiresp.options.OrientationOptions(reorientations=orient)

        conformer = psiresp.Conformer(psi4mol, name='nme2ala2', charge=2, multiplicity=2, orientation_options=options)
        assert conformer.name == 'nme2ala2'
        assert conformer.charge == 2
        assert conformer.psi4mol.molecular_charge() == 2
        assert conformer.multiplicity == 2
        assert conformer.psi4mol.multiplicity() == 2
        assert conformer.n_atoms == 25
        assert_equal(conformer.symbols, list('CHHHCONHCCHHHCHHHCONHCHHH'))
        assert len(conformer.orientations) == 3
        assert conformer.orientations[0].name == 'nme2ala2_o001'

    # @pytest.mark.optimize
    # @pytest.mark.slow
    # @pytest.mark.parametrize('save_xyz,save_files', [
    #     (False, False),
    #     (False, True),
    #     (True, False),
    #     (True, True),
    # ])
    # def test_optimize_psi4mol(self, conformer, opt_mol, save_xyz,
    #                            save_files, tmpdir):
    #     xyz = 'default_opt.xyz'
    #     log = 'default_opt.log'
    #     opt = opt_mol.psi4mol().np
    #     with tmpdir.as_cwd():
    #         conformer.optimize_psi4mol(save_opt_psi4mol=save_xyz,
    #                                     save_files=save_files)
    #         assert_allclose(conformer.psi4mol.psi4mol().np, opt,
    #                         rtol=0.05, atol=1e-4)
    #         if save_files:
    #             assert os.path.exists(log)
    #         else:
    #             assert not os.path.exists(log)

    #         if save_xyz:
    #             assert os.path.exists(xyz)
    #         else:
    #             assert not os.path.exists(xyz)

    @pytest.mark.fast
    def test_clone(self, conformer):
        new = conformer.clone()
        assert new.psi4mol is not conformer.psi4mol
        assert_almost_equal(new.psi4mol.geometry().np, conformer.psi4mol.geometry().np)
        assert new.name == 'conf_copy'
        assert new.charge == 0
        assert new.multiplicity == 1
        assert len(new.orientations) == len(conformer.orientations)
        assert new.orientation_options == conformer.orientation_options

    def test_get_unweighted_ab(self, conformer):
        ref = np.loadtxt(ABMAT)
        ref_a = ref[:-1]
        ref_b = ref[-1]
        assert_almost_equal(conformer.get_unweighted_a_matrix(), ref_a)
        assert_almost_equal(conformer.get_unweighted_b_matrix(), ref_b)
