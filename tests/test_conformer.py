
import psiresp
import pytest
import os
import numpy as np

from numpy.testing import assert_almost_equal, assert_equal, assert_allclose
from .utils import mol_from_file


class TestConformer:

    @pytest.fixture()
    def geometry(self):
        return mol_from_file('nme2ala2_c1.xyz')

    @pytest.fixture()
    def opt_mol(self):
        return mol_from_file('nme2ala2_opt_c1.xyz')

    @pytest.fixture(scope='function')
    def conformer(self, geometry):
        return psiresp.Conformer(geometry)

    def test_init_conformer_defaults(self, conformer):
        assert conformer.name == 'default'
        assert conformer.charge == 0
        assert conformer.molecule.molecular_charge() == 0
        assert conformer.multiplicity == 1
        assert conformer.molecule.multiplicity() == 1
        assert conformer.n_atoms == 25
        assert_equal(conformer.symbols, list('CHHHCONHCCHHHCHHHCONHCHHH'))
        assert len(conformer.orientations) == 1
        assert len(conformer._orientations) == 0

    def test_init_conformer_options(self, geometry):
        orient = [(5, 18, 19), (19, 18, 5)]
        conformer = psiresp.Conformer(geometry, name='nme2ala2', charge=2,
                                      multiplicity=2, orient=orient)
        assert conformer.name == 'nme2ala2'
        assert conformer.charge == 2
        assert conformer.molecule.molecular_charge() == 2
        assert conformer.multiplicity == 2
        assert conformer.molecule.multiplicity() == 2
        assert conformer.n_atoms == 25
        assert_equal(conformer.symbols, list('CHHHCONHCCHHHCHHHCONHCHHH'))
        assert len(conformer._orientations) == 0
        assert len(conformer.orientations) == 2
        assert len(conformer._orientations) == 2
        assert conformer.orientations[0].name == 'nme2ala2_o1'
        assert conformer.orientations[1].name == 'nme2ala2_o2'

    def test_add_orientations(self, conformer):
        assert len(conformer.orientations) == 1  # original molecule
        conformer.add_orientations(orient=[(5, 18, 19), (19, 18, 5)])
        assert len(conformer.orientations) == 2
        conformer.add_orientations(orient=[(6, 19, 20), (20, 19, 6)])
        assert len(conformer.orientations) == 4

    @pytest.mark.optimize
    @pytest.mark.slow
    def test_add_orientations_opt(self):
        conformer = psiresp.Conformer(mol_from_file('nme2ala2_c1.xyz'))
        assert len(conformer.orientations) == 1  # original molecule
        conformer.add_orientations(orient=[(5, 18, 19), (19, 18, 5)])
        xyz_pre_opt = conformer.orientations[0].coordinates
        conformer.optimize_geometry()
        xyz_post_opt = conformer.orientations[0].coordinates
        diff = np.linalg.norm(xyz_post_opt-xyz_pre_opt)
        assert diff > 0.01

    @pytest.mark.optimize
    @pytest.mark.slow
    @pytest.mark.parametrize('save_xyz,save_files', [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ])
    def test_optimize_geometry(self, conformer, opt_mol, save_xyz,
                               save_files, tmpdir):
        xyz = 'default_opt.xyz'
        log = 'default_opt.log'
        opt = opt_mol.geometry().np
        with tmpdir.as_cwd():
            conformer.optimize_geometry(save_opt_geometry=save_xyz,
                                        save_files=save_files)
            assert_allclose(conformer.molecule.geometry().np, opt,
                            rtol=0.05, atol=1e-4)
            if save_files:
                assert os.path.exists(log)
            else:
                assert not os.path.exists(log)

            if save_xyz:
                assert os.path.exists(xyz)
            else:
                assert not os.path.exists(xyz)

    def test_clone(self, conformer):
        new = conformer.clone()
        assert new.molecule is not conformer.molecule
        assert_almost_equal(new.molecule.geometry().np,
                            conformer.molecule.geometry().np)
        assert new.name == 'default_copy'
        assert new.charge == 0
        assert new.multiplicity == 1
        assert len(new.orientations) == len(conformer.orientations)
        assert_equal(new._orient, conformer._orient)

