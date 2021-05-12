"""
Unit and regression test for the psiresp package.
"""

# Import package, test suite, and other packages as needed
import psiresp
import os
import pytest
import sys
from rdkit import Chem
import numpy as np
from distutils.dir_util import copy_tree

from numpy.testing import assert_almost_equal, assert_allclose
from .utils import mol_from_file, charges_from_red_file, datafile


def test_rdkit_gen_confs(tmpdir):
    mol = Chem.MolFromSmiles('C1CCC1OC')
    with tmpdir.as_cwd():
        resp = psiresp.Resp.from_rdmol(mol, n_confs=10, rmsd_threshold=-1)
    assert len(resp.conformers) == 10
    first = resp.conformers[0].coordinates
    second = resp.conformers[1].coordinates
    assert (second - first).sum() > 0


class TestNoOrient:

    esp_1 = np.array([
        -0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529, -0.5085439, -0.46067469, 0.19091725, 0.15500465,
        0.18935764
    ])

    resp_1 = np.array([
        -0.31436216, 0.11376836, 0.14389443, 0.15583112, 0.30951582, -0.50568553, -0.33670393, 0.15982115, 0.12029174,
        0.153629
    ])

    esp_2 = np.array([
        -0.39199538, 0.15716631, 0.15716631, 0.15716631, 0.35743529, -0.5085439, -0.43701446, 0.16953984, 0.16953984,
        0.16953984
    ])

    resp_2 = np.array([
        -0.25158642, 0.11778735, 0.11778735, 0.11778735, 0.30951582, -0.50568553, -0.29298059, 0.12912489, 0.12912489,
        0.12912489
    ])

    @pytest.fixture()
    def resp_c1(self, tmpdir):
        with tmpdir.as_cwd():
            confs = [mol_from_file('dmso_opt_c1.xyz')]
            r = psiresp.Resp.from_molecules(confs, charge=0)
        return r

    @pytest.fixture()
    def resp_c1_o2(self, tmpdir):
        fns = ['dmso_opt_c1_o1.xyz', 'dmso_opt_c1_o2.xyz']
        mols = [mol_from_file(f) for f in fns]
        with tmpdir.as_cwd():
            r = psiresp.Resp.from_molecules(mols, charge=0)
        return r

    @pytest.fixture()
    def resp_opt_qmra(self, tmpdir):
        mols = [mol_from_file('dmso_opt_c1_qmra.xyz')]
        with tmpdir.as_cwd():
            r = psiresp.Resp.from_molecules(mols, charge=0)
        return r

    def test_esp_1(self, resp_c1):
        charges = resp_c1.fit(resp_options=dict(restrained=False))
        assert_almost_equal(charges.unrestrained_charges, self.esp_1, decimal=5)
        assert charges.restrained_charges is None

    def test_resp_1(self, resp_c1):
        charges = resp_c1.fit(resp_options=dict(restrained=True))
        assert_almost_equal(charges.unrestrained_charges, self.esp_1, decimal=5)
        assert_almost_equal(charges.restrained_charges, self.resp_1, decimal=5)

    def test_esp_2(self, resp_c1):
        charges = resp_c1.run(restrained=False, stage_2=True)
        ref = np.array([
            -0.39199538, 0.15716631, 0.15716631, 0.15716631, 0.35743529, -0.5085439, -0.43701446, 0.16953984,
            0.16953984, 0.16953984
        ])
        assert_almost_equal(charges, ref, decimal=5)

    def test_resp_2(self, resp_c1):
        charges = resp_c1.run(restrained=True, stage_2=True)
        ref = np.array([
            -0.25158642, 0.11778735, 0.11778735, 0.11778735, 0.30951582, -0.50568553, -0.29298059, 0.12912489,
            0.12912489, 0.12912489
        ])
        assert_almost_equal(charges, ref, decimal=5)

    @pytest.mark.parametrize('stage_2,a,redname', [(False, 0.01, 'respA2'), (True, 0.0005, 'respA1'),
                                                   (False, 0.0, 'espA1')])
    def test_preorient_dmso(self, stage_2, a, redname, resp_c1_o2):
        charge_options = psiresp.options.ChargeOptions(equivalent_methyls=True)
        charges = resp_c1_o2.run(stage_2=stage_2, hyp_a1=a, charge_constraint_options=charge_options)
        ref = charges_from_red_file('dmso_c1_o2_{}.dat'.format(redname))
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)

    @pytest.mark.parametrize('stage_2,a,redname', [(False, 0.01, 'respA2'), (True, 0.0005, 'respA1'),
                                                   (False, 0.0, 'espA1')])
    def test_noorient_dmso(self, stage_2, a, redname, resp_opt_qmra):
        charge_options = psiresp.options.ChargeOptions(equivalent_methyls=True)
        charges = resp_opt_qmra.run(stage_2=stage_2, hyp_a1=a, charge_constraint_options=charge_options)
        ref = charges_from_red_file('dmso_c1_o0_{}.dat'.format(redname))
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)


class TestRespNoOpt(object):
    molfile = '{molname}_opt_c{conf}.xyz'
    chargefile = '{molname}_c{n_conf}_o{n_orient}_{name}.dat'
    opt = False
    load_files = False

    GRID = datafile('test_resp/grid.dat')
    ESP = datafile('test_resp/grid_esp.dat')

    def load_mols(self, molname, nconf):
        fns = [self.molfile.format(molname=molname, conf=i + 1) for i in range(nconf)]
        return [mol_from_file(f) for f in fns]

    def load_charges(self, molname, n_conf, n_orient, name):
        chargefile = self.chargefile.format(molname=molname, n_conf=n_conf, n_orient=n_orient, name=name)
        return charges_from_red_file(chargefile)

    def create_resp_dmso(self, tmpdir):
        confs = self.load_mols('dmso', 1)
        io_options = psiresp.options.IOOptions(load_from_files=self.load_files)
        orientation_options = psiresp.options.OrientationOptions(n_reorientations=2, keep_original=False)
        with tmpdir.as_cwd():
            r = psiresp.Resp.from_molecules(confs,
                                            charge=0,
                                            name="dmso",
                                            optimize_geometry=self.opt,
                                            orientation_options=orientation_options,
                                            io_options=io_options)
        options = r.conformers[0].orientation_options
        assert len(r.conformers[0].orientations) == 2
        assert options.reorientations[0] == (1, 5, 6)
        assert options.reorientations[1] == (6, 5, 1)
        return r

    def create_resp_ethanol(self, tmpdir):
        confs = self.load_mols('ethanol', 2)
        io_options = psiresp.options.IOOptions(load_from_files=self.load_files)
        orient = [(1, 5, 8), (8, 5, 1), (9, 8, 5), (5, 8, 9)]
        orientation_options = psiresp.options.OrientationOptions(reorientations=orient, keep_original=False)
        with tmpdir.as_cwd():
            r = psiresp.Resp.from_molecules(confs,
                                            charge=0,
                                            name="ethanol",
                                            optimize_geometry=self.opt,
                                            orientation_options=orientation_options,
                                            io_options=io_options)
        return r

    def create_resp_nme2ala2(self, tmpdir):
        confs = self.load_mols('nme2ala2', 2)
        io_options = psiresp.options.IOOptions(load_from_files=self.load_files)
        orient = [(5, 18, 19), (19, 18, 5), (6, 19, 20), (20, 19, 6)]
        orientation_options = psiresp.options.OrientationOptions(reorientations=orient, keep_original=False)
        with tmpdir.as_cwd():
            r = psiresp.Resp.from_molecules(confs,
                                            charge=0,
                                            name="nme2ala2",
                                            optimize_geometry=self.opt,
                                            orientation_options=orientation_options,
                                            io_options=io_options)
        return r

    @pytest.fixture()
    def dmso_charge_options(self):
        return psiresp.options.ChargeOptions(equivalent_methyls=True)

    @pytest.mark.parametrize('stage_2,a,redname', [(False, 0.01, 'respA2'), (True, 0.0005, 'respA1'),
                                                   (False, 0.0, 'espA1')])
    def test_resp_single_conf(self, stage_2, a, redname, dmso_charge_options, executor, tmpdir):
        resp_dmso = self.create_resp_dmso(tmpdir)
        with tmpdir.as_cwd():
            if self.load_files:
                copy_tree(datafile("test_resp"), str(tmpdir))
            charges = resp_dmso.run(stage_2=stage_2,
                                    hyp_a1=a,
                                    restrained=True,
                                    charge_constraint_options=dmso_charge_options,
                                    executor=executor)
        ref = self.load_charges(
            'dmso',
            1,
            2,
            redname,
        )
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)

    @pytest.mark.parametrize('stage_2,a,redname', [(False, 0.01, 'respA2'), (True, 0.0005, 'respA1'),
                                                   (False, 0.0, 'espA1')])
    def test_resp_multi_conf(self, stage_2, a, redname, executor, tmpdir):
        resp_ethanol = self.create_resp_ethanol(tmpdir)
        confs = self.load_mols('ethanol', 2)
        if not stage_2:
            chrequiv = [[2, 3, 4], [6, 7]]
        else:
            chrequiv = []

        charge_options = psiresp.options.ChargeOptions(charge_equivalences=chrequiv)
        with tmpdir.as_cwd():
            if self.load_files:
                copy_tree(datafile("test_resp"), str(tmpdir))
            charges = resp_ethanol.run(stage_2=stage_2,
                                       hyp_a1=a,
                                       executor=executor,
                                       charge_constraint_options=charge_options)
        ref = self.load_charges('ethanol', 2, 4, redname)
        assert_allclose(charges, ref, rtol=0.01, atol=5e-4)

    @pytest.mark.parametrize(
        'stage_2,a,redname',
        [
            # (False, 0.01, 'respA2'),
            (True, 0.0005, 'respA1'),
            (False, 0.0, 'espA1')
        ])
    def test_intra_constraints(self, stage_2, a, redname, executor, tmpdir):
        resp_nme2ala2 = self.create_resp_nme2ala2(tmpdir)
        chargename = ""
        chrconstr = [(0, [1, 2, 3, 4, 5, 6]), (0, [20, 21, 22, 23, 24, 25]), (-0.4157, [7]), (0.2719, [8]),
                     (0.5973, [18]), (-0.5679, [19])]
        chrequiv = [[10, 14], [11, 12, 13, 15, 16, 17]]
        charge_options = psiresp.options.ChargeOptions(charge_constraints=chrconstr,
                                                       charge_equivalences=chrequiv,
                                                       equivalent_methyls=False,
                                                       equivalent_sp3_hydrogens=False)
        with tmpdir.as_cwd():
            if self.load_files:
                copy_tree(datafile("test_resp"), str(tmpdir))
            charges = resp_nme2ala2.run(stage_2=stage_2,
                                        hyp_a1=a,
                                        charge_constraint_options=charge_options,
                                        executor=executor)
        ref = self.load_charges('nme2ala2' + chargename, 2, 4, redname)
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)

    @pytest.mark.slow
    @pytest.mark.parametrize('stage_2,a,redname', [(False, 0.01, 'respA2'), (True, 0.0005, 'respA1'),
                                                   (False, 0.0, 'espA1')])
    def test_intra_multifit_constraints(self, stage_2, a, redname, executor, tmpdir):
        resp_nme2ala2 = self.create_resp_nme2ala2(tmpdir)
        chargename = "_multifit_constr"
        chrconstr = {0: [20, 21, 22, 23, 24, 25], 0.6163: [18], -0.5722: [19]}
        chrequiv = [[10, 14], [11, 12, 13, 15, 16, 17]]
        charge_options = psiresp.options.ChargeOptions(charge_constraints=chrconstr,
                                                       charge_equivalences=chrequiv,
                                                       equivalent_methyls=False,
                                                       equivalent_sp3_hydrogens=False)
        with tmpdir.as_cwd():
            if self.load_files:
                copy_tree(datafile("test_resp"), str(tmpdir))
            charges = resp_nme2ala2.run(stage_2=stage_2,
                                        hyp_a1=a,
                                        charge_constraint_options=charge_options,
                                        executor=executor)
        ref = self.load_charges('nme2ala2' + chargename, 2, 4, redname)
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)


@pytest.mark.fast
class TestLoadNoOpt(TestRespNoOpt):
    load_files = True


@pytest.mark.optimize
@pytest.mark.slow
class TestOpt(TestRespNoOpt):
    molfile = '{molname}_c{conf}.xyz'
    opt = True
