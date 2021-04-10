"""
Unit and regression test for the psiresp package.
"""

# Import package, test suite, and other packages as needed
import psiresp
import pytest
import sys
from rdkit import Chem
import numpy as np

from numpy.testing import assert_almost_equal, assert_allclose
from .utils import mol_from_file, charges_from_red_file, datafile


def test_rdkit_gen_confs():
    mol = Chem.MolFromSmiles('C1CCC1OC')
    resp = psiresp.Resp.from_rdmol(mol, n_confs=10, rmsd_threshold=-1)
    assert len(resp.conformers) == 10
    first = resp.conformers[0].coordinates
    second = resp.conformers[1].coordinates
    assert (second - first).sum() > 0

class TestNoOrient:

    @pytest.fixture()
    def resp(self):
        confs = [mol_from_file('dmso_opt_c1.xyz')]
        r = psiresp.Resp.from_molecules(confs, charge=0)
        return r

    def test_esp_1(self, resp):
        charges = resp.fit(restraint=False)
        ref = np.array([-0.43877469,  0.14814998,  0.17996033,  0.18716814,  0.35743529,
                        -0.5085439, -0.46067469,  0.19091725,  0.15500465,  0.18935764])
        assert_almost_equal(charges, ref, decimal=5)

    def test_resp_1(self, resp):
        charges = resp.fit(restraint=True)
        ref = np.array([-0.31436216,  0.11376836,  0.14389443,  0.15583112,  0.30951582,
                        -0.50568553, -0.33670393,  0.15982115,  0.12029174,  0.153629])
        assert_almost_equal(charges, ref, decimal=5)

    def test_esp_2(self, resp):
        charges = resp.run(restraint=False, stage_2=True, opt=False)
        ref = np.array([-0.39199538,  0.15716631,  0.15716631,  0.15716631,  0.35743529,
                        -0.5085439, -0.43701446,  0.16953984,  0.16953984,  0.16953984])
        assert_almost_equal(charges, ref, decimal=5)

    def test_resp_2(self, resp):
        charges = resp.run(restraint=True, stage_2=True, opt=False)
        ref = np.array([-0.25158642,  0.11778735,  0.11778735,  0.11778735,  0.30951582,
                        -0.50568553, -0.29298059,  0.12912489,  0.12912489,  0.12912489])
        assert_almost_equal(charges, ref, decimal=5)

    @pytest.mark.parametrize('stage_2,a,redname', [
        (False, 0.01, 'respA2'),
        (True, 0.0005, 'respA1'),
        (False, 0.0, 'espA1')
    ])
    def test_preorient_dmso(self, stage_2, a, redname):
        fns = ['dmso_opt_c1_o1.xyz', 'dmso_opt_c1_o2.xyz']
        mols = [mol_from_file(f) for f in fns]
        r = psiresp.Resp.from_molecules(mols, charge=0)
        charges = r.run(stage_2=stage_2, opt=False, hyp_a1=a, equal_methyls=True)
        ref = charges_from_red_file('dmso_c1_o2_{}.dat'.format(redname))
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)

    @pytest.mark.parametrize('stage_2,a,redname', [
        (False, 0.01, 'respA2'),
        (True, 0.0005, 'respA1'),
        (False, 0.0, 'espA1')
    ])
    def test_noorient_dmso(self, stage_2, a, redname):
        mols = [mol_from_file('dmso_opt_c1_qmra.xyz')]
        r = psiresp.Resp.from_molecules(mols, charge=0)
        charges = r.run(stage_2=stage_2, opt=False, hyp_a1=a, equal_methyls=True)
        ref = charges_from_red_file('dmso_c1_o0_{}.dat'.format(redname))
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)


@pytest.mark.parametrize('stage_2,a,redname', [
    (False, 0.01, 'respA2'),
    (True, 0.0005, 'respA1'),
    (False, 0.0, 'espA1')
])
class TestRespNoOpt(object):
    molfile = '{molname}_opt_c{conf}.xyz'
    chargefile = '{molname}_c{n_conf}_o{n_orient}_{name}.dat'
    opt = False
    load_files = False

    GRID = datafile('test_resp/grid.dat')
    ESP = datafile('test_resp/grid_esp.dat')

    def load_mols(self, molname, nconf):
        fns = [self.molfile.format(molname=molname, conf=i+1) for i in range(nconf)]
        return [mol_from_file(f) for f in fns]

    def load_charges(self, molname, n_conf, n_orient, name):
        chargefile = self.chargefile.format(molname=molname, n_conf=n_conf,
                                            n_orient=n_orient, name=name)
        return charges_from_red_file(chargefile)

    def test_resp_single_conf(self, stage_2, a, redname, tmpdir):
        confs = self.load_mols('dmso', 1)
        r = psiresp.Resp.from_molecules(confs, charge=0, name='dmso',
                                        load_files=self.load_files,
                                        grid_name=self.GRID,
                                        esp_name=self.ESP)
        with tmpdir.as_cwd():
            charges = r.run(stage_2=stage_2, opt=self.opt, hyp_a1=a, restraint=True,
                            equal_methyls=True, n_orient=2)
        ref = self.load_charges('dmso', 1, 2, redname,)
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)

    def test_resp_multi_conf(self, stage_2, a, redname, tmpdir):
        confs = self.load_mols('ethanol', 2)
        r = psiresp.Resp.from_molecules(confs, charge=0, name='ethanol',
                                        orient=[(1, 5, 8), (8, 5, 1),
                                                (9, 8, 5), (5, 8, 9)],
                                        load_files=self.load_files,
                                        grid_name=self.GRID,
                                        esp_name=self.ESP)
        if not stage_2:
            chrequiv = [[2, 3, 4], [6, 7]]
        else:
            chrequiv = []
        with tmpdir.as_cwd():
            charges = r.run(stage_2=stage_2, opt=self.opt, hyp_a1=a,
                            chrequiv=chrequiv)
        ref = self.load_charges('ethanol', 2, 4, redname)
        assert_allclose(charges, ref, rtol=0.01, atol=5e-4)

    @pytest.mark.parametrize('chargename,chrconstr', [
        ('', [(0, [1, 2, 3, 4, 5, 6]),
              (0, [20, 21, 22, 23, 24, 25]),
              (-0.4157, [7]), (0.2719, [8]),
              (0.5973, [18]), (-0.5679, [19]),
              ]),
        # # skip this b/c tests are timing out and it's replicated in MultiResp
        # ('_multifit_constr', {
        #     0: [20, 21, 22, 23, 24, 25],
        #     0.6163: [18],
        #     -0.5722: [19]}),
    ])
    def test_intra_constraints(self, chrconstr, chargename, stage_2, a, redname,
                               tmpdir):
        confs = self.load_mols('nme2ala2', 2)
        chrequiv = [[10, 14], [11, 12, 13, 15, 16, 17]]
        orient = [(5, 18, 19), (19, 18, 5), (6, 19, 20), (20, 19, 6)]
        r = psiresp.Resp.from_molecules(confs, charge=0, name='nme2ala2',
                                        load_files=self.load_files,
                                        grid_name=self.GRID,
                                        esp_name=self.ESP)
        with tmpdir.as_cwd():
            charges = r.run(stage_2=stage_2, opt=self.opt, hyp_a1=a,
                            equal_methyls=False, chrequiv=chrequiv,
                            chrconstr=chrconstr, orient=orient)
        ref = self.load_charges('nme2ala2'+chargename, 2, 4, redname)
        assert_allclose(charges, ref, rtol=0.01, atol=1e-4)


@pytest.mark.fast
class TestLoadNoOpt(TestRespNoOpt):
    load_files = True


@pytest.mark.optimize
@pytest.mark.slow
class TestOpt(TestRespNoOpt):
    molfile = '{molname}_c{conf}.xyz'
    opt = True
