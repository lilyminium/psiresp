"""
Unit and regression test for the psiresp package.
"""

# Import package, test suite, and other packages as needed
import psiresp
import pytest
import sys
import numpy as np

from numpy.testing import assert_almost_equal, assert_allclose
from .utils import (mol_from_file, mol_from_mol2, charges_from_mol2,
                    charges_from_red_file, charges_from_itp_file)


@pytest.fixture()
def dmso():
    return mol_from_file('dmso_c1.xyz')


@pytest.fixture()
def dmso_opt():
    return mol_from_file('dmso_opt_c1.xyz')


@pytest.fixture()
def dmso_orients():
    fns = ['dmso_opt_c1_o1.xyz', 'dmso_opt_c1_o2.xyz']
    return [mol_from_file(f) for f in fns]


class BaseTestRespConfigNoOpt:
    """Charges from R.E.D. jobs"""

    def test_resp_noopt_noorient(self, dmso_orients, tmpdir, ref):
        r = self.cls.from_molecules(dmso_orients, charge=0)
        r0 = psiresp.Resp.from_molecules(dmso_orients, charge=0)
        with tmpdir.as_cwd():
            charges = r.run(opt=False, equal_methyls=True)
        assert_allclose(charges, ref, rtol=0.05, atol=1e-4)

    def test_resp_noopt_orient(self, dmso_opt, tmpdir, ref):
        r = self.cls.from_molecules([dmso_opt])
        with tmpdir.as_cwd():
            charges = r.run(n_orient=2, opt=False, equal_methyls=True)
        assert_allclose(charges, ref, rtol=0.05, atol=1e-4)


class BaseTestRespConfigOpt:
    """Charges from R.E.D. jobs"""
    name = None
    cls = None
    nconf = 1
    norient = 2

    @property
    def chargefile(self):
        return 'dmso_c{}_o{}_{}.dat'.format(self.nconf, self.norient,
                                            self.name)

    @pytest.fixture()
    def ref(self):
        return charges_from_red_file(self.chargefile)

    @pytest.mark.slow
    def test_resp_opt(self, dmso, tmpdir, ref):
        r = self.cls.from_molecules([dmso])
        with tmpdir.as_cwd():
            charges = r.run(opt=True, n_orient=2, equal_methyls=True,
                            save_files=False, save_opt_geometry=False)
        assert_allclose(charges, ref, rtol=0.05, atol=1e-4)


class TestRespA1(BaseTestRespConfigNoOpt, BaseTestRespConfigOpt):
    cls = psiresp.RespA1
    name = 'respA1'


class TestRespA2(BaseTestRespConfigNoOpt, BaseTestRespConfigOpt):
    cls = psiresp.RespA2
    name = 'respA2'


class TestEspA1(BaseTestRespConfigNoOpt, BaseTestRespConfigOpt):
    cls = psiresp.EspA1
    name = 'espA1'

# whyyyy 😭
@pytest.mark.skip(reason='Psi4 minimises to a very different geometry to GAMESS')
class TestEspA2(BaseTestRespConfigOpt):
    cls = psiresp.EspA2
    name = 'espA2'


class BaseTestATBResp:
    """charges from ATB jobs"""
    cls = psiresp.ATBResp
    name = 'ATB'
    molname = None

    @property
    def chargefile(self):
        return '{}_c1_o0_ATB.itp'.format(self.molname)

    @property
    def molfile(self):
        return '{}_c1_ATB.xyz'.format(self.molname)

    @pytest.fixture(scope='function')
    def resp(self):
        opt = [mol_from_file('{}_c1_ATB_opt.xyz'.format(self.molname))]
        r = self.cls.from_molecules(opt)
        return r

    @pytest.fixture()
    def ref(self):
        return charges_from_itp_file(self.chargefile)

    def test_resp_noopt(self, tmpdir, resp, ref):
        with tmpdir.as_cwd():
            charges = resp.run(opt=False, n_orient=2, equal_methyls=True,
                            vdw_point_density=1, solvent='water')
        # no idea which point density ATB uses
        assert_allclose(charges, ref, rtol=0.05, atol=1e-3)

    @pytest.mark.slow
    def test_resp_opt(self, tmpdir, ref):
        mol = [mol_from_file(self.molfile)]
        r = self.cls.from_molecules(mol)
        with tmpdir.as_cwd():
            charges = r.run(opt=True, n_orient=2, equal_methyls=True,
                            vdw_point_density=1, solvent='water')
        assert_allclose(charges, ref, rtol=0.05, atol=1e-3)


class TestATBRespMethane(BaseTestATBResp):
    molname = 'methane'

# fails, idk why; ATB charges are also different from Malde et al 2011
@pytest.mark.skip(reason='Fails? ATB charges are also v. different from Malde et al 2011')
class TestATBRespIsopropanol(BaseTestATBResp):
    molname = 'isopropanol'


class TestATBRespDMSO(BaseTestATBResp):
    molname = 'dmso'

@pytest.mark.skip(reason='Fails? Are the mol2 geometries not minimised?')
@pytest.mark.resp2
@pytest.mark.slow
@pytest.mark.parametrize('name', ['C00', 'C64', 'MPE', 'PXY', 'TOL'])
@pytest.mark.parametrize('delta', [0, 0.6, 1])
class TestResp2Charges:
    """Charges from RESP2 paper Schauperl et al., 2020.

    See dataset for more: https://doi.org/10.5281/zenodo.3593762
    """

    def test_resp2_noopt(self, name, delta, tmpdir):
        fn = '{}_R2_{:d}.mol2'.format(name, int(delta*100))
        mols = [mol_from_mol2(fn)]
        ref = charges_from_mol2(fn)
        with tmpdir.as_cwd():
            r = psiresp.Resp2.from_molecules(mols, charge=0)
            charges = r.run(opt=False, n_orient=4, delta=delta)
        assert_almost_equal(charges, ref, decimal=3)


@pytest.mark.resp2
class TestResp2Ethanol:
    """Charges from ethanol example in MSchauperl/resp2.

    Assert almost equal with decimal=3 because not sure when values are rounded.
    See repo for more: https://github.com/MSchauperl/RESP2
    """

    solv = np.array([-0.2416,  0.3544, -0.6898,  0.0649,  0.0649,
                     0.0649, -0.0111, -0.0111,  0.4045])

    gas = np.array([-0.2300,  0.3063, -0.5658,  0.0621,  0.0621,
                    0.0621, -0.0153, -0.0153,  0.3339])

    ref = np.array([-0.2358,  0.33035, -0.6278,  0.0635,
                    0.0635,  0.0635, -0.0132, -0.0132,  0.3692])

    def test_ethanol_no_opt(self, tmpdir):
        mols = [mol_from_file('ethanol_resp2_opt_c1.xyz'),
                mol_from_file('ethanol_resp2_opt_c2.xyz')]
        r = psiresp.Resp2.from_molecules(mols, charge=0)
        with tmpdir.as_cwd():
            charges = r.run(opt=False, n_orient=0, delta=0.5)
        assert_almost_equal(r.gas_charges, self.gas, decimal=3)
        assert_almost_equal(r.solv_charges, self.solv, decimal=3)
        assert_almost_equal(charges, self.ref, decimal=3)

    @pytest.mark.slow
    def test_ethanol_opt(self, tmpdir):
        mols = [mol_from_file('ethanol_resp2_c1.xyz'),
                mol_from_file('ethanol_resp2_c2.xyz')]
        r = psiresp.Resp2.from_molecules(mols, charge=0)
        with tmpdir.as_cwd():
            charges = r.run(opt=True, n_orient=0, delta=0.5)
        assert_almost_equal(r.gas_charges, self.gas, decimal=3)
        assert_almost_equal(r.solv_charges, self.solv, decimal=3)
        assert_almost_equal(charges, self.ref, decimal=3)
