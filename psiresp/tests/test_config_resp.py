"""
Unit and regression test for the psiresp package.
"""

# Import package, test suite, and other packages as needed
import psiresp
import pytest

import numpy as np
from numpy.testing import assert_almost_equal

from .base import charges_from_red_file, data_dir
from .datafiles import (DMSO_RESPA2_CHARGES, DMSO_RESPA1_CHARGES, DMSO_ESPA1_CHARGES,
                        NME2ALA2_OPT_C1, NME2ALA2_OPT_C2, METHYLAMMONIUM_OPT,
                        AMM_NME_OPT_RESPA2_CHARGES,
                        AMM_NME_OPT_RESPA1_CHARGES,
                        AMM_NME_OPT_ESPA1_CHARGES)

# from numpy.testing import assert_almost_equal, assert_allclose
# from .utils import (mol_from_file, mol_from_mol2, charges_from_mol2, charges_from_red_file, charges_from_itp_file,
#                     datafile, molfile)


# @pytest.fixture()
# def dmso():
#     return mol_from_file('dmso_c1.xyz')


# @pytest.fixture()
# def dmso_opt():
#     return mol_from_file('dmso_opt_c1.xyz')


# @pytest.fixture()
# def dmso_orients():
#     fns = ['dmso_opt_c1_o1.xyz', 'dmso_opt_c1_o2.xyz']
#     return [mol_from_file(f) for f in fns]


# class BaseTestRespConfigNoOpt:
#     """Charges from R.E.D. jobs"""

#     def test_resp_noopt_noorient(self, dmso_orients, ref):
#         charge_options = psiresp.ChargeOptions(equivalent_methyls=True)
#         r = self.cls.from_molecules(dmso_orients, charge=0)
#         charges = r.run(charge_constraint_options=charge_options)
#         assert_allclose(charges, ref, rtol=0.05, atol=1e-4)

#     def test_resp_noopt_orient(self, dmso_opt, ref):
#         charge_options = psiresp.ChargeOptions(equivalent_methyls=True)
#         orientation_options = psiresp.OrientationOptions(n_reorientations=2)
#         r = self.cls.from_molecules([dmso_opt], orientation_options=orientation_options)
#         charges = r.run(charge_constraint_options=charge_options)
#         assert_allclose(charges, ref, rtol=0.05, atol=1e-4)

@pytest.mark.parametrize("config_class, charge_file", [
    (psiresp.EspA1, DMSO_ESPA1_CHARGES),
    (psiresp.RespA1, DMSO_RESPA1_CHARGES),
    (psiresp.RespA2, DMSO_RESPA2_CHARGES),

])
def test_dmso_resp_config(config_class, charge_file, dmso_o1_psi4mol, dmso_o2_psi4mol):
    charge_options = dict(charge_equivalences=[(1, 7), (2, 3, 4, 8, 9, 10)])
    resp = config_class(psi4mol=dmso_o1_psi4mol,
                        charge_constraint_options=charge_options)
    resp.generate_conformers()
    conformer = resp.conformers[0]
    print("confs", len(resp.conformers))
    conformer.add_orientation(dmso_o1_psi4mol)
    conformer.add_orientation(dmso_o2_psi4mol)
    print("ori", conformer.n_orientations)
    charges = resp.run()

    reference = charges_from_red_file(charge_file)
    assert_almost_equal(charges, reference, decimal=3)


@pytest.mark.parametrize("config_class, charge_file", [
    (psiresp.MultiEspA1, AMM_NME_OPT_ESPA1_CHARGES),
    (psiresp.MultiRespA2, AMM_NME_OPT_RESPA2_CHARGES),
    (psiresp.MultiRespA1, AMM_NME_OPT_RESPA1_CHARGES),

])
def test_multiresp_config(config_class, charge_file,
                          nme2ala2_opt_c1_psi4mol, nme2ala2_opt_c2_psi4mol,
                          methylammonium_psi4mol):

    overall = dict(charge_constraints=[(0, [(1, 1), (1, 2), (1, 3), (1, 4),
                                            (2, 1), (2, 2), (2, 3), (2, 4),
                                            (2, 5), (2, 6), (2, 7), (2, 8)]),

                                       (0, [(2, 20), (2, 21), (2, 22),
                                            (2, 23), (2, 24), (2, 25)]),
                                       (0.6163, [(2, 18)]),
                                       (-0.5722, [(2, 19)]),
                                       ],
                   charge_equivalences=[[(2, 10), (2, 14)],
                                        [(2, 11), (2, 12), (2, 13),
                                         (2, 15), (2, 16), (2, 17)],
                                        [(1, 6), (1, 7), (1, 8)],
                                        ],
                   symmetric_methyls=False)
    multiresp = config_class(charge_constraint_options=overall,
                             resp_options=dict(conformer_options=dict(orientation_options=dict(load_input=True))),
                             directory_path=data_dir("data/test_multiresp"))

    methylammonium = multiresp.add_resp(methylammonium_psi4mol, charge=1, name="methylammonium")
    methylammonium.conformer_options = dict(reorientations=[(1, 5, 7), (7, 5, 1)])
    nme2ala2 = multiresp.add_resp(nme2ala2_opt_c1_psi4mol, name="nme2ala2")
    nme2ala2.conformer_options.reorientations = [(5, 18, 19), (19, 18, 5), (6, 19, 20), (20, 19, 6)]

    nme2ala2.add_conformer(nme2ala2_opt_c1_psi4mol)
    nme2ala2.add_conformer(nme2ala2_opt_c2_psi4mol)
    methylammonium.add_conformer(methylammonium_psi4mol)
    multiresp.generate_orientations()
    assert multiresp.n_orientations == 10
    charges = multiresp.run()

    assert_almost_equal(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0)
    assert_almost_equal(charges[[27, 28, 29, 30, 31, 32]].sum(), 0)
    assert_almost_equal(charges[25], 0.6163)
    assert_almost_equal(charges[26], -0.5722)
    assert_almost_equal(charges[18], charges[22])

    reference = np.concatenate(charges_from_red_file(charge_file))
    assert_almost_equal(charges, reference, decimal=3)


# class BaseTestRespConfigOpt:
#     """Charges from R.E.D. jobs"""
#     name = None
#     cls = None
#     nconf = 1
#     norient = 2

#     @property
#     def chargefile(self):
#         return 'dmso_c{}_o{}_{}.dat'.format(self.nconf, self.norient, self.name)

#     @pytest.fixture()
#     def ref(self):
#         return charges_from_red_file(self.chargefile)

#     @pytest.mark.slow
#     def test_resp_opt(self, dmso, ref):
#         orientation_options = psiresp.OrientationOptions(n_reorientations=2)
#         charge_options = psiresp.ChargeOptions(equivalent_methyls=True)
#         r = self.cls.from_molecules([dmso], optimize_geometry=True,
#                                     orientation_options=orientation_options)
#         charges = r.run(charge_constraint_options=charge_options)
#         assert_allclose(charges, ref, rtol=0.05, atol=1e-4)


# class TestRespA1(BaseTestRespConfigNoOpt, BaseTestRespConfigOpt):
#     cls = psiresp.RespA1
#     name = 'respA1'


# class TestRespA2(BaseTestRespConfigNoOpt, BaseTestRespConfigOpt):
#     cls = psiresp.RespA2
#     name = 'respA2'


# class TestEspA1(BaseTestRespConfigNoOpt, BaseTestRespConfigOpt):
#     cls = psiresp.EspA1
#     name = 'espA1'


# # whyyyy 😭
# @pytest.mark.skip(reason='Psi4 minimises to a very different geometry to GAMESS')
# class TestEspA2(BaseTestRespConfigOpt):
#     cls = psiresp.EspA2
#     name = 'espA2'


# class BaseTestATBResp:
#     """charges from ATB jobs"""
#     cls = psiresp.ATBResp
#     name = 'ATB'
#     molname = None

#     @property
#     def chargefile(self):
#         return '{}_c1_o0_ATB.itp'.format(self.molname)

#     @property
#     def molfile(self):
#         return '{}_c1_ATB.xyz'.format(self.molname)

#     # @pytest.fixture(scope='function')
#     # def resp(self):
#     #     orientation_options = psiresp.OrientationOptions(n_reorientations=2)
#     #     opt = [mol_from_file('{}_c1_ATB_opt.xyz'.format(self.molname))]
#     #     r = self.cls.from_molecules(opt, orientation_options=orientation_options)
#     #     return r

#     @pytest.fixture()
#     def ref(self):
#         return charges_from_itp_file(self.chargefile)

#     def test_resp_noopt(self, ref):
#         orientation_options = psiresp.OrientationOptions(n_reorientations=2)
#         opt = [mol_from_file('{}_c1_ATB_opt.xyz'.format(self.molname))]
#         charge_options = psiresp.ChargeOptions(equivalent_methyls=True)
#         r = self.cls.from_molecules(opt, optimize_geometry=False,
#                                     orientation_options=orientation_options)
#         charges = r.run(charge_options=charge_options)
#         # no idea which point density ATB uses
#         assert_allclose(charges, ref, rtol=0.05, atol=1e-3)

#     @pytest.mark.slow
#     def test_resp_opt(self, ref):
#         mol = [mol_from_file(self.molfile)]
#         orientation_options = psiresp.OrientationOptions(n_reorientations=2)
#         charge_options = psiresp.ChargeOptions(equivalent_methyls=True)
#         r = self.cls.from_molecules(mol, optimize_geometry=True,
#                                     orientation_options=orientation_options)
#         charges = r.run(charge_options=charge_options)
#         assert_allclose(charges, ref, rtol=0.05, atol=1e-3)


# class TestATBRespMethane(BaseTestATBResp):
#     molname = 'methane'


# # fails, idk why; ATB charges are also different from Malde et al 2011
# @pytest.mark.skip(reason='Fails? ATB charges are also v. different from Malde et al 2011')
# class TestATBRespIsopropanol(BaseTestATBResp):
#     molname = 'isopropanol'


# class TestATBRespDMSO(BaseTestATBResp):
#     molname = 'dmso'


# # @pytest.mark.skip(reason='Fails? Are the mol2 geometries not minimised?')
# @pytest.mark.resp2
# @pytest.mark.slow
# @pytest.mark.parametrize('name', ['C00', 'C64', 'MPE', 'PXY', 'TOL'])
# @pytest.mark.parametrize('delta', [0, 0.6, 1])
# class TestResp2Charges:
#     """Charges from RESP2 paper Schauperl et al., 2020.

#     See dataset for more: https://doi.org/10.5281/zenodo.3593762
#     """

#     def test_resp2_noopt(self, name, delta):
#         fn = '{}_R2_{:d}.mol2'.format(name, int(delta * 100))
#         mols = [mol_from_mol2(fn)]
#         ref = charges_from_mol2(fn)
#         r = psiresp.Resp2.from_molecules(mols, delta=delta, charge=0)
#         charges = r.run()
#         assert_almost_equal(charges, ref, decimal=3)


# class BaseTestResp2Ethanol:
#     """Charges from ethanol example in MSchauperl/resp2.

#     Assert almost equal with decimal=3 because not sure when values are rounded.
#     See repo for more: https://github.com/MSchauperl/RESP2
#     """

#     load_files = False

#     solv = np.array([-0.2416, 0.3544, -0.6898, 0.0649, 0.0649, 0.0649, -0.0111, -0.0111, 0.4045])

#     gas = np.array([-0.2300, 0.3063, -0.5658, 0.0621, 0.0621, 0.0621, -0.0153, -0.0153, 0.3339])

#     ref = np.array([-0.2358, 0.33035, -0.6278, 0.0635, 0.0635, 0.0635, -0.0132, -0.0132, 0.3692])

#     def test_ethanol_no_opt(self, tmpdir):
#         mols = [mol_from_file('ethanol_resp2_opt_c1.xyz'), mol_from_file('ethanol_resp2_opt_c2.xyz')]

#         with tmpdir.as_cwd():
#             if self.load_files:
#                 copy_tree(datafile("test_resp2"), str(tmpdir))
#             io_options = psiresp.IOOptions(load_from_files=self.load_files)
#             r = psiresp.Resp2.from_molecules(mols, charge=0, name='resp2_ethanol', delta=0.5, io_options=io_options)
#             charges = r.run()
#         assert_almost_equal(r.gas_charges, self.gas, decimal=3)
#         assert_almost_equal(r.solvated_charges, self.solv, decimal=3)
#         assert_almost_equal(charges, self.ref, decimal=3)


# @pytest.mark.fast
# class TestLoadResp2Ethanol(BaseTestResp2Ethanol):
#     load_files = True


# @pytest.mark.resp2
# class TestResp2Ethanol(BaseTestResp2Ethanol):
#     @pytest.mark.slow
#     def test_ethanol_opt(self):
#         mols = [mol_from_file('ethanol_resp2_c1.xyz'), mol_from_file('ethanol_resp2_c2.xyz')]
#         r = psiresp.Resp2.from_molecules(mols, charge=0, delta=0.5, optimize_geometry=True)
#         charges = r.run()
#         assert_almost_equal(r.gas_charges, self.gas, decimal=3)
#         assert_almost_equal(r.solv_charges, self.solv, decimal=3)
#         assert_almost_equal(charges, self.ref, decimal=3)


# def test_methanol_1993_paper(tmpdir):
#     # grid and ESP are not generated from molecule
#     # Off by ~0.1 when I generate it myself 🤷‍♀️
#     mol = mol_from_file("methanol_1993.xyz")
#     ref = [-0.6498, 0.4215, 0.1166, 0.0372, 0.0372, 0.0372]
#     io_options = psiresp.IOOptions(load_from_files=True)
#     with tmpdir.as_cwd():
#         copy_tree(datafile("test_resp"), str(tmpdir))
#         r = psiresp.RespA1.from_molecules([mol], charge=0, io_options=io_options, name="methanol_1993")
#         charges = r.run()
#     assert_almost_equal(charges, ref, decimal=4)
