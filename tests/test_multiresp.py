"""
Unit and regression test for the psiresp package.
"""

# Import package, test suite, and other packages as needed
import psiresp
import pytest
import sys
import numpy as np

from numpy.testing import assert_almost_equal, assert_allclose
from .utils import mol_from_file, charges_from_red_file, datafile


class TestMultiRespNoOptNoOrient(object):
    opt = False
    load_files = False
    nme2ala2_names = ['nme2ala2_opt_c1.xyz', 'nme2ala2_opt_c2.xyz']
    methylammonium_names = ['methylammonium_opt_c1.xyz']
    n_orient = 0
    orient = [[(1, 5, 7), (7, 5, 1)],
              [(5, 18, 19), (19, 18, 5), (6, 19, 20), (20, 19, 6)]]

    intra_chrconstr = [[], {0: [20, 21, 22, 23, 24, 25],
                            0.6163: [18],
                            -0.5722: [19]}]
    intra_chrequiv = [[[6, 7, 8]], [[10, 14], [11, 12, 13, 15, 16, 17]]]
    inter_chrconstr = {0: [(1, [1, 2, 3, 4]), (2, [1, 2, 3, 4, 5, 6, 7, 8])]}

    # GRID = datafile('test_multiresp/grid.dat')
    # ESP = datafile('test_multiresp/grid_esp.dat')
    rtol = 0.01
    atol = 1e-4

    @pytest.fixture()
    def nme2ala2(self, tmpdir):
        mols = [mol_from_file(f) for f in self.nme2ala2_names]
        orient = [(5, 18, 19), (19, 18, 5), (6, 19, 20), (20, 19, 6)]
        or_options = psiresp.OrientationOptions(reorientations=orient,
                                                keep_original=False)
        chrconstrs = {0: [20, 21, 22, 23, 24, 25],
                            0.6163: [18],
                            -0.5722: [19]}
        chrequivs = [[10, 14], [11, 12, 13, 15, 16, 17]]
        ch_options = psiresp.ChargeOptions(charge_equivalences=chrequivs,
                                           charge_constraints=chrconstrs,
                                           equivalent_sp3_hydrogens=False)
        with tmpdir.as_cwd():
            resp = psiresp.Resp.from_molecules(mols, charge=0,
                                                orientation_options=or_options,
                                                charge_constraint_options=ch_options,
                                                optimize_geometry=self.opt,
                                                name='nme2ala2')
        return resp

    @pytest.fixture()
    def methylammonium(self, tmpdir):
        mols = [mol_from_file(f) for f in self.methylammonium_names]
        orient = [(1, 5, 7), (7, 5, 1)]
        or_options = psiresp.OrientationOptions(reorientations=orient,
                                                keep_original=False)
        chrconstrs = []
        chrequivs = [[6, 7, 8]]
        ch_options = psiresp.ChargeOptions(charge_equivalences=chrequivs,
                                           charge_constraints=chrconstrs,
                                           equivalent_sp3_hydrogens=False)
        with tmpdir.as_cwd():
            resp = psiresp.Resp.from_molecules(mols, charge=1, multiplicity=1,
                                            orientation_options=or_options,
                                            charge_constraint_options=ch_options,
                                            optimize_geometry=self.opt,
                                            name='methylammonium')
            print(resp.conformers[0].orientations[0].coordinates)
        return resp

    @pytest.fixture()
    def nme2ala2_charges(self, redname):
        fn = 'nme2ala2_multifit_constr_c2_o4_{}.dat'.format(redname)
        return charges_from_red_file(fn)

    @pytest.fixture()
    def multifit_charges(self, redname):
        fn = 'amm_dimethyl_{}.dat'.format(redname)
        return charges_from_red_file(fn)

    # @pytest.mark.parametrize('stage_2,a,redname', [
    #     (False, 0.01, 'respA2'),  # really off
    #     (True, 0.0005, 'respA1'),
    #     (False, 0.0, 'espA1')
    # ])
    # def test_single_mol_in_multiresp(self, stage_2, a, nme2ala2,
    #                                  nme2ala2_charges, tmpdir):
    #     r = psiresp.MultiResp([nme2ala2])
    #     ch_options = psiresp.options.ChargeOptions(equivalent_sp3_hydrogens=False)
    #     charges = r.run(stage_2=stage_2, hyp_a1=a, charge_constraint_options=ch_options)
    #     assert_allclose(charges[0], nme2ala2_charges, rtol=self.rtol, 
    #                     atol=self.atol)

    @pytest.mark.parametrize('stage_2,a,redname', [
        (False, 0.01, 'respA2'),  # really off
        (True, 0.0005, 'respA1'),
        (False, 0.0, 'espA1')
    ])
    def test_multi_mol(self, stage_2, a, nme2ala2, methylammonium,
                       multifit_charges):
        r = psiresp.MultiResp([methylammonium, nme2ala2])
        chrequivs = [[(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8)]]
        ch_options = psiresp.ChargeOptions(charge_equivalences=chrequivs, equivalent_sp3_hydrogens=False)
        charges = r.run(stage_2=stage_2, hyp_a1=a, charge_constraint_options=ch_options)
        for charge, ref in zip(charges, multifit_charges):
            assert_allclose(charge, ref, rtol=self.rtol, atol=self.atol)



@pytest.mark.skip(reason='fails even with these looser comparison constraints. orientations are important!')
class TestMultiRespNoOptAutoOrient(TestMultiRespNoOptNoOrient):
    n_orient = 8
    orient = [[], []]
    rtol = 0.15  # will have different orientations
    atol = 1e-2


@pytest.mark.optimize
@pytest.mark.slow
class TestMultiRespOptNoOrient(TestMultiRespNoOptNoOrient):
    opt = True
    nme2ala2_names = ['nme2ala2_c1.xyz', 'nme2ala2_c2.xyz']
    methylammonium_names = ['methylammonium_c1.xyz']
