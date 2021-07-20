import pytest
import psiresp

from numpy.testing import assert_almost_equal

from .base import psi4mol_from_xyzfile, charges_from_red_file
from .datafiles import (DMSO_QMRA,
                        DMSO_RESPA2_CHARGES, DMSO_RESPA1_CHARGES, DMSO_ESPA1_CHARGES,
                        DMSO_QMRA_RESPA2_CHARGES, DMSO_QMRA_RESPA1_CHARGES, DMSO_QMRA_ESPA1_CHARGES,
                        ETHANOL_C1, ETHANOL_C2,
                        ETHANOL_RESPA2_CHARGES, ETHANOL_RESPA1_CHARGES, ETHANOL_ESPA1_CHARGES,
                        NME2ALA2_OPT_RESPA2_CHARGES, NME2ALA2_OPT_RESPA1_CHARGES, NME2ALA2_OPT_ESPA1_CHARGES,
                        )

# from .datafiles import DMSO, DMSO_O1, DMSO_O2, DMSO_O3, DMSO_O4


def test_resp_from_file():
    resp = psiresp.Resp.from_molfile(DMSO_QMRA,
                                     conformer_options=dict(n_reorientations=2))
    resp.generate_conformers()
    resp.generate_orientations()
    assert len(resp.conformers) == 1
    assert len(list(resp.orientations)) == 3


class TestNoOrient:

    esp_1 = [-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
             -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]

    resp_1 = [-0.31436216, 0.11376836, 0.14389443, 0.15583112, 0.30951582,
              -0.50568553, -0.33670393, 0.15982115, 0.12029174, 0.153629]

    esp_2 = [-0.39199538, 0.15716631, 0.15716631, 0.15716631, 0.35743529,
             -0.5085439, -0.43701446, 0.16953984, 0.16953984, 0.16953984]

    resp_2 = [-0.25158642, 0.11778735, 0.11778735, 0.11778735, 0.30951582,
              -0.50568553, -0.29298059, 0.12912489, 0.12912489, 0.12912489]

    @pytest.fixture()
    def resp(self, dmso_psi4mol):
        resp = psiresp.Resp(psi4mol=dmso_psi4mol, stage_2=True,
                            conformer_options=dict(optimize_geometry=False))
        resp.generate_conformers()
        resp.generate_orientations()
        return resp

    def test_unrestrained(self, resp):
        resp.restrained = False
        resp.run()

        assert_almost_equal(resp.stage_1_charges.unrestrained_charges,
                            self.esp_1, decimal=5)

        assert_almost_equal(resp.stage_2_charges.unrestrained_charges,
                            self.esp_2, decimal=5)

    def test_restrained(self, resp):
        resp.restrained = True
        resp.run()
        assert_almost_equal(resp.stage_1_charges.restrained_charges,
                            self.resp_1, decimal=5)
        assert_almost_equal(resp.stage_2_charges.restrained_charges,
                            self.resp_2, decimal=5)

    @pytest.fixture()
    def charge_options(self):
        equivalences = [(1, 7), (2, 3, 4, 8, 9, 10)]
        return psiresp.ChargeConstraintOptions(charge_equivalences=equivalences)

    @pytest.fixture()
    def dmso_o2_resp(self, dmso_o1_psi4mol, dmso_o2_psi4mol, charge_options):
        resp = psiresp.Resp(psi4mol=dmso_o1_psi4mol, charge_constraint_options=charge_options)
        resp.generate_conformers()
        conformer = resp.conformers[0]
        conformer.add_orientation(dmso_o1_psi4mol)
        conformer.add_orientation(dmso_o2_psi4mol)
        return resp

    @pytest.fixture()
    def dmso_qmra_resp(self, charge_options):
        psi4mol = psi4mol_from_xyzfile(DMSO_QMRA)
        resp = psiresp.Resp(psi4mol=psi4mol, charge_constraint_options=charge_options)
        resp.generate_conformers()
        resp.generate_orientations()
        return resp

    @pytest.fixture()
    def ethanol_resp(self):
        charge_options = psiresp.ChargeConstraintOptions(symmetric_methyls=True,
                                                         symmetric_methylenes=True)
        # conformer_options = dict(reorientations=[(1, 5, 8), (8, 5, 1),
        #                                          (9, 8, 5), (5, 8, 9)])
        conformer_options = dict(n_reorientations=2)

        eth1 = psi4mol_from_xyzfile(ETHANOL_C1)
        eth2 = psi4mol_from_xyzfile(ETHANOL_C2)
        resp = psiresp.Resp(psi4mol=eth1, charge_constraint_options=charge_options,
                            conformer_options=conformer_options)
        resp.add_conformer(eth1)
        resp.add_conformer(eth2)
        resp.generate_orientations()
        return resp

    @pytest.mark.parametrize("stage_2, hyp_a1, charge_file", [
        (False, 0.01, DMSO_RESPA2_CHARGES),
        (True, 0.0005, DMSO_RESPA1_CHARGES),
        (False, 0.0, DMSO_ESPA1_CHARGES),
    ])
    def test_dmso_multiple_conformers(self, dmso_o2_resp, stage_2, hyp_a1, charge_file):
        dmso_o2_resp.stage_2 = stage_2
        dmso_o2_resp.hyp_a1 = hyp_a1
        charges = dmso_o2_resp.run()
        reference = charges_from_red_file(charge_file)
        assert_almost_equal(charges, reference, decimal=3)

    @pytest.mark.parametrize("stage_2, hyp_a1, charge_file", [
        (False, 0.01, DMSO_QMRA_RESPA2_CHARGES),
        (True, 0.0005, DMSO_QMRA_RESPA1_CHARGES),
        (False, 0.0, DMSO_QMRA_ESPA1_CHARGES),
    ])
    def test_dmso_single_conformer(self, dmso_qmra_resp, stage_2, hyp_a1, charge_file):
        dmso_qmra_resp.stage_2 = stage_2
        dmso_qmra_resp.hyp_a1 = hyp_a1
        charges = dmso_qmra_resp.run()
        reference = charges_from_red_file(charge_file)
        assert_almost_equal(charges, reference, decimal=3)

    @pytest.mark.parametrize("stage_2, hyp_a1, charge_file", [
        (False, 0.01, ETHANOL_RESPA2_CHARGES),
        (True, 0.0005, ETHANOL_RESPA1_CHARGES),
        (False, 0.0, ETHANOL_ESPA1_CHARGES),
    ])
    def test_ethanol_symmetric_hs(self, ethanol_resp, stage_2, hyp_a1, charge_file):
        ethanol_resp.stage_2 = stage_2
        ethanol_resp.hyp_a1 = hyp_a1
        charges = ethanol_resp.run()
        reference = charges_from_red_file(charge_file)
        assert_almost_equal(charges, reference, decimal=2)

    @pytest.mark.parametrize("stage_2, hyp_a1, charge_file", [
        (False, 0.01, NME2ALA2_OPT_RESPA2_CHARGES),
        (True, 0.0005, NME2ALA2_OPT_RESPA1_CHARGES),
        (False, 0.0, NME2ALA2_OPT_ESPA1_CHARGES),
    ])
    def test_fit_single_constraints(self, nme2ala2_opt_resp, stage_2, hyp_a1, charge_file):
        chrconstr = [(0, [1, 2, 3, 4, 5, 6]),
                     (0, [20, 21, 22, 23, 24, 25]),
                     (-0.4157, [7]), (0.2719, [8]),
                     (0.5973, [18]), (-0.5679, [19])]
        chrequiv = [[10, 14], [11, 12, 13, 15, 16, 17]]
        options = psiresp.ChargeConstraintOptions(charge_constraints=chrconstr,
                                                  charge_equivalences=chrequiv,
                                                  symmetric_methyls=False,
                                                  symmetric_methylenes=False)
        nme2ala2_opt_resp.charge_constraint_options = options
        nme2ala2_opt_resp.stage_2 = stage_2
        nme2ala2_opt_resp.hyp_a1 = hyp_a1
        charges = nme2ala2_opt_resp.run()
        reference = charges_from_red_file(charge_file)
        assert_almost_equal(charges, reference, decimal=3)


# class TestRespNoOptimization:

#     @pytest.fixture()
#     def dmso_resp(self, dmso_psi4mol):
#         orientation_options = dict(reorientations=[(1, 5, 6), (6, 5, 1)])
#         conformer_options = dict(orientation_options=orientation_options)
#         resp = psiresp.Resp(psi4mol=dmso_psi4mol,
#                             conformer_options=conformer_options)
