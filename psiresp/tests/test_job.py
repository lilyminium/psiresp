import pytest

import numpy as np
from numpy.testing import assert_allclose


import psiresp
from psiresp.job import Job
from psiresp.resp import RespOptions


from psiresp.tests.datafiles import (AMM_NME_OPT_ESPA1_CHARGES,
                                     AMM_NME_OPT_RESPA2_CHARGES,
                                     AMM_NME_OPT_RESPA1_CHARGES,
                                     )


class TestSingleResp:
    # @pytest.mark.slow
    def test_unrestrained(self, dmso, fractal_client):
        options = RespOptions(stage_2=True, restrained_fit=False)
        job = Job(molecules=[dmso],
                  resp_options=options)
        job.run(client=fractal_client)

        esp_1 = [[-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
                 -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]]
        esp_2 = [[-0.39199538, 0.15716631, 0.15716631, 0.15716631, 0.35743529,
                 -0.5085439, -0.43701446, 0.16953984, 0.16953984, 0.16953984]]
        assert_allclose(job.stage_1_charges.unrestrained_charges, esp_1)
        assert_allclose(job.stage_2_charges.unrestrained_charges, esp_2)

    # @pytest.mark.slow
    def test_restrained(self, dmso, fractal_client):
        options = RespOptions(stage_2=True, restrained_fit=True)
        job = Job(molecules=[dmso],
                  resp_options=options)
        job.run(client=fractal_client)

        resp_1 = [[-0.31436216, 0.11376836, 0.14389443, 0.15583112, 0.30951582,
                  -0.50568553, -0.33670393, 0.15982115, 0.12029174, 0.153629]]
        resp_2 = [[-0.25158642, 0.11778735, 0.11778735, 0.11778735, 0.30951582,
                  -0.50568553, -0.29298059, 0.12912489, 0.12912489, 0.12912489]]
        assert_allclose(job.stage_1_charges.restrained_charges, resp_1, atol=1e-5)
        assert_allclose(job.stage_2_charges.restrained_charges, resp_2, atol=1e-5)


class TestMultiRespFast:
    @pytest.mark.parametrize("stage_2, resp_a, red_charges", [
        (False, 0.0, AMM_NME_OPT_ESPA1_CHARGES),
        (False, 0.01, AMM_NME_OPT_RESPA2_CHARGES),
        (True, 0.0005, AMM_NME_OPT_RESPA1_CHARGES),
    ], indirect=['red_charges'])
    def test_given_esps(self, nme2ala2, methylammonium,
                        methylammonium_nme2ala2_charge_constraints,
                        stage_2, resp_a, red_charges,
                        fractal_client, job_esps, job_grids):

        resp_options = RespOptions(stage_2=stage_2, resp_a1=resp_a)
        job = Job(molecules=[methylammonium, nme2ala2],
                  charge_constraints=methylammonium_nme2ala2_charge_constraints,
                  resp_options=resp_options)
        for mol in job.molecules:
            for conf in mol.conformers:
                for orient in conf.orientations:
                    fname = orient.qcmol.get_hash()
                    orient.esp = job_esps[fname]
                    orient.grid = job_grids[fname]
        job.compute_charges()
        charges = np.concatenate(job.charges)

        assert_allclose(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0, atol=1e-7)
        assert_allclose(charges[[27, 28, 29, 30, 31, 32]].sum(), 0, atol=1e-7)
        assert_allclose(charges[25], 0.6163)
        assert_allclose(charges[26], -0.5722)
        assert_allclose(charges[18], charges[22])
        for calculated, reference in zip(job.charges[::-1], red_charges[::-1]):
            assert_allclose(calculated, reference, atol=1e-3)

    # @pytest.mark.slow
    @pytest.mark.parametrize("stage_2, resp_a, red_charges", [
        (False, 0.0, AMM_NME_OPT_ESPA1_CHARGES),
        (False, 0.01, AMM_NME_OPT_RESPA2_CHARGES),
        (True, 0.0005, AMM_NME_OPT_RESPA1_CHARGES),
    ], indirect=['red_charges'])
    def test_calculated_esps(self, nme2ala2, methylammonium,
                             methylammonium_nme2ala2_charge_constraints,
                             stage_2, resp_a, red_charges,
                             fractal_client, job_esps, job_grids):

        resp_options = RespOptions(stage_2=stage_2, resp_a1=resp_a)
        job = Job(molecules=[methylammonium, nme2ala2],
                  charge_constraints=methylammonium_nme2ala2_charge_constraints,
                  resp_options=resp_options)
        job.compute_orientation_energies(client=fractal_client)
        job.compute_esps()
        for mol in job.molecules:
            for conf in mol.conformers:
                for orient in conf.orientations:
                    fname = orient.qcmol.get_hash()
                    assert_allclose(orient.grid, job_grids[fname])
                    assert_allclose(orient.esp, job_esps[fname])

        job.compute_charges()
        charges = np.concatenate(job.charges)

        assert_allclose(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0, atol=1e-7)
        assert_allclose(charges[[27, 28, 29, 30, 31, 32]].sum(), 0, atol=1e-7)
        assert_allclose(charges[25], 0.6163)
        assert_allclose(charges[26], -0.5722)
        assert_allclose(charges[18], charges[22])
        for calculated, reference in zip(job.charges[::-1], red_charges[::-1]):
            assert_allclose(calculated, reference, atol=1e-3)

    def test_run_with_empty(self, empty_client):
        nme2ala2 = psiresp.Molecule.from_smiles("CC(=O)NC(C)(C)C(NC)=O", optimize_geometry=False)
        assert nme2ala2._rdmol is not None
        methylammonium = psiresp.Molecule.from_smiles("C[NH3+]", optimize_geometry=False)
        assert methylammonium._rdmol is not None
        constraints = psiresp.ChargeConstraintOptions()
        nme_smiles = "CC(=O)NC(C)(C)C([N:1]([H:2])[C:3]([H:4])([H:5])([H:6]))=O"
        nme_indices = nme2ala2.get_smarts_matches(nme_smiles)
        print(nme_indices)
        constraints.add_charge_sum_constraint_for_molecule(nme2ala2,
                                                           charge=0,
                                                           indices=nme_indices[0])
        methyl_atoms = methylammonium.get_atoms_from_smarts("C([H])([H])([H])")
        ace_atoms = nme2ala2.get_atoms_from_smarts("C([H])([H])([H])C(=O)N([H])")
        constraint_atoms = methyl_atoms[0] + ace_atoms[0]
        constraints.add_charge_sum_constraint(charge=0, atoms=constraint_atoms)

        h_smiles = "C(C([H:2])([H:2])([H:2]))(C([H:2])([H:2])([H:2]))"
        h_atoms = nme2ala2.get_atoms_from_smarts(h_smiles)[0]
        print(len(h_atoms))
        constraints.add_charge_equivalence_constraint(atoms=h_atoms)

        geometry_options = psiresp.QMGeometryOptimizationOptions(
            method="b3lyp", basis="sto-3g")
        esp_options = psiresp.QMEnergyOptions(
            method="b3lyp", basis="sto-3g",
        )

        job_multi = psiresp.Job(molecules=[methylammonium, nme2ala2],
                                charge_constraints=constraints,
                                qm_optimization_options=geometry_options,
                                qm_esp_options=esp_options,)
        job_multi.run(client=empty_client)

        for mol in job_multi.molecules:
            for conf in mol.conformers:
                for orient in conf.orientations:
                    assert orient._rdmol is not None
        # print(job_multi.charges[0])
        # print(job_multi.molecules[0].to_smiles())
        # print(job_multi.charges[1])
        # print(job_multi.molecules[1].to_smiles())
        # assert False
