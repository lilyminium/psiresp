import pytest
import pathlib
import glob
import shutil
import random

import numpy as np
from numpy.testing import assert_allclose


import psiresp
from psiresp.job import Job
from psiresp.resp import RespOptions


from psiresp.tests.datafiles import (AMM_NME_OPT_ESPA1_CHARGES,
                                     AMM_NME_OPT_RESPA2_CHARGES,
                                     AMM_NME_OPT_RESPA1_CHARGES,
                                     MANUAL_JOBS_WKDIR,
                                     )

pytest.importorskip("psi4")


class TestSingleResp:
    def test_unrestrained(self, dmso, fractal_client):
        options = RespOptions(stage_2=True, restrained_fit=False)
        job = Job(molecules=[dmso],
                  resp_options=options)
        job.run(client=fractal_client)

        esp_1 = [[-0.43877469, 0.14814998, 0.17996033, 0.18716814, 0.35743529,
                 -0.5085439, -0.46067469, 0.19091725, 0.15500465, 0.18935764]]
        esp_2 = [[-0.39199538, 0.15716631, 0.15716631, 0.15716631, 0.35743529,
                 -0.5085439, -0.43701446, 0.16953984, 0.16953984, 0.16953984]]
        assert_allclose(job.stage_1_charges.unrestrained_charges, esp_1, atol=1e-7)
        assert_allclose(job.stage_2_charges.unrestrained_charges, esp_2, atol=1e-7)

        chgrepr = """<RespCharges(restraint_height=0.0005, restraint_slope=0.1, restrained_fit=False, exclude_hydrogens=True) with 0 charge constraints; unrestrained_charges=[array([-0.43877,  0.14815,  0.17996,  0.18717,  0.35744, -0.50854,
       -0.46067,  0.19092,  0.155  ,  0.18936])], restrained_charges=None>"""
        assert repr(job.stage_1_charges) == chgrepr

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
    @pytest.mark.parametrize("stage_2, restraint_height, red_charges", [
        (False, 0.0, AMM_NME_OPT_ESPA1_CHARGES),
        (False, 0.01, AMM_NME_OPT_RESPA2_CHARGES),
        (True, 0.0005, AMM_NME_OPT_RESPA1_CHARGES),
    ], indirect=['red_charges'])
    def test_given_esps(self, nme2ala2, methylammonium,
                        methylammonium_nme2ala2_charge_constraints,
                        stage_2, restraint_height, red_charges, job_esps, job_grids):

        resp_options = RespOptions(stage_2=stage_2, restraint_height_stage_1=restraint_height)
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
    @pytest.mark.parametrize("stage_2, restraint_height, red_charges", [
        (False, 0.0, AMM_NME_OPT_ESPA1_CHARGES),
        (False, 0.01, AMM_NME_OPT_RESPA2_CHARGES),
        (True, 0.0005, AMM_NME_OPT_RESPA1_CHARGES),
    ], indirect=['red_charges'])
    def test_calculated_esps(self, nme2ala2_empty, methylammonium_empty,
                             methylammonium_nme2ala2_charge_constraints,
                             stage_2, restraint_height, red_charges,
                             fractal_client, job_esps, job_grids):

        resp_options = RespOptions(stage_2=stage_2, restraint_height_stage_1=restraint_height)
        job = Job(molecules=[methylammonium_empty, nme2ala2_empty],
                  charge_constraints=methylammonium_nme2ala2_charge_constraints,
                  resp_options=resp_options)
        job.compute_orientation_energies(client=fractal_client)
        job.compute_esps()
        for mol in job.molecules:
            for conf in mol.conformers:
                for orient in conf.orientations:
                    fname = orient.qcmol.get_hash()
                    assert_allclose(orient.grid, job_grids[fname], atol=1e-7)
                    assert_allclose(orient.esp, job_esps[fname], atol=1e-7)

        job.compute_charges()
        charges = np.concatenate(job.charges)

        assert_allclose(charges[[0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15]].sum(), 0, atol=1e-7)
        assert_allclose(charges[[27, 28, 29, 30, 31, 32]].sum(), 0, atol=1e-7)
        assert_allclose(charges[25], 0.6163)
        assert_allclose(charges[26], -0.5722)
        assert_allclose(charges[18], charges[22])
        for calculated, reference in zip(job.charges[::-1], red_charges[::-1]):
            assert_allclose(calculated, reference, atol=1e-3)

    @pytest.mark.slow
    def test_run_with_empty(self, empty_client):
        conformer_options = psiresp.ConformerGenerationOptions(
            n_max_conformers=2,
            keep_original_conformer=False,
            rms_tolerance=0.01,
        )
        nme2ala2 = psiresp.Molecule.from_smiles("CC(=O)NC(C)(C)C(NC)=O",
                                                optimize_geometry=False,
                                                conformer_generation_options=conformer_options)
        assert nme2ala2._rdmol.GetNumAtoms() == 25
        methylammonium = psiresp.Molecule.from_smiles("C[NH3+]",
                                                      optimize_geometry=False,
                                                      conformer_generation_options=conformer_options)
        assert methylammonium._rdmol.GetNumAtoms() == 8

        # add constraints
        # nme to 0
        constraints = psiresp.ChargeConstraintOptions()
        nme_smiles = "CC(=O)NC(C)(C)C([N:1]([H:2])[C:3]([H:4])([H:5])([H:6]))=O"
        nme_indices = nme2ala2.get_smarts_matches(nme_smiles)
        assert len(nme_indices) == 1
        assert len(nme_indices[0]) == 6
        constraints.add_charge_sum_constraint_for_molecule(nme2ala2,
                                                           charge=0,
                                                           indices=nme_indices[0])
        # inter-aa to 0
        methyl_atoms = methylammonium.get_atoms_from_smarts("C([H])([H])([H])")
        ace_atoms = nme2ala2.get_atoms_from_smarts("C([H])([H])([H])C(=O)N([H])")
        constraint_atoms = methyl_atoms[0] + ace_atoms[0]
        assert len(constraint_atoms) == 12
        constraints.add_charge_sum_constraint(charge=0, atoms=constraint_atoms)

        # constrain particular atoms
        co_atoms = nme2ala2.get_atoms_from_smarts("CC(=O)NC(C)(C)[C:1]=[O:2]")
        assert len(co_atoms) == 1
        assert len(co_atoms[0]) == 2
        constraints.add_charge_sum_constraint(charge=0.6163, atoms=co_atoms[0][:1])
        constraints.add_charge_sum_constraint(charge=-0.5722, atoms=co_atoms[0][1:])

        # equivalent hs
        h_smiles = "C(C([H:2])([H:2])([H:2]))(C([H:2])([H:2])([H:2]))"
        h_atoms = nme2ala2.get_atoms_from_smarts(h_smiles)[0]
        assert len(h_atoms) == 6
        constraints.add_charge_equivalence_constraint(atoms=h_atoms)
        c_smiles = "C([C:1]([H])([H])([H]))([C:1]([H])([H])([H]))"
        c_atoms = nme2ala2.get_atoms_from_smarts(c_smiles)[0]
        assert len(c_atoms) == 2
        constraints.add_charge_equivalence_constraint(atoms=c_atoms)
        h_smiles = "[N+]([H:1])([H:1])([H:1])"
        h_atoms = methylammonium.get_atoms_from_smarts(h_smiles)[0]
        assert len(h_atoms) == 3
        constraints.add_charge_equivalence_constraint(atoms=h_atoms)

        geometry_options = psiresp.QMGeometryOptimizationOptions(
            method="b3lyp", basis="sto-3g",
        )
        esp_options = psiresp.QMEnergyOptions(
            method="b3lyp", basis="sto-3g",
        )

        job_multi = psiresp.Job(molecules=[methylammonium, nme2ala2],
                                charge_constraints=constraints,
                                qm_optimization_options=geometry_options,
                                qm_esp_options=esp_options,)
        job_multi.run(client=empty_client)

        nme_charges = job_multi.molecules[1].stage_2_restrained_charges
        assert_allclose(nme_charges[list(nme_indices[0])].sum(), 0, atol=1e-7)

        methylammonium_charges = [-0.433807,  0.033999,  0.19933,  0.19933,
                                  0.19933,  0.267273, 0.267273,  0.267273]
        nme2ala2_charges = [-0.34123, 0.38045, -0.31095, -0.44075, 0.27644,
                            -0.30134, -0.30134, 0.6163, -0.20008, -0.37819,
                            -0.5722, 0.09333, 0.09333, 0.09333, 0.2683,
                            0.07438, 0.07438, 0.07438, 0.07438, 0.07438,
                            0.07438, 0.13245, 0.14861, 0.14861, 0.14861]

        # low precision -- generation of conformers can be flaky
        assert_allclose(job_multi.charges[0],
                        methylammonium_charges, atol=5e-2)
        assert_allclose(job_multi.charges[1],
                        nme2ala2_charges,
                        atol=5e-2)

    def test_run_manual(self, nme2ala2_empty, methylammonium_empty, tmpdir):
        pytest.importorskip("rdkit")

        nme2ala2_empty.optimize_geometry = True
        methylammonium_empty.optimize_geometry = True
        assert len(nme2ala2_empty.conformers) == 2
        assert len(methylammonium_empty.conformers) == 1
        job = Job(molecules=[methylammonium_empty, nme2ala2_empty])

        data_wkdir = pathlib.Path(MANUAL_JOBS_WKDIR)

        with tmpdir.as_cwd():

            # OPTIMIZATION
            assert not job.working_directory.exists()
            with pytest.raises(SystemExit, match="Exiting to allow running"):
                job.run()

            assert job.working_directory.exists()

            # check run file contents
            runfile = job.qm_optimization_options.get_run_file(job.working_directory)
            assert str(runfile) == "psiresp_working_directory/optimization/run_optimization.sh"
            with runfile.open() as f:
                optlines = [x.strip() for x in f.readlines()]
            assert len(optlines) == 4

            # check existence and copy completed files in
            opt_filenames = glob.glob(str(data_wkdir / "optimization/*"))
            assert len(opt_filenames) == 4
            for file in opt_filenames:
                path = pathlib.Path(file.split("manual_jobs/")[1])
                assert path.exists()

                if file.endswith("msgpack"):
                    assert f"psi4 --qcschema {path.name}" in optlines
                shutil.copyfile(file, path)

            assert all(not conf.is_optimized for conf in job.iter_conformers())

            # SINGLE POINT
            assert not job.qm_esp_options.get_working_directory(job.working_directory).exists()
            with pytest.raises(SystemExit, match="Exiting to allow running"):
                job.run()

            assert all(conf.is_optimized for conf in job.iter_conformers())

            # check run file contents
            runfile = job.qm_esp_options.get_run_file(job.working_directory)
            assert str(runfile) == "psiresp_working_directory/single_point/run_single_point.sh"
            with runfile.open() as f:
                splines = [x.strip() for x in f.readlines()]
            assert len(splines) == 11

            # check existence and copy completed files in
            opt_filenames = glob.glob(str(data_wkdir / "single_point/*"))
            assert len(opt_filenames) == 11
            for file in opt_filenames:
                path = pathlib.Path(file.split("manual_jobs/")[1])
                assert path.exists()

                if file.endswith("msgpack"):
                    assert f"psi4 --qcschema {path.name}" in splines
                shutil.copyfile(file, path)

            job.run()
            methylammonium_charges = [-0.281849,  0.174016,  0.174016,  0.174016,
                                      -0.528287,  0.215501, 0.857086,  0.215501]
            nme2ala2_charges = [4.653798, -1.174226, -1.1742263, -1.1742263, -1.2224316,
                                0.1732717, -3.9707787, 0.8309516, 8.291471, 4.749616,
                                -1.870866, -1.870866, -1.870866, 2.275405, -1.3199896,
                                -1.3199896, -1.3199896, 1.6195477, -1.5787892, -14.898019,
                                2.9755072, 30.303230, -7.035844, -7.035844, -7.035844]

            assert_allclose(job.charges[0], methylammonium_charges, atol=1e-6)
            assert_allclose(job.charges[1], nme2ala2_charges, atol=1e-6)
