from multiprocessing.sharedctypes import Value
import pytest
import numpy as np
from numpy.testing import assert_allclose

import qcelemental as qcel
import psiresp

from psiresp.tests.datafiles import (AMM_NME_OPT_ESPA1_CHARGES,
                                     AMM_NME_OPT_RESPA2_CHARGES,
                                     AMM_NME_OPT_RESPA1_CHARGES,
                                     DMSO_ESPA1_CHARGES,
                                     DMSO_RESPA1_CHARGES,
                                     DMSO_RESPA2_CHARGES,
                                     DMSO_O1, DMSO_O2,
                                     ETHANOL_RESP2_C1, ETHANOL_RESP2_C2,
                                     )


@pytest.mark.parametrize("config_class, red_charges", [
    (psiresp.configs.ESP, DMSO_ESPA1_CHARGES),
    (psiresp.configs.TwoStageRESP, DMSO_RESPA1_CHARGES),
    (psiresp.configs.OneStageRESP, DMSO_RESPA2_CHARGES),

], indirect=['red_charges'])
def test_config_resp(config_class, red_charges, fractal_client, dmso):
    pytest.importorskip("psi4")

    qcdmso = qcel.models.Molecule.from_file(DMSO_O1, fix_com=True,
                                            fix_orientation=True)
    qcdmso2 = qcel.models.Molecule.from_file(DMSO_O2, fix_com=True,
                                             fix_orientation=True)
    dmso = psiresp.Molecule(qcmol=qcdmso, optimize_geometry=False,
                            keep_original_orientation=True)

    constraints = psiresp.ChargeConstraintOptions(symmetric_methylenes=False,
                                                  symmetric_methyls=False)
    indices = [[0, 6],
               [1, 2, 3, 7, 8, 9]]
    for ix in indices:
        constraints.add_charge_equivalence_constraint_for_molecule(dmso,
                                                                   indices=ix)
    job = config_class(molecules=[dmso],
                       charge_constraints=constraints,
                       )
    assert isinstance(job, config_class)

    job.generate_conformers()
    dmso_c1 = job.molecules[0].conformers[0]
    dmso_c1.add_orientation_with_coordinates(qcdmso.geometry,
                                             units="bohr")
    dmso_c1.add_orientation_with_coordinates(qcdmso2.geometry,
                                             units="bohr")
    assert len(job.molecules[0].conformers) == 1
    assert len(job.molecules[0].conformers[0].orientations) == 2

    job.compute_orientation_energies(client=fractal_client)
    job.compute_esps()
    job.compute_charges()
    assert_allclose(job.charges, red_charges, atol=1e-3)


@pytest.mark.parametrize("config_class, red_charges", [
    (psiresp.configs.ESP, AMM_NME_OPT_ESPA1_CHARGES),
    (psiresp.configs.OneStageRESP, AMM_NME_OPT_RESPA2_CHARGES),
    (psiresp.configs.TwoStageRESP, AMM_NME_OPT_RESPA1_CHARGES),
], indirect=['red_charges'])
def test_config_multiresp(nme2ala2, methylammonium,
                          methylammonium_nme2ala2_charge_constraints,
                          config_class, red_charges,
                          job_esps, job_grids):
    job = config_class(molecules=[methylammonium, nme2ala2],
                       charge_constraints=methylammonium_nme2ala2_charge_constraints)
    assert isinstance(job, config_class)

    for orient in job.iter_orientations():
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


def test_resp2(fractal_client):
    pytest.importorskip("psi4")

    # generate molecule
    c1 = qcel.models.Molecule.from_file(ETHANOL_RESP2_C1)
    c2 = qcel.models.Molecule.from_file(ETHANOL_RESP2_C2)
    mol = psiresp.Molecule(qcmol=c1, optimize_geometry=False,
                           keep_original_orientation=True)
    mol.generate_conformers()
    mol.add_conformer_with_coordinates(c2.geometry, units="bohr")

    assert mol.n_conformers == 2
    assert mol.n_orientations == 0

    job = psiresp.RESP2(molecules=[mol])
    job.run(client=fractal_client)

    assert job.vacuum.n_conformers == 2
    assert job.vacuum.n_orientations == 2
    assert job.solvated.n_conformers == 2
    assert job.solvated.n_orientations == 2

    ETOH_SOLV_CHARGES = np.array([-0.2416,  0.3544, -0.6898,  0.0649,  0.0649,
                                  0.0649, -0.0111, -0.0111,  0.4045])

    ETOH_GAS_CHARGES = np.array([-0.2300,  0.3063, -0.5658,  0.0621,  0.0621,
                                0.0621, -0.0153, -0.0153,  0.3339])
    ETOH_REF_CHARGES = np.array([-0.2358,  0.33035, -0.6278,  0.0635,
                                0.0635,  0.0635, -0.0132, -0.0132,  0.3692])

    assert_allclose(job.solvated.charges[0], ETOH_SOLV_CHARGES, atol=5e-03)
    assert_allclose(job.vacuum.charges[0], ETOH_GAS_CHARGES, atol=5e-03)
    assert_allclose(job.get_charges(delta=0.5)[0], ETOH_REF_CHARGES, atol=5e-03)
