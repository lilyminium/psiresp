import time
from typing import Optional, List, Tuple
import multiprocessing
import functools
import itertools

import tqdm
from pydantic import Field, validator, root_validator
import numpy as np
import scipy.linalg

from . import base, psi4utils, orutils, molecule, charge, qm, grid, resp
from psiresp.charge import MoleculeChargeConstraints
from psiresp.resp import RespCharges
from psiresp.orientation import Orientation
from psiresp.constraint import ConstraintMatrix


class Job(base.Model):

    molecules: List[molecule.Molecule] = []
    qm_optimization_options: qm.QMGeometryOptimizationOptions = Field(
        default=qm.QMGeometryOptimizationOptions(),
        description="QM options for geometry optimization"
    )
    qm_esp_options: qm.QMEnergyOptions = Field(
        default=qm.QMEnergyOptions(),
        description="QM options for ESP computation"
    )
    grid_options: grid.GridOptions = Field(
        default=grid.GridOptions(),
        description="Options for generating grid for ESP computation"
    )
    resp_options: resp.RespOptions = Field(
        default=resp.RespOptions(),
        description="Options for fitting ESP for charges"
    )
    charge_constraints: charge.ChargeConstraintOptions = Field(
        default=charge.ChargeConstraintOptions(),
        description="Charge constraints"
    )

    verbose: bool = False
    ignore_errors: bool = False
    temperature: float = 298.15

    stage_1_charges: Optional[RespCharges] = None
    stage_2_charges: Optional[RespCharges] = None

    n_processes: Optional[int] = None

    def generate_conformers(self):
        for mol in self.molecules:
            mol.generate_conformers()

    def generate_orientations(self):
        for mol in self.molecules:
            clear = not mol.keep_original_orientation
            mol.generate_orientations(clear_existing_orientations=clear)

    def optimize_geometries(self, client):
        conformers = [conformer.qcmol
                      for mol in self.molecules
                      for conformer in mol.conformers
                      if mol.optimize_geometry and not conformer.is_optimized]

        records = self.qm_optimization_options.add_compute_and_wait(client,
                                                                    conformers)

        for conf, rec in zip(conformers, records):
            conf.molecule.geometry = rec.return_result

    def compute_esps(self, client):
        orientations = [orientation
                        for mol in self.molecules
                        for conformer in mol.conformers
                        for orientation in conformer.orientations
                        if orientation._orientation_esp is None]
        qcmols = [o.qcmol for o in orientations]

        records = self.qm_esp_options.add_compute_and_wait(client,
                                                           qcmols)

        # create functions for multiprocessing mapping
        computer = self._try_compute_esp if self.ignore_errors else self._compute_esp

        # compute esp with possible verbosity
        with multiprocessing.Pool(processes=self.n_processes) as pool:
            results = tqdm.tqdm(
                pool.starmap(computer, zip(orientations, records)),
                disable=not self.verbose,
            )

        # raise errors if any occurred
        errors = [r for r in results if not isinstance(r, Orientation)]
        if errors:
            raise ValueError(*errors)

        for orientation, result in zip(orientations, results):
            orientation._orientation_esp = result._orientation_esp

    def _compute_esp(self, orientation, record):
        orientation.compute_esp(record, grid_options=self.grid_options)
        assert orientation._orientation_esp is not None
        return orientation

    def _try_compute_esp(self, orientation, record):
        try:
            return self._compute_esp(orientation, record)
        except BaseException as e:
            return str(e)

    def run(self, client, update_molecules: bool = True):
        self.generate_conformers()
        self.optimize_geometries(client=client)
        self.generate_orientations()
        self.compute_esps(client=client)
        self.compute_charges(update_molecules=update_molecules)

    def construct_molecule_constraint_matrix(self):
        matrices = [
            ConstraintMatrix.from_orientations(
                orientations=[o for conf in mol.conformers for o in conf.orientations],
                temperature=self.temperature,
            )
            for mol in self.molecules
        ]
        a_mol = scipy.linalg.block_diag(*[mat.a for mat in matrices])
        a_row = scipy.linalg.block_diag(*[np.ones(mat.b.shape[0]) for mat in matrices])
        a_zeros = np.zeros((a_row.shape[0], a_row.shape[0]))
        a_block = np.bmat([[a_mol, a_row.T], [a_row, a_zeros]])

        b_block = np.concatenate([mat.b for mat in matrices]
                                 + [[mol.charge for mol in self.molecules]])
        return ConstraintMatrix.from_a_and_b(a_block, b_block)

    def generate_molecule_charge_constraints(self):
        return MoleculeChargeConstraints.from_charge_constraints(self.charge_constraints,
                                                                 molecules=self.molecules)

    def compute_charges(self, update_molecules=True):
        surface_constraints = self.construct_molecule_constraint_matrix()

        stage_1_constraints = self.generate_molecule_charge_constraints()

        if self.resp_options.stage_2:
            stage_2_constraints = stage_1_constraints.copy(deep=True)
            stage_1_constraints.charge_equivalence_constraints = []

        self.stage_1_charges = RespCharges(charge_constraints=stage_1_constraints,
                                           surface_constraints=surface_constraints,
                                           resp_a=self.resp_options.resp_a1,
                                           **self.resp_options._base_kwargs)
        self.stage_1_charges.solve()

        if self.resp_options.stage_2:
            stage_2_constraints.add_constraints_from_charges(self.stage_1_charges._charges)
            self.stage_2_charges = RespCharges(charge_constraints=stage_2_constraints,
                                               surface_constraints=surface_constraints,
                                               resp_a=self.resp_options.resp_a2,
                                               **self.resp_options._base_kwargs)
            self.stage_2_charges.solve()

        if update_molecules:
            self.update_molecule_charges()

    @ property
    def charges(self):
        if self.stage_2_charges is None:
            try:
                return self.stage_1_charges.charges
            except AttributeError:
                return self.stage_1_charges
        return self.stage_2_charges.charges

    def update_molecule_charges(self):

        all_charges = {}
        default = [None] * len(self.molecules)

        for stage, restraint_ in itertools.product([1, 2], ["un", ""]):
            restraint = f"{restraint_}restrained"
            all_charges[(stage, restraint)] = default
            charges = getattr(self, f"stage_{stage}_charges")
            if charges is not None:
                rcharges = getattr(charges, f"{restraint}_charges")
                if rcharges is not None:
                    all_charges[(stage, restraint)] = rcharges

        for (stage, restraint), charges in all_charges.items():
            for mol, ch in zip(self.molecules, charges):
                setattr(mol, f"stage_{stage}_{restraint}_charges", ch)
