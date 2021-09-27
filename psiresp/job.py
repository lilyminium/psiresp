import time
from typing import Optional, List, Tuple
import multiprocessing
import functools

import tqdm
from pydantic import Field, validator, root_validator
import numpy as np

from . import base, psi4utils, orutils, molecule, charge, qm, grid, resp


class Job(base.Model):

    molecules: List[molecule.Molecule] = []
    qm_optimization_options: qm.QMGeometryOptimizationOptions = Field(
        default=qm.QMGeometryOptimizationOptions(),
        description="QM options for geometry optimization"
    )
    qm_esp_options: qm.EnergyOptions = Field(
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

    stage_1_charges = None
    stage_2_charges = None

    def generate_conformers(self):
        kwargs = self.conformer_generation_options.dict()
        kwargs.pop("clear_existing_orientations", None)
        for mol in self.molecules:
            mol.generate_conformers(**kwargs)

    def generate_orientations(self):
        for mol in self.molecules:
            mol.generate_orientations(self.grid_options)

    def optimize_geometries(self):
        conformers = [conformer.qcmol
                      for mol in self.molecules
                      for conformer in mol.conformers
                      if mol.optimize_geometry and not conformer.is_optimized]

        records = self.qm_optimization_options.add_compute_and_wait(self.client,
                                                                    conformers)

        for conf, rec in zip(conformers, records):
            conf.molecule.geometry = rec.return_result

    def compute_esps(self):
        orientations = [orientation
                        for mol in self.molecules
                        for conformer in mol.conformers
                        for orientation in conformer.orientations
                        if orientation.esp is None]
        qcmols = [o.qcmol for o in orientations]

        records = self.qm_energy_options.add_compute_and_wait(self.client,
                                                              qcmols)

        # create functions for multiprocessing mapping
        def compute_esp(orientation, record):
            orientation.compute_esp(record)

        def try_compute_esp(orientation, record):
            try:
                compute_esp(orientation, record)
            except BaseException as e:
                return str(e)
        computer = try_compute_esp if self.ignore_errors else compute_esp

        # compute esp with possible verbosity
        with multiprocessing.Pool(processes=self.n_processors) as pool:
            results = tqdm.tqdm(
                pool.starmap(computer, orientations, records),
                disable=not self.verbose,
            )

        # raise errors if any occurred
        errors = [r for r in results if r is not None]
        if errors:
            raise ValueError(*errors)

    def run(self, update_molecules: bool = True):
        self.generate_conformers()
        self.optimize_geometries()
        self.generate_orientations()
        self.compute_esps()
        self.resp_options.compute_charges(self.charge_constraints,
                                          update_molecules=update_molecules)

    def compute_charges(self, update_molecules=True):
        stage_1_constraints = MoleculeChargeConstraints.from_charge_constraints(self.charge_constraints,
                                                                                molecules=self.molecules)

        if self.resp_options.stage_2:
            stage_2_constraints = stage_1_constraints.copy(deep=True)
            stage_1_constraints.charge_equivalence_constraints = []

        self.stage_1_charges = RespCharges(molecule_constraints=stage_1_constraints,
                                           resp_a=self.resp_options.resp_a1,
                                           **self.resp_options._base_kwargs)
        self.stage_1_charges.solve()

        if self.stage_2:
            stage_2_constraints.add_constraints_from_charges(self.stage_1_charges)
            self.stage_2_charges = RespCharges(molecule_constraints=stage_2_constraints,
                                               resp_a=self.resp_options.resp_a2,
                                               **self.resp_options._base_kwargs)
            self.stage_2_charges.solve()

        if update_molecules:
            self.update_molecule_charges()

    @property
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
