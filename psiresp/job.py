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
from psiresp.constraint import ESPSurfaceConstraintMatrix
from psiresp.qcutils import QCWaveFunction


class Job(base.Model):
    """Class to manage RESP jobs. It is expected that
    all RESP calculations will be run through this class.
    """

    molecules: List[molecule.Molecule] = Field(
        default_factory=list,
        description="Molecules to use for the RESP job"
    )
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

    working_directory: str = Field(
        default="working_directory",
        description="working directory"
    )

    verbose: bool = True
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

    def optimize_geometries(self, client=None):
        conformers = [conformer
                      for mol in self.molecules
                      for conformer in mol.conformers
                      if mol.optimize_geometry and not conformer.is_optimized]
        if client is not None:
            qcmols = [conf.qcmol for conf in conformers]
            ids = self.qm_optimization_options.add_compute(client, qcmols).ids
            for conf, id_ in zip(conformers, ids):
                conformer._qc_id = id_
            results = self.qm_optimization_options.wait_for_results(client,
                                                                    response_ids=ids)
            for conf, record in zip(conformers, results):
                conf.molecule.geometry = record.get_final_molecule().geometry

        elif len(conformers):
            import sys
            for conf in conformers:
                self.qm_optimization_options.write_psi4_input(conf.qcmol,
                                                              working_directory=self.working_directory)
            sys.exit()

    def compute_orientation_energies(self, client=None):
        orientations = [orientation
                        for mol in self.molecules
                        for conformer in mol.conformers
                        for orientation in conformer.orientations
                        if orientation.qc_wavefunction is None]
        qcmols = [o.qcmol for o in orientations]
        if client is not None:
            ids = self.qm_esp_options.add_compute(client, qcmols).ids
            for orientation, id_ in zip(orientations, ids):
                orientation._qc_id = int(id_)
            
            results = self.qm_esp_options.wait_for_results(client, response_ids=ids)
            # set up progress bar
            tqorientations = tqdm.tqdm(orientations, disable=not self.verbose,
                                       desc="qcwavefunction-construction")
            for orient, record in zip(tqorientations, results):
                orient.energy = record.properties.return_energy
                orient.qc_wavefunction = QCWaveFunction.from_qcrecord(record)

    def compute_esps(self):
        orientations = [orientation
                        for mol in self.molecules
                        for conformer in mol.conformers
                        for orientation in conformer.orientations
                        if orientation.esp is None]
        tqorientations = tqdm.tqdm(orientations, disable=not self.verbose,
                                   desc="compute-esp")
        # create functions for multiprocessing mapping
        computer = self._try_compute_esp if self.ignore_errors else self._compute_esp

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            results = pool.map(computer, orientations)
        
        # raise errors if any occurred
        errors = [r for r in results if not isinstance(r, Orientation)]
        if errors:
            raise ValueError(*errors)
        for orientation, o2 in zip(orientations, results):
            orientation.esp = o2.esp
            orientation.grid = o2.grid

    def _compute_esp(self, orientation):
        if orientation.grid is None:
            orientation.compute_grid(grid_options=self.grid_options)
        orientation.compute_esp()
        assert orientation.esp is not None
        return orientation

    def _try_compute_esp(self, orientation):
        try:
            return self._compute_esp(orientation)
        except BaseException as e:
            return str(e)

    def run(self, client=None, update_molecules: bool = True):
        self.generate_conformers()
        self.optimize_geometries(client=client)
        self.generate_orientations()
        self.compute_orientation_energies(client=client)
        self.compute_esps()
        self.compute_charges(update_molecules=update_molecules)

    def construct_surface_constraint_matrix(self):
        matrices = [
            ESPSurfaceConstraintMatrix.from_orientations(
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
        return ESPSurfaceConstraintMatrix.from_a_and_b(a_block, b_block)

    def generate_molecule_charge_constraints(self):
        return MoleculeChargeConstraints.from_charge_constraints(self.charge_constraints,
                                                                 molecules=self.molecules)

    def compute_charges(self, update_molecules=True):
        surface_constraints = self.construct_surface_constraint_matrix()
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
