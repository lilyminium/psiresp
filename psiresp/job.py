from typing import Optional, List
import multiprocessing
import itertools
import pathlib
import logging

import tqdm
from pydantic import Field  # , validator, root_validator
import numpy as np
import scipy.linalg

from . import base, molecule, charge, qm, grid, resp
from .charge import MoleculeChargeConstraints
from .resp import RespCharges
from .orientation import Orientation
from .constraint import ESPSurfaceConstraintMatrix
from .utils import require_package

logger = logging.getLogger(__name__)


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

    working_directory: pathlib.Path = Field(
        default=pathlib.Path("psiresp_working_directory"),
        description="Working directory for saving intermediate files"
    )

    defer_errors: bool = Field(
        default=False,
        description=("Whether to raise an error immediately, "
                     "or gather all errors during ESP computation "
                     "and raise at the end")
    )
    temperature: float = Field(
        default=298.15,
        description="Temperature (in Kelvin) to use when Boltzmann-weighting conformers."
    )

    stage_1_charges: Optional[RespCharges] = Field(
        default=None,
        description="Stage 1 charges. These need to be computed by calling `run()` or `compute_charges()` directly."
    )
    stage_2_charges: Optional[RespCharges] = Field(
        default=None,
        description="Stage 2 charges. These need to be computed by calling `run()` or `compute_charges()` directly."
    )

    n_processes: Optional[int] = Field(
        default=None,
        description=("Number of processes to use in multiprocessing "
                     "during ESP computation. `n_processes=None` uses "
                     "the number of CPUs.")
    )

    @property
    def charges(self):
        if self.stage_2_charges is None:
            try:
                return self.stage_1_charges.charges
            except AttributeError:
                return self.stage_1_charges
        return self.stage_2_charges.charges

    @property
    def n_conformers(self):
        return sum(mol.n_conformers for mol in self.molecules)

    @property
    def n_orientations(self):
        return sum(mol.n_orientations for mol in self.molecules)

    def iter_conformers(self):
        for mol in self.molecules:
            yield from mol.conformers

    def iter_orientations(self):
        for mol in self.molecules:
            for conf in mol.conformers:
                yield from conf.orientations

    def generate_conformers(self):
        """Generate conformers for every molecule"""
        logger.debug(f"Generating conformers for {len(self.molecules)} molecules")
        for mol in tqdm.tqdm(self.molecules, desc="generate-conformers"):
            mol.generate_conformers()
        logger.debug(f"Job has total {self.n_conformers} conformers")

    def generate_orientations(self):
        """Generate orientations for every conformer of every molecule"""
        for mol in self.molecules:
            clear = not mol.keep_original_orientation
            mol.generate_orientations(clear_existing_orientations=clear)

    def optimize_geometries(self, client=None, **kwargs):
        """Compute optimized geometries"""
        conformers = [conformer
                      for mol in self.molecules
                      for conformer in mol.conformers
                      if mol.optimize_geometry and not conformer.is_optimized]
        qcmols = [conf.qcmol for conf in conformers]

        results = self.qm_optimization_options.run(client=client,
                                                   qcmols=qcmols,
                                                   working_directory=self.working_directory,
                                                   **kwargs)
        for conf, geometry in zip(conformers, results):
            conf.set_optimized_geometry(geometry)

    def compute_orientation_energies(self, client=None, **kwargs):
        """Compute wavefunction for each orientation"""
        orientations = [orientation
                        for mol in self.molecules
                        for conformer in mol.conformers
                        for orientation in conformer.orientations
                        if orientation.qc_wavefunction is None]
        qcmols = [o.qcmol for o in orientations]

        results = self.qm_esp_options.run(client=client,
                                          qcmols=qcmols,
                                          working_directory=self.working_directory,
                                          **kwargs)
        for orient, wfn in zip(orientations, results):
            orient.qc_wavefunction = wfn

    def compute_esps(self):
        """Compute ESP on a grid for each orientation in a multiprocessing pool"""
        orientations = [orientation
                        for mol in self.molecules
                        for conformer in mol.conformers
                        for orientation in conformer.orientations
                        if orientation.esp is None]
        tqorientations = tqdm.tqdm(orientations,
                                   desc="compute-esp")
        # create functions for multiprocessing mapping
        computer = self._try_compute_esp if self.defer_errors else self._compute_esp

        with multiprocessing.Pool(processes=self.n_processes) as pool:
            results = pool.map(computer, tqorientations)

        # raise errors if any occurred
        errors = [r for r in results if not isinstance(r, Orientation)]
        if errors:
            raise ValueError(*errors)
        for orientation, o2 in zip(orientations, results):
            # TODO: fix this, it's clumsy
            orientation.esp = o2.esp
            orientation.grid = o2.grid

    def _compute_esp(self, orientation):
        """Compute the grid and ESP for an orientation with the job's grid options"""
        require_package("psi4")
        if orientation.grid is None:
            orientation.compute_grid(grid_options=self.grid_options)
        orientation.compute_esp()
        assert orientation.esp is not None
        return orientation

    def _try_compute_esp(self, orientation):
        """Wrap ESP computation in a try/except to defer errors"""
        require_package("psi4")
        try:
            return self._compute_esp(orientation)
        except BaseException as e:
            return str(e)

    def run(self, client=None, update_molecules: bool = True) -> np.ndarray:
        """Run the whole job"""
        # die early on failure to import
        # we can't decorate the function because pickle is sad
        require_package("psi4")

        self.generate_conformers()
        self.optimize_geometries(client=client)
        self.generate_orientations()
        return self.compute_esps_and_charges(client=client, update_molecules=update_molecules)

    def compute_esps_and_charges(self, client=None, update_molecules: bool = True) -> np.ndarray:
        require_package("psi4")

        self.compute_orientation_energies(client=client)
        self.compute_esps()
        self.compute_charges(update_molecules=update_molecules)
        return self.charges

    def construct_surface_constraint_matrix(self) -> ESPSurfaceConstraintMatrix:
        """
        Construct the constraint matrix for each atom,
        as generated by the ESP at each grid point
        """
        matrices = [
            ESPSurfaceConstraintMatrix.from_orientations(
                orientations=[o for conf in mol.conformers for o in conf.orientations],
                temperature=self.temperature,
            )
            for mol in self.molecules
        ]
        a_mol = scipy.linalg.block_diag(*[mat.coefficient_matrix for mat in matrices])
        a_row = scipy.linalg.block_diag(*[np.ones(mat.constant_vector.shape[0]) for mat in matrices])
        a_zeros = np.zeros((a_row.shape[0], a_row.shape[0]))
        a_block = np.bmat([[a_mol, a_row.T], [a_row, a_zeros]])

        b_block = np.concatenate([mat.constant_vector for mat in matrices]
                                 + [[mol.charge for mol in self.molecules]])
        return ESPSurfaceConstraintMatrix.from_coefficient_matrix(a_block, b_block)

    def generate_molecule_charge_constraints(self) -> MoleculeChargeConstraints:
        """
        Gather the charge constraints pertaining to the molecules
        in this RESP job
        """
        return MoleculeChargeConstraints.from_charge_constraints(self.charge_constraints,
                                                                 molecules=self.molecules)

    def compute_charges(self, update_molecules=True) -> np.ndarray:
        """
        Compute the charges for each molecule. Each Orientation must have had
        the ESP computed, and there must be at least one orientation present.
        """
        surface_constraints = self.construct_surface_constraint_matrix()
        stage_1_constraints = self.generate_molecule_charge_constraints()

        if self.resp_options.stage_2:
            stage_2_constraints = stage_1_constraints.copy(deep=True)
            stage_1_constraints.charge_equivalence_constraints = []

        self.stage_1_charges = RespCharges(charge_constraints=stage_1_constraints,
                                           surface_constraints=surface_constraints,
                                           restraint_height=self.resp_options.restraint_height_stage_1,
                                           **self.resp_options._base_kwargs)
        self.stage_1_charges.solve()

        if self.resp_options.stage_2:
            stage_2_constraints.add_constraints_from_charges(self.stage_1_charges._charges)
            self.stage_2_charges = RespCharges(charge_constraints=stage_2_constraints,
                                               surface_constraints=surface_constraints,
                                               restraint_height=self.resp_options.restraint_height_stage_2,
                                               **self.resp_options._base_kwargs)
            self.stage_2_charges.solve()

        if update_molecules:
            self.update_molecule_charges()
        return self.charges

    def update_molecule_charges(self):
        """
        Update the molecules in the job with the calculated charges
        """
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
