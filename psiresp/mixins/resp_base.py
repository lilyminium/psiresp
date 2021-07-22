
import numpy as np
from pydantic import Field

from .. import base
from .qm import QMMixin
from .grid import GridMixin
from .conformer import ConformerOptions


class BaseRespOptions(base.Model):
    """Base RESP options"""
    restrained: bool = Field(default=True,
                             description="Perform a restrained fit")
    hyp_b: float = Field(default=0.1,
                         description="Tightness of hyperbola at its minimum")
    ihfree: bool = Field(
        default=True,
        description="if True, exclude hydrogens from restraint",
    )
    resp_convergence_tol: float = Field(
        default=1e-6,
        description="threshold for convergence",
    )
    resp_max_iter: int = Field(
        default=500,
        description="max number of iterations to solve constraint matrices",
    )
    stage_2: bool = Field(
        default=True,
        description="Whether to run a two stage fit",
    )


class RespMoleculeOptions(base.Model):
    """RESP Molecule options"""

    charge: int = Field(default=0,
                        description="Overall charge of molecule")
    multiplicity: int = Field(default=1,
                              description="Overall multiplicity of molecule")

    conformer_name_template: str = Field(
        default="{resp.name}_c{counter:03d}",
        description="Template string for generating names for new conformers",
    )
    max_generated_conformers: int = Field(
        default=0,
        description="Maximum number of conformers to generate using RDKit",
    )
    min_conformer_rmsd: float = Field(
        default=1.5,
        description="RMSD for pruning duplicates from generated conformers",
    )
    minimize_conformer_geometries: bool = Field(
        default=False,
        description="Whether to minimize the geometries generated by RDKit",
    )
    minimize_max_iter: int = Field(
        default=2000,
        description="Maximum number of iterations to use in RDKit minimization",
    )
    keep_original_resp_geometry: bool = Field(
        default=False,
        description=("Whether to keep the original molecule geometry "
                     "as a conformer"),
    )
    conformer_options: ConformerOptions = Field(
        default_factory=ConformerOptions,
        description="Options for creating new conformer",
    )

    def fix_charge_and_multiplicity(self):
        if self.charge != self.psi4mol.molecular_charge():
            self.psi4mol.set_molecular_charge(self.charge)
            self.psi4mol.update_geometry()
        if self.multiplicity != self.psi4mol.multiplicity():
            self.psi4mol.set_multiplicity(self.multiplicity)
            self.psi4mol.update_geometry()


class RespStage(BaseRespOptions):
    """Resp Stage options"""
    hyp_a: float = Field(default=0.0005,
                         description="scale factor of asymptote limits of hyperbola")

    def get_mask_indices(self, symbols):
        symbols = np.asarray(symbols)
        mask = np.ones_like(symbols, dtype=bool)
        if self.ihfree:
            mask[np.where(symbols == "H")[0]] = False
        return mask


class ContainsQMandGridMixin(base.Model):
    grid_options: GridMixin = Field(
        default_factory=GridMixin,
        description="Options for generating a grid for ESP computation"
    )

    qm_options: QMMixin = Field(
        default_factory=QMMixin,
        description="Options for running QM jobs"
    )

    def __getattr__(self, attrname):
        for options in (self.qm_options, self.grid_options):
            if hasattr(options, attrname):
                return getattr(options, attrname)
        print(type(self))
        return super(ContainsQMandGridMixin, self).__getattr__(attrname)

    def __setattr__(self, attrname, value):
        for options in (self.qm_options, self.grid_options):
            if hasattr(options, attrname):
                return setattr(options, attrname, value)
        return super().__setattr__(attrname, value)
