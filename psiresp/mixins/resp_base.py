
import numpy as np
from pydantic import Field


from .. import base
from .conformer import ConformerOptions


class BaseRespOptions(base.Model):
    """ Base RESP options
    Parameters
    ----------
    restrained: bool (optional)
        Whether to perform a restrained fit
    hyp_b: float (optional)
        tightness of hyperbola at its minimum
    ihfree: bool (optional)
        if True, exclude hydrogens from restraint
    resp_convergence_tol: float (optional)
        threshold for convergence
    resp_max_iter: int (optional)
        maximum number of iterations to solve constraint matrices
    stage_2: bool (optional)
        Whether to run a two stage fit
    """
    restrained: bool = True
    hyp_b: float = 0.1
    ihfree: bool = True
    resp_convergence_tol: float = 1e-6
    resp_max_iter: int = 500
    stage_2: bool = False


class RespMoleculeOptions(base.Model):
    """RESP Molecule options

    Parameters
    ----------
    charge: int
        Overall charge of the molecule
    multiplicity: int
        Overall multiplicity of the molecule
    conformer_options: ConformerOptions
        Options for creating new conformer
    conformer_name_template: str
        Template string for generating names for new conformers
    max_generated_conformers: int
        Maximum number of conformers to auto-generate using RDKit
    min_conformer_rmsd: float
        RMSD used to prune duplicates from the RDKit conformer generation
    minimize_conformer_geometries: bool
        Whether to minimize the geometries generated by RDKit
    minimize_max_iter: int
        Maximum number of iterations to use in RDKit minimization
    keep_original_resp_geometry: bool
        Whether to keep the original RESP molecule geometry as a conformer
    """

    charge: int = 0
    multiplicity: int = 1

    conformer_name_template: str = "{resp.name}_c{counter:03d}"
    max_generated_conformers: int = 0
    min_conformer_rmsd: float = 1.5
    minimize_conformer_geometries: bool = False
    minimize_max_iter: int = 2000
    keep_original_resp_geometry: bool = True
    conformer_options: ConformerOptions = Field(default_factory=ConformerOptions)

    def fix_charge_and_multiplicity(self):
        if self.charge != self.psi4mol.molecular_charge():
            self.psi4mol.set_molecular_charge(self.charge)
            self.psi4mol.update_geometry()
        if self.multiplicity != self.psi4mol.multiplicity():
            self.psi4mol.set_multiplicity(self.multiplicity)
            self.psi4mol.update_geometry()


class RespStage(BaseRespOptions):
    """Resp Stage options

    Parameters
    ----------
    hyp_a: float
        scale factor of asymptote limits of hyperbola
    """
    hyp_a: float = 0.0005

    def get_mask_indices(self, symbols):
        symbols = np.asarray(symbols)
        mask = np.ones_like(symbols, dtype=bool)
        if self.ihfree:
            mask[np.where(symbols == "H")[0]] = False
        return mask
