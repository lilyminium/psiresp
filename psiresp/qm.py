import time
from typing import List, Callable, Optional

import numpy as np
from typing_extensions import Literal
import qcelemental as qcel
import qcengine as qcng
import qcfractal.interface as ptl
from pydantic import Field

from .base import Model
from . import qcutils, psi4utils


METHODS = [
    # TODO: can I get this dynamically from Psi4?
    "scf", "hf", "b3lyp", "pw6b95",
    "ccsd", "ccsd(t)", "fno-df-ccsd(t)",
    "pbe", "pbe-d3", "pbe-d3bj",
    "m06-2x", "pw6b95-d3bj",
]

BASIS_SETS = [
    "sto-3g", "3-21g",
    "6-31g", "6-31+g", "6-31++g",
    "6-31g(d)", "6-31g*", "6-31+g(d)", "6-31+g*", "6-31++g(d)", "6-31++g*",
    "6-31g(d,p)", "6-31g**", "6-31+g(d,p)", "6-31+g**", "6-31++g(d,p)", "6-31++g**",
    "aug-cc-pVXZ", "aug-cc-pV(D+d)Z", "heavy-aug-cc-pVXZ",
]

SOLVENTS = ["water"]

G_CONVERGENCES = [
    "qchem", "molpro", "turbomole", "cfour", "nwchem_loose",
    "gau", "gau_loose", "gau_tight", "interfrag_tight", "gau_verytight",
]

QMMethod = Literal[(*METHODS,)]
QMBasisSet = Literal[(*BASIS_SETS,)]
QMSolvent = Optional[Literal[(*SOLVENTS,)]]
QMGConvergence = Literal[(*G_CONVERGENCES,)]


class PCMOptions(Model):
    medium_solver_type: Literal["CPCM", "IEFPCM"] = "CPCM"
    medium_solvent: Literal["water"] = "water"
    cavity_radii_set: Literal["Bondi", "UFF", "Alinger"] = "Bondi"
    cavity_type: Literal["GePol"] = "GePol"
    cavity_scaling: bool = True
    cavity_area: float = 0.3
    cavity_mode: Literal["Implicit"] = "Implicit"

    def to_psi4_string(self):
        return f"""
        Units = Angstrom
        Medium {{
            SolverType = {self.medium_solver_type}
            Solvent = {self.medium_solvent}
        }}

        Cavity {{
            RadiiSet = {self.cavity_radii_set} # Bondi | UFF | Allinger
            Type = {self.cavity_type}
            Scaling = {self.cavity_scaling} # radii for spheres scaled by 1.2
            Area = {self.cavity_area}
            Mode = {self.cavity_mode}
        }}
        """

    def generate_keywords(self):
        keywords = {
            "pcm": "true",
            "pcm_scf_type": "total",
            "pcm__input": self.to_psi4_string()
        }
        return keywords


class BaseQMOptions(Model):
    method: QMMethod = Field(
        default="hf",
        description="QM method for optimizing geometry and calculating ESPs",
    )
    basis: QMBasisSet = Field(
        default="6-31g*",
        description="QM basis set for optimizing geometry and calculating ESPs",
    )
    pcm_options: Optional[PCMOptions] = Field(
        default=None,
        description="Implicit solvent for QM jobs, if any.",
    )
    driver: str = Field(
        default="energy",
        description="QM property to compute",
    )

    def _generate_keywords(self):
        return {}

    @property
    def solvent(self):
        if not self.pcm_options:
            return None
        return self.pcm_options.solvent

    def generate_keywords(self):
        keywords = self._generate_keywords()
        if self.solvent:
            keywords.update(self.pcm_options.generate_keywords())
        return keywords

    def add_compute(self,
                    client: ptl.FractalClient,
                    qcmols: List[qcel.models.Molecule] = [],
                    **kwargs
                    ) -> ptl.models.ComputeResponse:

        keywords = self.generate_keywords()
        kwset = ptl.models.KeywordSet(values=keywords)
        kwid = client.add_keywords([kwset])[0]

        protocols = {"wavefunction": "orbitals_and_eigenvalues"}

        return client.add_compute(program="psi4",
                                  method=self.method,
                                  basis=self.basis,
                                  driver=self.driver,
                                  keywords=kwid,
                                  molecule=qcmols,
                                  protocols=protocols,
                                  **kwargs)

    def add_compute_and_wait(self,
                             client: ptl.FractalClient,
                             qcmols: List[qcel.models.Molecule] = [],
                             query_interval: int = 20,
                             ignore_error: bool = True,
                             **kwargs
                             ) -> List[ptl.models.ResultRecord]:
        from qcfractal.interface.models.records import RecordStatusEnum

        start_time = time.time()
        response = self.add_compute(client, qcmols, **kwargs)

        n_jobs = len(response.submitted)
        if n_jobs:
            n_complete = 0
            n_error = 0
            while n_complete + n_error < n_jobs:
                time.sleep(query_interval)
                results = client.query_results(id=response.submitted)
                status = [r.status for r in results]
                status = np.array([s.value
                                   if isinstance(s, RecordStatusEnum)
                                   else s
                                   for s in status])
                n_complete = (status == "COMPLETE").sum()
                n_error = (status == "ERROR").sum()
                break
            if n_error and not ignore_error:
                err = f"{n_error} jobs errored. Job ids: {[x.id for x in error]}"
                raise ValueError(err)

        elapsed = time.time() - start_time

        records = client.query_results(response.ids)

        # sort by input, not sure if it does this already
        id_num = {x: i for i, x in enumerate(response.ids)}
        records.sort(key=lambda x: id_num[x.id])
        return records


class QMGeometryOptimizationOptions(BaseQMOptions):

    g_convergence: QMGConvergence = "gau_tight"
    driver: str = "gradient"

    max_iter: int = Field(
        default=200,
        description="Maximum number of geometry optimization steps",
    )

    full_hess_every: int = Field(
        default=10,
        description=("Number of steps between each Hessian computation "
                     "during geometry optimization. "
                     "0 computes only the initial Hessian, "
                     "1 means to compute every step, "
                     "-1 means to never compute the full Hessian. "
                     "N means to compute every N steps."),
    )

    def generate_keywords(self):
        return {
            "geom_maxiter": self.max_iter,
            "full_hess_every": self.full_hess_every,
            "g_convergence": self.g_convergence
        }


class QMEnergyOptions(BaseQMOptions):
    pass
