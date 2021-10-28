import time
import pathlib
from typing import List, Callable, Optional, Dict

import numpy as np
from typing_extensions import Literal
import qcelemental as qcel
import qcengine as qcng
import qcfractal.interface as ptl
from qcfractal.interface.models.records import RecordStatusEnum
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
    query_interval: int = Field(
        default=20,
        description="Number of seconds between queries"
    )
    protocols: Dict[str, str] = Field(
        default={"wavefunction": "orbitals_and_eigenvalues"},
        description="Wavefunction protocols"
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


        return client.add_compute(program="psi4",
                                  method=self.method,
                                  basis=self.basis,
                                  driver=self.driver,
                                  keywords=kwid,
                                  molecule=qcmols,
                                  protocols=self.protocols,
                                  **kwargs)
    
    def write_input(self, qcmol, working_directory="."):
        cwd = pathlib.Path(working_directory) / self._generate_id(qcmol)
        cwd.mkdir(exist_ok=True, parents=True)

        compute = qcel.models.AtomicInput(
            model=dict(method=self.method,
                       basis=self.basis),
            driver=self.driver,
            keywords=self.generate_keywords(),
            protocols=self.protocols,
            molecule=qcmol,
        )

        infile = cwd / "data.msgpack"

        with infile.open() as f:
            f.write(compute.serialize("msgpack-ext"))


    def _generate_id(self, qcmol):
        return hash(qcmol.get_hash(), self)


    # def add_compute_and_wait(self,
    #                          client: ptl.FractalClient,
    #                          qcmols: List[qcel.models.Molecule] = [],
    #                          query_interval: int = 20,
    #                          ignore_error: bool = True,
    #                          **kwargs
    #                          ) -> List[ptl.models.ResultRecord]:
    #     from qcfractal.interface.models.records import RecordStatusEnum

    #     start_time = time.time()
    #     response = self.add_compute(client, qcmols, **kwargs)
    #     n_incomplete = len(qcmols)
    #     while(n_incomplete):
    #         time.sleep(query_interval)
    #         results = client.query_results(id=response.ids)
    #         status = [r.status for r in results]
    #         status = np.array([s.value
    #                            if isinstance(s, RecordStatusEnum)
    #                            else s
    #                            for s in status])
    #         n_incomplete = (status == "INCOMPLETE").sum()

    #     elapsed = time.time() - start_time

    #     records = client.query_results(response.ids)

    #     # sort by input, not sure if it does this already
    #     id_num = {x: i for i, x in enumerate(response.ids)}
    #     records.sort(key=lambda x: id_num[x.id])
    #     return records


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

    def wait_for_results(self, client, response_ids=[],
                         working_directory=None):
        results = wait(client, response_ids=response_ids,
                        query_interval=self.query_interval,
                        query_target="procedures",
                        working_directory=working_directory)
        return results



class QMEnergyOptions(BaseQMOptions):
    def wait_for_results(self, client, response_ids=[]):
        results = wait(client, response_ids=response_ids,
                       query_interval=self.query_interval,
                       query_target="results")
        return results

def wait(client, response_ids=[], query_interval=20, query_target="results",
         working_directory=None):
    query = getattr(client, f"query_{query_target}")
    n_incomplete = len(response_ids)
    while(n_incomplete):
        time.sleep(query_interval)
        results = query(id=response_ids)
        status = [r.status for r in results]
        status = np.array([s.value
                            if isinstance(s, RecordStatusEnum)
                            else s
                            for s in status])
        n_incomplete = (status == "INCOMPLETE").sum()

        if working_directory:
            for i in np.where(status == "COMPLETE")[0]:
                results[i]
    results = query(id=response_ids)
    return sort_results(response_ids=response_ids, results=results)

def sort_results(response_ids=[], results=[]):
    query_order = {x: i for i, x in enumerate(response_ids)}
    return sorted(results, key=lambda x: query_order[x.id])