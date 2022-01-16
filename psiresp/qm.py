import time
import pathlib
import logging
from typing import List, Optional, Dict, Any, Union

import numpy as np
from typing_extensions import Literal
import qcelemental as qcel
from pydantic import Field, ValidationError

from .base import Model
from .qcutils import QCWaveFunction

logger = logging.getLogger(__name__)


#: A limited list of QM methods in Psi4
METHODS = [
    # TODO: can I get this dynamically from Psi4?
    "scf", "hf", "b3lyp", "pw6b95",
    "ccsd", "ccsd(t)", "fno-df-ccsd(t)",
    "pbe", "pbe-d3", "pbe-d3bj",
    "m06-2x", "pw6b95-d3bj",
]

#: A limited list of basis sets in Psi4
BASIS_SETS = [
    "sto-3g", "3-21g",
    "6-31g", "6-31+g", "6-31++g",
    "6-31g(d)", "6-31g*", "6-31+g(d)", "6-31+g*", "6-31++g(d)", "6-31++g*",
    "6-31g(d,p)", "6-31g**", "6-31+g(d,p)", "6-31+g**", "6-31++g(d,p)", "6-31++g**",
    "aug-cc-pVXZ", "aug-cc-pV(D+d)Z", "heavy-aug-cc-pVXZ",
]

#: A list of supported and tested solvents in Psi4
SOLVENTS = ["water", ]

#: Optimization convergence options in Psi4
G_CONVERGENCES = [
    "qchem", "molpro", "turbomole", "cfour", "nwchem_loose",
    "gau", "gau_loose", "gau_tight", "interfrag_tight", "gau_verytight",
]

QMMethod = Literal[(*METHODS,)]
QMBasisSet = Literal[(*BASIS_SETS,)]
Solvent = Literal[(*SOLVENTS,)]
QMSolvent = Optional[Solvent]
QMGConvergence = Literal[(*G_CONVERGENCES,)]


class PCMOptions(Model):
    medium_solver_type: Literal["CPCM", "IEFPCM"] = "CPCM"
    medium_solvent: Literal["water"] = "water"
    cavity_radii_set: Literal["Bondi", "UFF", "Alinger"] = "Bondi"
    cavity_type: Literal["GePol"] = "GePol"
    cavity_scaling: bool = True
    cavity_area: float = 0.3
    cavity_mode: Literal["Implicit"] = "Implicit"

    def to_psi4_string(self) -> str:
        """Generate an input string for Psi4's PCM input"""
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

    def generate_keywords(self) -> Dict[str, str]:
        """Generate QCSchema-compatible set of keywords specifying PCM solvent options"""
        keywords = {
            "pcm": "true",
            "pcm_scf_type": "total",
            "pcm__input": self.to_psi4_string()
        }
        return keywords


class BaseQMOptions(Model):
    """Base class for QM computations"""
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
        return self.pcm_options.medium_solvent

    def generate_keywords(self):
        """Generate QCSchema-compatible set of keywords specifying job options"""
        keywords = self._generate_keywords()
        if self.solvent:
            keywords.update(self.pcm_options.generate_keywords())
        return keywords

    def _generate_spec(self, client, **kwargs):
        keywords = self.generate_keywords()
        kwset = {"values": keywords}
        kwid = client.add_keywords([kwset])[0]
        logger.debug(f"Added {kwset} with ID {kwid}")

        spec = dict(
            program="psi4",
            method=self.method,
            basis=self.basis,
            driver=self.driver,
            keywords=kwid,
            protocols=self.protocols,
            **kwargs
        )
        return spec

    def add_compute(self,
                    client: "qcfractal.interface.FractalClient",
                    qcmols: List[qcel.models.Molecule] = [],
                    **kwargs
                    ) -> "qcfractal.interface.models.ComputeResponse":
        """Add compute specification to the client"""
        client = self._validate_client(client)
        spec = self._generate_spec(client, **kwargs)
        logger.debug(f"Submitting specification {spec} for {len(qcmols)} molecules to {client}")
        response = client.add_compute(molecule=qcmols, **spec)
        logger.debug(f"Received response {response} with ids {response.ids}")
        return response

    def add_compute_and_wait(self,
                             client: "qcfractal.interface.FractalClient",
                             qcmols: List[qcel.models.Molecule] = [],
                             **kwargs
                             ) -> List[Any]:
        """
        Add compute specification to the client and
        return post-processed results upon completion.

        See ``wait_for_results`` for more detail.
        """
        response = self.add_compute(client, qcmols=qcmols, **kwargs)
        return self.wait_for_results(client, response_ids=response.ids)

    def wait_for_results(self, client, response_ids=[]):
        raise NotImplementedError

    def run(self,
            client: Optional["qcfractal.interface.FractalClient"] = None,
            qcmols: List[qcel.models.Molecule] = [],
            working_directory=".",
            **kwargs) -> List[Any]:
        """Run the QM computation and return post-processed results.

        If a ``client`` is given, the computations are submitted to
        the server. After all computations have completed, the records
        are returned and post-processed.

        If a ``client`` is not given, the workflow passes to
        :meth:`~psiresp.qm.BaseQMOptions.manage_external_output`. In
        this workflow, Psi4 input files are written out to be executed
        separately and a SystemExit is raised.
        When the job is run again, the workflow checks the
        files to see if the program has successfully executed. If
        an error is found, an error is raised. If all computations
        have completed, the job continues.
        """
        if not qcmols:
            return []
        if not client:
            results = self.manage_external_output(qcmols, working_directory, **kwargs)
            return self.postprocess_atomic_results(results)
        records = self.add_compute_and_wait(client, qcmols=qcmols, **kwargs)
        return self.postprocess_qcrecords(records)

    def postprocess_atomic_results(self, results=[]) -> List[Any]:
        return [self._postprocess_result(r) for r in results]

    def postprocess_qcrecords(self, records=[]) -> List[Any]:
        return [self._postprocess_record(r) for r in records]

    def _postprocess_result(self, result):
        raise NotImplementedError

    def _postprocess_record(self, record):
        raise NotImplementedError

    def write_input(self, qcmol: qcel.models.Molecule,
                    working_directory: Union[str, pathlib.Path] = ".",
                    **kwargs) -> pathlib.Path:
        """Write a Psi4 input file.

        Parameters
        ----------
        qcmol: qcelemental.models.Molecule
            QCElemental molecule
        working_directory: Union[str, pathlib.Path]
            Directory to save input files within

        Returns
        -------
        pathlib.Path
            The path of the input file
        """
        infile = self.get_job_file_for_molecule(qcmol,
                                                working_directory=working_directory,
                                                make_directory=True)
        spec = dict(
            model=dict(method=self.method,
                       basis=self.basis),
            driver=self.driver,
            keywords=self.generate_keywords(),
            protocols=self.protocols,
            molecule=qcmol,
        )
        spec.update(kwargs)

        compute = qcel.models.AtomicInput(**spec)
        with infile.open("wb") as f:
            f.write(compute.serialize("msgpack-ext"))
        logger.debug(f"Wrote to {infile}")
        return infile

    def _validate_client(self, client: "qcfractal.interface.FractalClient"):
        try:
            import qcfractal.interface as ptl
        except ImportError:
            raise ImportError(
                "Could not import qcfractal. "
                "This function requires a QCFractal client. "
                "Please ensure qcfractal is installed"
            )
        if not isinstance(client, ptl.FractalClient):
            try:
                client = ptl.FractalClient(client)
            except TypeError:
                raise TypeError(
                    f"`client` is {type(client)}, "
                    "but must be an instance of qcportal.FractalClient. "
                )
        return client

    def manage_external_output(self, qcmols: List[qcel.models.Molecule],
                               working_directory: Union[str, pathlib.Path] = ".",
                               **kwargs) -> List[Any]:
        results = []
        to_execute = []
        errors = []
        for qcmol in qcmols:
            try:
                result, path = self.read_output(qcmol,
                                                working_directory=working_directory,
                                                return_path=True)
            except (FileNotFoundError, ValidationError):
                path = self.write_input(qcmol, working_directory, **kwargs)
                to_execute.append(path)
            else:
                if not result.success:
                    error_data = result.error
                    if error_data:
                        error_data = result.dict()["error"]
                        error_message = error_data.get("error_message", error_data)
                        error_type = error_data.get("error_type", "Nonspecific")
                        errors.append(f"{error_type} error for {path}: {error_message}")
                    else:
                        to_execute.append(path)
                else:
                    results.append(result)
        if to_execute:
            logger.debug(f"{len(to_execute)} calculations remaining")
            lines = ["#!/usr/bin/env bash"] + [f"psi4 --qcschema {path.name}" for path in to_execute]
            runfile = self.get_run_file(working_directory)
            with runfile.open("w") as f:
                f.write("\n".join(lines))
            logger.debug(f"Wrote to {runfile}")

        if errors:
            raise ValueError(f"Found {len(errors)} errors", *errors)

        if to_execute:
            raise SystemExit("Exiting to allow running QM computations; "
                             f"commands are in {runfile}")
        return results

    def read_output(self, qcmol, working_directory=".",
                    return_path=False):
        infile = self.get_job_file_for_molecule(qcmol,
                                                working_directory=working_directory,
                                                make_directory=False)
        if not infile.exists():
            raise FileNotFoundError(f"Expected file not found: {infile}")
        with infile.open("rb") as f:
            content = f.read()
        data = qcel.util.deserialize(content, "msgpack")
        if data["model"]["basis"] == "":
            data["model"]["basis"] = None
        if "provenance" not in data:
            data["provenance"] = {}
        for kw in ("memory", "nthreads"):
            if kw in data:
                data["provenance"][kw] = data[kw]
        data.pop("return_output", None)
        result = qcel.models.AtomicResult(**data)
        if return_path:
            return result, infile
        return result

    def get_working_directory(self, working_directory="."):
        return pathlib.Path(working_directory) / self.jobname

    def get_job_file_for_molecule(self, qcmol, working_directory=".",
                                  make_directory=False):
        cwd = pathlib.Path(working_directory) / self.jobname
        if make_directory:
            cwd.mkdir(exist_ok=True, parents=True)

        name = qcmol.name if qcmol.name else qcmol.get_molecular_formula()
        filename = f"{name}_{qcmol.get_hash()}_{self.get_hash()}.msgpack"
        return cwd / filename

    def get_run_file(self, working_directory="."):
        cwd = pathlib.Path(working_directory) / self.jobname
        return cwd / f"run_{self.jobname}.sh"

    def _generate_spec_hash(self):
        kw_str = [k.lower() + str(v).lower() for k, v in self.generate_keywords().items()]
        prot_str = [k.lower() + str(v).lower() for k, v in self.protocols.items()]

        fields = (
            self.method.lower(),
            self.basis.lower(),
            self.driver.lower(),
            tuple(sorted(kw_str)),
            tuple(sorted(prot_str))
        )
        spec_hash = hash(fields)
        return spec_hash


class QMGeometryOptimizationOptions(BaseQMOptions):

    jobname = "optimization"

    g_convergence: QMGConvergence = Field(
        default="gau_tight",
        description="Criteria for concluding geometry optimization",
    )

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

    def add_compute(self,
                    client: "qcfractal.interface.FractalClient",
                    qcmols: List[qcel.models.Molecule] = [],
                    **kwargs
                    ) -> "qcfractal.interface.models.ComputeResponse":
        client = self._validate_client(client)
        spec = dict(qc_spec=self._generate_spec(client, **kwargs), keywords=None)
        logger.debug(f"Submitting specification {spec} for {len(qcmols)} molecules to {client}")
        response = client.add_procedure("optimization", "geometric", spec, qcmols)
        logger.debug(f"Received response {response} with ids {response.ids}")
        return response

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

    def _postprocess_result(self, result):
        return result.molecule.geometry

    def _postprocess_record(self, record):
        return record.get_final_molecule().geometry


class QMEnergyOptions(BaseQMOptions):
    jobname = "single_point"

    def wait_for_results(self, client, response_ids=[]):
        results = wait(client, response_ids=response_ids,
                       query_interval=self.query_interval,
                       query_target="results")
        return results

    def _postprocess_result(self, result):
        return QCWaveFunction.from_atomicresult(result)

    def _postprocess_record(self, record):
        return QCWaveFunction.from_qcrecord(record)


def wait(client, response_ids=[], query_interval=20, query_target="results",
         working_directory=None):
    query = getattr(client, f"query_{query_target}")
    n_incomplete = len(response_ids)
    while(n_incomplete):
        time.sleep(query_interval)
        results = query(id=response_ids)
        statuses = [r.status for r in results]
        status = []
        accepted_statuses = ["COMPLETE", "ERROR", "INCOMPLETE", "RUNNING"]
        for each_status in statuses:
            if getattr(each_status, "value") in accepted_statuses:
                status.append(each_status.value)
            else:
                status.append(each_status)
        status = np.array(status)
        n_incomplete = (status == "INCOMPLETE").sum()
        n_complete = (status == "COMPLETE").sum()
        n_error = (status == "ERROR").sum()
        logger.debug(f"{n_incomplete} incomplete, "
                     f"{n_error} errored, "
                     f"{n_complete} complete {query_target} out of {len(response_ids)}")

        if working_directory:
            for i in np.where(status == "COMPLETE")[0]:
                results[i]
    results = query(id=response_ids)
    return sort_results(response_ids=response_ids, results=results)


def sort_results(response_ids=[], results=[]):
    query_order = {x: i for i, x in enumerate(response_ids)}
    return sorted(results, key=lambda x: query_order[x.id])
