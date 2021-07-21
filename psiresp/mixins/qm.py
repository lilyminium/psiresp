import os
import io
import re
import logging
import textwrap
import subprocess
from typing import Optional, Tuple

from typing_extensions import Literal
import psi4
from pydantic import Field, validator

from .. import base, utils

logger = logging.getLogger(__name__)

command_stream = io.StringIO()

QM_METHODS = [
    # TODO: can I get this dynamically from Psi4?
    "scf", "hf", "b3lyp", "pw6b95",
    # "rscf", "uscf", "roscf",
    # "rhf", "uhf", "rohf",
    # "blyp", "b3lyp-d3", "b3lyp-d3bj",
    # "mp2", "mp3", "mp4",
    # "cc2", "cc3",
    # "cisd", "cisdt", "cisdtq",
    "ccsd", "ccsd(t)", "fno-df-ccsd(t)",
    # "sapt0", "sapt2", "sapt2+", "sapt2+(3)", "sapt2+3",
    "pbe", "pbe-d3", "pbe-d3bj",
    "m06-2x", "pw6b95-d3bj",
]

QM_BASIS_SETS = [
    "sto-3g", "3-21g",
    "6-31g", "6-31+g", "6-31++g",
    "6-31g(d)", "6-31g*", "6-31+g(d)", "6-31+g*", "6-31++g(d)", "6-31++g*",
    "6-31g(d,p)", "6-31g**", "6-31+g(d,p)", "6-31+g**", "6-31++g(d,p)", "6-31++g**",
    # "6-311g", "6-311+g", "6-311++g",
    # "6-311g(d)", "6-311g*", "6-311+g(d)", "6-311+g*", "6-311++g(d)", "6-311++g*",
    # "6-311g(d,p)", "6-311g**", "6-311+g(d,p)", "6-311+g**", "6-311++g(d,p)", "6-311++g**",
    # "6-311g(2d)", "6-311+g(2d)", "6-311++g(2d)",
    # "6-311g(2d,p)", "6-311+g(2d,p)", "6-311++g(2d,p)",
    # "6-311g(2d,2p)", "6-311+g(2d,2p)", "6-311++g(2d,2p)",
    # "6-311g(2df)", "6-311+g(2df)", "6-311++g(2df)",
    # "6-311g(2df,p)", "6-311+g(2df,p)", "6-311++g(2df,p)",
    # "6-311g(2df,2p)", "6-311+g(2df,2p)", "6-311++g(2df,2p)",
    # "6-311g(2df,2pd)", "6-311+g(2df,2pd)", "6-311++g(2df,2pd)",
    # "6-311g(3df)", "6-311+g(3df)", "6-311++g(3df)",
    # "6-311g(3df,p)", "6-311+g(3df,p)", "6-311++g(3df,p)",
    # "6-311g(3df,2p)", "6-311+g(3df,2p)", "6-311++g(3df,2p)",
    # "6-311g(3df,2pd)", "6-311+g(3df,2pd)", "6-311++g(3df,2pd)",
    # "6-311g(3df,3pd)", "6-311+g(3df,3pd)", "6-311++g(3df,3pd)",
    "aug-cc-pVXZ", "aug-cc-pV(D+d)Z", "heavy-aug-cc-pVXZ",
    # "jun-cc-pVXZ", "may-cc-pVXZ", "cc-pVXZ",
    # "cc-pVXZ", "cc-pV(X+d)Z", "cc-pCVXZ",
    # "cc-pCV(X+d)Z", "cc-pwCVXZ", "cc-pwCV(X+d)Z"
]

QM_SOLVENTS = ["water"]

QM_G_CONVERGENCES = [
    "qchem", "molpro", "turbomole", "cfour", "nwchem_loose",
    "gau", "gau_loose", "gau_tight", "interfrag_tight", "gau_verytight",
]

QMMethod = Literal[(*QM_METHODS,)]
QMBasisSet = Literal[(*QM_BASIS_SETS,)]
QMSolvent = Optional[Literal[(*QM_SOLVENTS,)]]
QMGConvergence = Literal[(*QM_G_CONVERGENCES,)]


def get_cased_value(value, allowed_values=[]):
    lower = value.strip().lower()
    for allowed in allowed_values:
        if allowed.lower() == lower:
            return allowed


class QMMixin(base.Model):
    """Mixin for QM jobs in Psi4"""

    qm_method: QMMethod = Field(
        default="hf",
        description="QM method for optimizing geometry and calculating ESPs",
    )
    qm_basis_set: QMBasisSet = Field(
        default="6-31g*",
        description="QM basis set for optimizing geometry and calculating ESPs",
    )
    solvent: QMSolvent = Field(
        default=None,
        description="Implicit solvent for QM jobs, if any.",
    )
    geom_max_iter: int = Field(
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
    g_convergence: QMGConvergence = Field(
        default="gau_tight",
        description="Criteria for concluding geometry optimization"
    )
    esp_infile: str = Field(
        default="{name}_esp.in",
        description="Filename to write Psi4 ESP job input",
    )
    opt_infile: str = Field(
        default="{name}_opt.in",
        description="Filename to write Psi4 optimisation job input",
    )
    opt_outfile: str = Field(
        default="{name}_opt.out",
        description="Filename for Psi4 optimisation job output",
    )
    execute_qm: bool = Field(
        default=True,
        description=("Whether to execute the QM jobs. "
                     "If ``False``, input files will be written but not run. "
                     "If called from `Resp.run()` or similar, "
                     "the job will exit so that you can run the QM jobs "
                     "in parallel yourself."
                     ),
    )

    @validator("qm_method")
    def validate_method(cls, v):
        cased = get_cased_value(v, QM_METHODS)
        assert cased, "must be one of `psiresp.mixins.qm.QM_METHODS`"
        return cased

    @validator("qm_basis_set")
    def validate_basis_set(cls, v):
        cased = get_cased_value(v, QM_BASIS_SETS)
        assert cased, "must be one of `psiresp.mixins.qm.QM_BASIS_SETS`"
        return cased

    @validator("solvent")
    def validate_solvent(cls, v):
        if v is None:
            return v
        cased = get_cased_value(v, QM_SOLVENTS)
        assert cased, "must be one of `psiresp.mixins.qm.QM_SOLVENTS` or None"
        return cased

    @validator("g_convergence")
    def validate_convergence(cls, v):
        cased = get_cased_value(v, QM_G_CONVERGENCES)
        assert cased, "must be one of `psiresp.mixins.qm.QM_G_CONVERGENCES`"
        return cased

    @staticmethod
    def get_mol_spec(molecule: psi4.core.Molecule) -> str:
        """Create Psi4 molecule specification from Psi4 molecule

        Parameters
        ----------
        molecule: Psi4Mol

        Returns
        -------
        mol_spec: str
        """
        mol = molecule.create_psi4_string_from_molecule()
        # remove any clashing fragment charge/multiplicity
        pattern = r"--\n\s*\d \d\n"
        mol = re.sub(pattern, "", mol)
        return f"molecule {molecule.name()} {{\n{mol}\n}}\n\n"

    def write_opt_file(self, psi4mol: psi4.core.Molecule) -> Tuple[str, str]:
        """Write psi4 optimization input to file
        and return expected input and output paths.

        Parameters
        ----------
        psi4mol: psi4.core.Molecule
            Psi4 molecule

        Returns
        -------
        infile, outfile: tuple[str, str]
            Input and output paths
        """
        opt_file = self.get_mol_spec(psi4mol)
        opt_file += textwrap.dedent(f"""
        set {{
            basis {self.qm_basis_set}
            geom_maxiter {self.geom_maxiter}
            full_hess_every {self.full_hess_every}
            g_convergence {self.g_convergence}
        }}

        optimize('{self.qm_method}')
        """)

        infile = self.opt_infile.format(name=psi4mol.name())
        infile = os.path.abspath(infile)
        outfile = self.opt_outfile.format(name=psi4mol.name())
        outfile = os.path.abspath(outfile)

        with open(infile, "w") as f:
            f.write(opt_file)
        logger.info(f"Wrote optimization input to {infile}")

        return infile, outfile

    def write_esp_file(self, psi4mol: psi4.core.Molecule) -> str:
        """Write psi4 esp input to file and return input filename

        Parameters
        ----------
        psi4mol: psi4.core.Molecule
            Psi4 molecule

        Returns
        -------
        filename: str
            Input filename
        """
        esp_file = self.get_mol_spec(psi4mol)

        esp_file += f"set basis {self.qm_basis_set}\n"

        if self.solvent:
            esp_file += textwrap.dedent(f"""
            set {{
                pcm true
                pcm_scf_type total
            }}

            pcm = {{
                Units = Angstrom
                Medium {{
                    SolverType = CPCM
                    Solvent = {self.solvent}
                }}

                Cavity {{
                    RadiiSet = bondi # Bondi | UFF | Allinger
                    Type = GePol
                    Scaling = True # radii for spheres scaled by 1.2
                    Area = 0.3
                    Mode = Implicit
                }}
            }}

            """)

        esp_file += textwrap.dedent(f"""\
        E, wfn = prop('{self.qm_method}', properties=['GRID_ESP'], return_wfn=True)
        esp = wfn.oeprop.Vvals()
            """)

        filename = self.esp_infile.format(name=psi4mol.name())

        with open(filename, "w") as f:
            f.write(esp_file)
        logger.info(f"Wrote ESP input to {filename}")

        return filename

    def try_run_qm(self, infile: str, outfile: Optional[str] = None,
                   cwd: Optional[str] = None):
        """
        Try to run QM job if ``execute_qm`` is True; else logs the
        command and raises an error to exist

        Parameters
        ----------
        infile: str
            Input Psi4 file
        outfile: str (optional)
            Output Psi4 file
        cwd: str (optional)
            Working directory to run the process from

        Raises
        ------
        NoQMExecutionError
            if QM is not run
        """
        command = f"{psi4.executable} -i {infile}"
        if outfile is not None:
            command += f"-o {outfile}"

        if not self.execute_qm:
            command_stream.write(command + "\n")
            logger.info(command)
            raise utils.NoQMExecutionError("Not running qm")

        proc = subprocess.run(command, shell=True,
                              cwd=cwd, stderr=subprocess.PIPE)
        return proc
