import os
import io
import re
import logging
import textwrap
import subprocess
from typing import Optional, Tuple
from dataclasses import dataclass

from typing_extensions import Literal
import numpy as np
import psi4

from .. import base, psi4utils, utils

logger = logging.getLogger(__name__)

command_stream = io.StringIO()


class QMMixin(base.Model):
    """Mixin for QM jobs in Psi4

    Parameters
    ----------
    qm_method: str
        QM method
    qm_basis_set: str
        Basis set
    solvent: str
        Solvent, if any.
        This has only been tested on "water" and None.
    geom_max_iter: int
        Maximum number of geometry optimization steps
    full_hess_every: int
        Number of steps between each Hessian computation during geometry
        optimization. 0 computes only the initial Hessian, 1 means
        to compute every step, -1 means to never compute the full Hessian.
        N means to compute every N steps.
    g_convergence: str
        Optimization criteria.
    """
    # qm_method: Literal["scf", "hf", "mp2", "mp3", "ccsd"] = "scf"
    # TODO: https://psicode.org/psi4manual/master/api/psi4.driver.energy.html
    # add in all relevant methods
    qm_method: str = "scf"
    # TODO: https://psicode.org/psi4manual/master/basissets_tables.html#apdx-basistables
    # TODO: add in all relevant bases
    # qm_basis_set: Literal[]
    qm_basis_set: str = "6-31g*"
    # TODO: should I restrict the solvents?
    solvent: Optional[str] = None
    geom_max_iter: int = 200
    full_hess_every: int = 10
    g_convergence: str = "gau_tight"
    esp_infile: str = "{name}_esp.in"
    opt_infile: str = "{name}_opt.in"
    opt_outfile: str = "{name}_opt.out"
    execute_qm: bool = True

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
