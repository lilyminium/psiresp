import os
import json

import numpy as np
from qcelemental.models import AtomicResult, BasisSet
from qcengine.programs.base import register_program
from qcelemental.util import deserialize, parse_version, safe_version, which, which_import

from qcengine.exceptions import InputError, RandomError, ResourceError, UnknownError
from qcengine.programs.psi4 import Psi4Harness
from qcengine.util import execute, popen, temporary_directory


class Psi4ESPHarness(Psi4Harness):

    def compute(self, input_model: "AtomicInput", config: "TaskConfig") -> "AtomicResult":
        """
        Runs Psi4 in API mode
        """

        if "properties" in input_model.model:
            input_properties = input_model.model["properties"]
            if len(input_properties) == 1 and input_properties[0].lower() == "grid_esp":
                return self.compute_esp()
        return super().compute(input_model, config)

    def _get_version(self):
        self.found(raise_error=True)
        pversion = parse_version(self.get_version())
        if pversion < parse_version("1.2"):
            raise ResourceError("Psi4 version '{}' not understood.".format(self.get_version()))
        return pversion

    def _get_parent(self, input_model: "AtomicInput", config: "TaskConfig"):
        # Location resolution order config.scratch_dir, $PSI_SCRATCH, /tmp
        parent = config.scratch_directory
        if parent is None:
            parent = os.environ.get("PSI_SCRATCH", None)

        if isinstance(input_model.model.basis, BasisSet):
            raise InputError("QCSchema BasisSet for model.basis not implemented. Use string basis name.")

        # Basis must not be None for HF3c
        old_basis = input_model.model.basis
        input_model.model.__dict__["basis"] = old_basis or ""

        return parent

    def _esp_input_file_from_model(self, input_model: "AtomicInput", config: "TaskConfig"):
        import psi4

        mol_json = input_model.molecule.json()
        mol = psi4.core.Molecule.from_schema(mol_json)
        mol_section = f"""molecule {input_model.molecule.name} {{
            {mol.create_psi4_string_from_molecule()}
            }}\n"""

        memory = int(config.memory * 1024 * 1024 * 1024 * 0.95)
        config_section = f"memory {memory}\nset_num_threads({config.ncores})\n"
        options = ""

        for k, v in input_model.keywords.items():
            if k == "grid":
                continue
            if k.endswith("__input"):
                k_ = k.split("__input")[0]
                options += f"{k_} = {{\n{v}\n}}"
            else:
                options += f"\nset {k} {v}"

        method = input_model.model["method"]
        esp_file = f"""\
            {config_section}
            {mol_section}

            set basis {input_model.model["basis"]}
            {options}

            E, wfn = prop('{method}', properties=['GRID_ESP'], return_wfn=True)
            esp = wfn.oeprop.Vvals()
            """
        return esp_file

    def _grid_input_from_array(self, input_model: "AtomicInput"):
        try:
            grid = input_model.keywords["grid"]
        except KeyError:
            raise ValueError("'grid' must be provided as a keyword option "
                             "for property 'GRID_ESP'")

        assert isinstance(grid, np.ndarray) and len(grid.shape) == 2
        assert grid.shape[1] == 3

        return "\n".join([" ".join(list(map(str, row))) for row in grid])

    def compute_esp(self, input_model: "AtomicInput", config: "TaskConfig") -> "AtomicResult":
        pversion = self._get_version()
        parent = self._get_parent(input_model, config)

        # get input files
        esp_file = self._esp_input_file_from_model(input_model, config)
        grid_text = self._grid_input_from_array(input_model)

        # do the actual computation

        error_type = None
        error_message = None
        compute_success = False

        with temporary_directory(parent=parent, suffix="_psi_scratch") as tmpdir:

            caseless_keywords = {k.lower(): v for k, v in input_model.keywords.items()}
            if (input_model.molecule.molecular_multiplicity != 1) and ("reference" not in caseless_keywords):
                input_model.keywords["reference"] = "uhf"

            # Old-style JSON-based command line
            if pversion < parse_version("1.4a2.dev160"):
                # Execute the program
                success, output = execute(
                    [which("psi4"), "--scratch", tmpdir, "input.dat"],
                    {"input.dat": esp_file,
                     "grid.dat": ...},
                    ["data.json"],
                    scratch_directory=tmpdir,
                )

                output_data = input_model.json()
                if success:
                    output_data = json.loads(output["outfiles"]["data.json"])
                    if "extras" not in output_data:
                        output_data["extras"] = {}

                    # Check QCVars
                    local_qcvars = output_data.pop("psi4:qcvars", None)
                    if local_qcvars:
                        # Edge case where we might already have qcvars, should not happen
                        if "qcvars" in output_data["extras"]:
                            output_data["extras"]["local_qcvars"] = local_qcvars
                        else:
                            output_data["extras"]["qcvars"] = local_qcvars

                    if output_data.get("success", False) is False:
                        error_message, error_type = self._handle_errors(output_data)
                    else:
                        compute_success = True

                else:
                    error_message = output.get("stderr", "No STDERR output")
                    error_type = "execution_error"

                # Reset the schema if required
                output_data["schema_name"] = "qcschema_output"
                output_data.pop("memory", None)
                output_data.pop("nthreads", None)
                output_data["stdout"] = output_data.pop("raw_output", None)

            else:

                if input_model.extras.get("psiapi", False):
                    import psi4

                    orig_scr = psi4.core.IOManager.shared_object().get_default_path()
                    psi4.core.set_num_threads(config.ncores, quiet=True)
                    psi4.set_memory(f"{config.memory}GB", quiet=True)
                    # psi4.core.IOManager.shared_object().set_default_path(str(tmpdir))
                    if pversion < parse_version("1.5rc1"):  # adjust to where DDD merged
                        # slightly dangerous in that if `qcng.compute({..., psiapi=True}, "psi4")` called *from psi4
                        #   session*, session could unexpectedly get its own files cleaned away.
                        output_data = psi4.schema_wrapper.run_qcschema(input_model).dict()
                    else:
                        output_data = psi4.schema_wrapper.run_qcschema(input_model, postclean=False).dict()
                    # success here means execution returned. output_data may yet be qcel.models.AtomicResult or qcel.models.FailedOperation
                    success = True
                    if output_data.get("success", False):
                        output_data["extras"]["psiapi_evaluated"] = True
                    psi4.core.IOManager.shared_object().set_default_path(orig_scr)
                else:
                    run_cmd = [
                        which("psi4"),
                        "--scratch",
                        str(tmpdir),
                        "--nthread",
                        str(config.ncores),
                        "--memory",
                        f"{config.memory}GB",
                        "--qcschema",
                        "data.msgpack",
                    ]
                    input_files = {"data.msgpack": input_model.serialize("msgpack-ext")}
                    success, output = execute(
                        run_cmd, input_files, ["data.msgpack"], as_binary=["data.msgpack"], scratch_directory=tmpdir
                    )
                    if success:
                        output_data = deserialize(output["outfiles"]["data.msgpack"], "msgpack-ext")
                    else:
                        output_data = input_model.dict()

                if success:
                    if output_data.get("success", False) is False:
                        error_message, error_type = self._handle_errors(output_data)
                    else:
                        compute_success = True
                else:
                    error_message = output.get("stderr", "No STDERR output")
                    error_type = "execution_error"

        # Dispatch errors, PSIO Errors are not recoverable for future runs
        if compute_success is False:

            if "PSIO Error" in error_message:
                if "scratch directory" in error_message:
                    # Psi4 cannot access the folder or file
                    raise ResourceError(error_message)
                else:
                    # Likely a random error, worth retrying
                    raise RandomError(error_message)
            elif ("SIGSEV" in error_message) or ("SIGSEGV" in error_message) or ("segmentation fault" in error_message):
                raise RandomError(error_message)
            elif ("TypeError: set_global_option" in error_message) or (error_type == "ValidationError"):
                raise InputError(error_message)
            elif "RHF reference is only for singlets" in error_message:
                raise InputError(error_message)
            else:
                raise UnknownError(error_message)

        # Reset basis
        output_data["model"]["basis"] = old_basis

        # Move several pieces up a level
        output_data["provenance"]["memory"] = round(config.memory, 3)
        output_data["provenance"]["nthreads"] = config.ncores

        # Delete keys
        output_data.pop("return_output", None)

        return AtomicResult(**output_data)
