import pathlib
from typing import List, Optional, ClassVar

from pydantic import Field

from . import base, qm, grid, molecule, resp, charge
from .job import Job
from .due import due, Doi
from .utils import update_dictionary

__all__ = [
    "ConfiguredJob",
    "TwoStageRESP",
    "OneStageRESP",
    "ESP",
    "WeinerESP",
    "ATBRESP",
    "RESP2"
]


class ConfiguredJob(Job):

    _configuration: ClassVar[dict] = {}

    def __init__(self, *args, **kwargs):
        obj = Job(*args, **kwargs)
        objdct = obj.dict()
        for option_name, option_config in self._configuration.items():
            prefix = option_name.split("_")[0] + "_"
            for field in objdct.keys():
                if field.startswith(prefix):
                    for name, value in option_config.items():
                        update_dictionary(objdct[field], name, value)
        super().__init__(**objdct)


@due.dcite(
    Doi("10.1021/j100142a004"),
    description="RESP-A1 model",
    path="psiresp.configs.TwoStageRESP",
)
class TwoStageRESP(ConfiguredJob):
    """Configuration for typical RESP

    This corresponds to RESP-A1 in the R.E.D. server.
    """
    _configuration: ClassVar[dict] = dict(
        qm_options=dict(method="hf",
                        basis="6-31g*"
                        ),
        grid_options=dict(use_radii="msk"),
        resp_options=dict(restraint_height_stage_1=0.0005,
                          restraint_height_stage_2=0.001,
                          restraint_slope=0.1,
                          stage_2=True,
                          exclude_hydrogens=True,
                          restrained_fit=True),
    )


@due.dcite(
    Doi("10.1039/c0cp00111b"),
    description="RESP-A2 model",
    path="psiresp.configs.OneStageRESP",
)
class OneStageRESP(ConfiguredJob):
    """Configuration for one-stage RESP

    This corresponds to RESP-A2 in the R.E.D. server.
    """
    _configuration = dict(qm_options=dict(method="hf",
                                          basis="6-31g*"
                                          ),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(restraint_height_stage_1=0.01,
                                            restraint_height_stage_2=0.0,
                                            restraint_slope=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=True,
                                            restrained_fit=True),
                          )


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    description="ESP-A1 multimolecule model",
    path="psiresp.configs.ESP",
)
class ESP(ConfiguredJob):
    """Configuration for typical unrestrained ESP

    This corresponds to ESP-A1 in the R.E.D. server.
    """
    _configuration = dict(qm_options=dict(method="hf",
                                          basis="6-31g*"
                                          ),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(restraint_height_stage_1=0.0,
                                            restraint_height_stage_2=0.0,
                                            restraint_slope=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=True,
                                            restrained_fit=False),
                          )


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    Doi("10.1039/c0cp00111b"),
    description="ESP-A2 model",
    path="psiresp.configs.WeinerESP",
)
class WeinerESP(ConfiguredJob):
    """
    Configuration for the unrestrained ESP fit
    used in the Weiner et al. AMBER force field.

    This corresponds to ESP-A2 in the R.E.D. server.
    """

    _configuration = dict(qm_options=dict(method="hf",
                                          basis="sto-3g"
                                          ),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(restraint_height_stage_1=0.0,
                                            restraint_height_stage_2=0.0,
                                            restraint_slope=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=True,
                                            restrained_fit=False),
                          )


@due.dcite(
    Doi("10.1021/ct200196m"),
    description="ATB model",
    path="psiresp.configs.ATBRESP",
)
class ATBRESP(ConfiguredJob):
    """Configuration used by the AutomatedTopologyBuilder"""
    _configuration = dict(qm_options=dict(method="b3lyp",
                                          basis="6-31g*",
                                          pcm_options=dict(
                                              solvent="water"
                                          )),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(restraint_height_stage_1=0.0,
                                            restraint_height_stage_2=0.0,
                                            restraint_slope=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=False,
                                            restrained_fit=False),
                          )


@ due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class RESP2(base.Model):
    """Class to manage RESP2 jobs

    This is based off the method Schauperl et al. 2021,
    which uses a much higher level of theory and
    interpolates charges between the gas and aqueous phases.
    """
    molecules: List[molecule.Molecule] = Field(
        default_factory=list,
        description="Molecules to use for the RESP job"
    )
    solvent_qm_optimization_options: qm.QMGeometryOptimizationOptions = Field(
        default=qm.QMGeometryOptimizationOptions(
            method="pw6b95",
            basis="aug-cc-pV(D+d)Z",
            pcm_options=qm.PCMOptions(solvent="water")
        ),
        description="QM options for geometry optimization"
    )
    solvent_qm_esp_options: qm.QMEnergyOptions = Field(
        default=qm.QMEnergyOptions(
            method="pw6b95",
            basis="aug-cc-pV(D+d)Z",
            pcm_options=qm.PCMOptions(medium_solvent="water")
        ),
        description="QM options for ESP computation"
    )
    grid_options: grid.GridOptions = Field(
        default=grid.GridOptions(
            use_radii="bondi",
            vdw_point_density=2.5
        ),
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

    n_processes: Optional[int] = Field(
        default=None,
        description=("Number of processes to use in multiprocessing "
                     "during ESP computation. `n_processes=None` uses "
                     "the number of CPUs.")
    )

    vacuum: Optional[Job] = None
    solvated: Optional[Job] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vacuum_opt = self.solvent_qm_optimization_options.copy(deep=True)
        vacuum_opt.pcm_options = None
        vacuum_esp = self.solvent_qm_esp_options.copy(deep=True)
        vacuum_esp.pcm_options = None

        self.vacuum = Job(molecules=[x.copy(deep=True) for x in self.molecules],
                          qm_optimization_options=vacuum_opt,
                          qm_esp_options=vacuum_esp,
                          grid_options=self.grid_options,
                          resp_options=self.resp_options,
                          charge_constraints=self.charge_constraints,
                          defer_errors=self.defer_errors,
                          temperature=self.temperature,
                          n_processes=self.n_processes,
                          working_directory=self.working_directory / "vacuum")

        self.solvated = Job(molecules=[x.copy(deep=True) for x in self.molecules],
                            qm_optimization_options=self.solvent_qm_optimization_options,
                            qm_esp_options=self.solvent_qm_esp_options,
                            grid_options=self.grid_options,
                            resp_options=self.resp_options,
                            charge_constraints=self.charge_constraints,
                            defer_errors=self.defer_errors,
                            temperature=self.temperature,
                            n_processes=self.n_processes,
                            working_directory=self.working_directory / "solvated")

    def run(self, client=None):
        self.vacuum.run(client=client, update_molecules=False)
        self.solvated.run(client=client, update_molecules=False)

    @property
    def charges(self):
        try:
            return self.get_charges()
        except ValueError:
            return None

    def get_charges(self, delta=0.6):
        if self.solvated.charges is None or self.vacuum.charges is None:
            raise ValueError("Neither `self.solvated.charges` "
                             "nor `self.vacuum.charges` should be `None`. "
                             "Perhaps you need to call `.run()` ?")

        return [
            solv * delta + (1 - delta) * vac
            for solv, vac in zip(self.solvated.charges, self.vacuum.charges)
        ]
