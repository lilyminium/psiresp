import pathlib
from typing import List, Optional, ClassVar

from pydantic import Field

from . import base, qm, grid, molecule, resp, charge
from .job import Job
from .due import due, Doi
from .utils import update_dictionary


# def configure(**configuration):
#     def wrapper(cls):
#         class ConfiguredJob(Job):

#             def __init__(self, *args, **kwargs):
#                 obj = Job(*args, **kwargs)
#                 objdct = obj.dict()
#                 for option_name, option_config in configuration.items():
#                     prefix = option_name.split("_")[0] + "_"
#                     for field in objdct.keys():
#                         if field.startswith(prefix):
#                             for name, value in option_config.items():
#                                 update_dictionary(objdct[field], name, value)
#                 super().__init__(**objdct)

#         ConfiguredJob.__name__ = cls.__name__
#         ConfiguredJob.__doc__ = cls.__doc__
#         return ConfiguredJob

#     return wrapper


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
            if prefix == "resp_":
                print(objdct["resp_options"])
        super().__init__(**objdct)


@due.dcite(
    Doi("10.1021/j100142a004"),
    description="RESP-A1 model",
    path="psiresp.configs.RespA1",
)
class RespA1(ConfiguredJob):
    """RespA1 config"""
    _configuration = dict(qm_options=dict(method="hf",
                                          basis="6-31g*"
                                          ),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(resp_a1=0.0005,
                                            resp_a2=0.001,
                                            resp_b=0.1,
                                            stage_2=True,
                                            exclude_hydrogens=True,
                                            restrained_fit=True),
                          )


@due.dcite(
    Doi("10.1039/c0cp00111b"),
    description="RESP-A2 model",
    path="psiresp.configs.RespA2",
)
class RespA2(ConfiguredJob):
    """RespA2 config"""
    _configuration = dict(qm_options=dict(method="hf",
                                          basis="6-31g*"
                                          ),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(resp_a1=0.01,
                                            resp_a2=0.0,
                                            resp_b=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=True,
                                            restrained_fit=True),
                          )


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    description="ESP-A1 multimolecule model",
    path="psiresp.configs.EspA1",
)
class EspA1(ConfiguredJob):
    """EspA1 config"""
    _configuration = dict(qm_options=dict(method="hf",
                                          basis="6-31g*"
                                          ),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(resp_a1=0.0,
                                            resp_a2=0.0,
                                            resp_b=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=True,
                                            restrained_fit=False),
                          )


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    Doi("10.1039/c0cp00111b"),
    description="ESP-A2 model",
    path="psiresp.configs.EspA2",
)
class EspA2(ConfiguredJob):
    """EspA2 config"""

    _configuration = dict(qm_options=dict(method="hf",
                                          basis="sto-3g"
                                          ),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(resp_a1=0.0,
                                            resp_a2=0.0,
                                            resp_b=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=True,
                                            restrained_fit=False),
                          )


@due.dcite(
    Doi("10.1021/ct200196m"),
    description="ATB model",
    path="psiresp.configs.ATBResp",
)
class ATBResp(ConfiguredJob):
    """ATB Resp config"""
    _configuration = dict(qm_options=dict(method="b3lyp",
                                          basis="6-31g*",
                                          pcm_options=dict(
                                              solvent="water"
                                          )),
                          grid_options=dict(use_radii="msk"),
                          resp_options=dict(resp_a1=0.0,
                                            resp_a2=0.0,
                                            resp_b=0.1,
                                            stage_2=False,
                                            exclude_hydrogens=False,
                                            restrained_fit=False),
                          )


@ due.dcite(
    Doi("10.1038/s42004-020-0291-4"),
    description="RESP2",
    path="psiresp.resp2",
)
class Resp2(base.Model):
    """Class to manage Resp2 jobs"""
    molecules: List[molecule.Molecule] = Field(
        default_factory=list,
        description="Molecules to use for the RESP job"
    )
    solvent_qm_optimization_options: qm.QMGeometryOptimizationOptions = Field(
        default=qm.QMGeometryOptimizationOptions(),
        description="QM options for geometry optimization"
    )
    solvent_qm_esp_options: qm.QMEnergyOptions = Field(
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

    n_processes: Optional[int] = Field(
        default=None,
        description=("Number of processes to use in multiprocessing "
                     "during ESP computation. `n_processes=None` uses "
                     "the number of CPUs.")
    )

    vacuum: Optional[Job] = None
    solvent: Optional[Job] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vacuum_opt = self.solvent_qm_optimization_options.copy(deep=True)
        vacuum_opt.pcm_options = None
        vacuum_esp = self.solvent_qm_esp_options.copy(deep=True)
        vacuum_esp.pcm_options = None

        self.vacuum = Job(molecules=self.molecules,
                          qm_optimization_options=vacuum_opt,
                          qm_esp_options=vacuum_esp,
                          grid_options=self.grid_options,
                          resp_options=self.resp_options,
                          charge_constraints=self.charge_constraints,
                          defer_errors=self.defer_errors,
                          temperature=self.temperature,
                          n_processes=self.n_processes,
                          working_directory=self.working_directory / "vacuum")

        self.solvated = Job(molecules=self.molecules,
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

    @ property
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

        return delta * self.solvated.charges + (1 - delta) * self.vacuum.charges


# @due.dcite(
#     Doi("10.1021/j100142a004"),
#     description="RESP-A1 model",
#     path="psiresp.configs.RespA1",
# )
# @configure(qm_options=dict(method="hf",
#                            basis="6-31g*"
#                            ),
#            grid_options=dict(use_radii="msk"),
#            resp_options=dict(resp_a1=0.0005,
#                              resp_a2=0.001,
#                              resp_b=0.1,
#                              stage_2=True,
#                              exclude_hydrogens=True,
#                              restrained_fit=True),
#            )
# class RespA1(Job):
#     """RespA1 config"""


# @due.dcite(
#     Doi("10.1039/c0cp00111b"),
#     description="RESP-A2 model",
#     path="psiresp.configs.RespA2",
# )
# @configure(qm_options=dict(method="hf",
#                            basis="6-31g*"
#                            ),
#            grid_options=dict(use_radii="msk"),
#            resp_options=dict(resp_a1=0.01,
#                              resp_a2=0.0,
#                              resp_b=0.1,
#                              stage_2=False,
#                              exclude_hydrogens=True,
#                              restrained_fit=True),
#            )
# class RespA2(Job):
#     """RespA2 config"""


# @due.dcite(
#     Doi("10.1002/jcc.540050204"),
#     description="ESP-A1 multimolecule model",
#     path="psiresp.configs.EspA1",
# )
# @configure(qm_options=dict(method="hf",
#                            basis="6-31g*"
#                            ),
#            grid_options=dict(use_radii="msk"),
#            resp_options=dict(resp_a1=0.0,
#                              resp_a2=0.0,
#                              resp_b=0.1,
#                              stage_2=False,
#                              exclude_hydrogens=True,
#                              restrained_fit=False),
#            )
# class EspA1(Job):
#     """EspA1 config"""


# @due.dcite(
#     Doi("10.1002/jcc.540050204"),
#     Doi("10.1039/c0cp00111b"),
#     description="ESP-A2 model",
#     path="psiresp.configs.EspA2",
# )
# @configure(qm_options=dict(method="hf",
#                            basis="sto-3g"
#                            ),
#            grid_options=dict(use_radii="msk"),
#            resp_options=dict(resp_a1=0.0,
#                              resp_a2=0.0,
#                              resp_b=0.1,
#                              stage_2=False,
#                              exclude_hydrogens=True,
#                              restrained_fit=False),
#            )
# class EspA2(Job):
#     """EspA2 config"""


# @due.dcite(
#     Doi("10.1021/ct200196m"),
#     description="ATB model",
#     path="psiresp.configs.ATBResp",
# )
# @configure(qm_options=dict(method="b3lyp",
#                            basis="6-31g*",
#                            pcm_options=dict(
#                                solvent="water"
#                            )),
#            grid_options=dict(use_radii="msk"),
#            resp_options=dict(resp_a1=0.0,
#                              resp_a2=0.0,
#                              resp_b=0.1,
#                              stage_2=False,
#                              exclude_hydrogens=False,
#                              restrained_fit=False),
#            )
# class ATBResp(Job):
#     """ATB Resp config"""


# @due.dcite(
#     Doi("10.1038/s42004-020-0291-4"),
#     description="RESP2",
#     path="psiresp.resp2",
# )
# class Resp2(base.Model):
#     """Class to manage Resp2 jobs"""
#     molecules: List[molecule.Molecule] = Field(
#         default_factory=list,
#         description="Molecules to use for the RESP job"
#     )
#     solvent_qm_optimization_options: qm.QMGeometryOptimizationOptions = Field(
#         default=qm.QMGeometryOptimizationOptions(),
#         description="QM options for geometry optimization"
#     )
#     solvent_qm_esp_options: qm.QMEnergyOptions = Field(
#         default=qm.QMEnergyOptions(),
#         description="QM options for ESP computation"
#     )
#     grid_options: grid.GridOptions = Field(
#         default=grid.GridOptions(),
#         description="Options for generating grid for ESP computation"
#     )
#     resp_options: resp.RespOptions = Field(
#         default=resp.RespOptions(),
#         description="Options for fitting ESP for charges"
#     )
#     charge_constraints: charge.ChargeConstraintOptions = Field(
#         default=charge.ChargeConstraintOptions(),
#         description="Charge constraints"
#     )

#     working_directory: pathlib.Path = Field(
#         default=pathlib.Path("psiresp_working_directory"),
#         description="Working directory for saving intermediate files"
#     )

#     defer_errors: bool = Field(
#         default=False,
#         description=("Whether to raise an error immediately, "
#                      "or gather all errors during ESP computation "
#                      "and raise at the end")
#     )
#     temperature: float = Field(
#         default=298.15,
#         description="Temperature (in Kelvin) to use when Boltzmann-weighting conformers."
#     )

#     n_processes: Optional[int] = Field(
#         default=None,
#         description=("Number of processes to use in multiprocessing "
#                      "during ESP computation. `n_processes=None` uses "
#                      "the number of CPUs.")
#     )

#     vacuum: Optional[Job] = None
#     solvent: Optional[Job] = None

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         vacuum_opt = self.solvent_qm_optimization_options.copy(deep=True)
#         vacuum_opt.pcm_options = None
#         vacuum_esp = self.solvent_qm_esp_options.copy(deep=True)
#         vacuum_esp.pcm_options = None

#         self.vacuum = Job(molecules=self.molecules,
#                           qm_optimization_options=vacuum_opt,
#                           qm_esp_options=vacuum_esp,
#                           grid_options=self.grid_options,
#                           resp_options=self.resp_options,
#                           charge_constraints=self.charge_constraints,
#                           defer_errors=self.defer_errors,
#                           temperature=self.temperature,
#                           n_processes=self.n_processes,
#                           working_directory=self.working_directory / "vacuum")

#         self.solvent = Job(molecules=self.molecules,
#                            qm_optimization_options=self.solvent_qm_optimization_options,
#                            qm_esp_options=self.solvent_qm_esp_options,
#                            grid_options=self.grid_options,
#                            resp_options=self.resp_options,
#                            charge_constraints=self.charge_constraints,
#                            defer_errors=self.defer_errors,
#                            temperature=self.temperature,
#                            n_processes=self.n_processes,
#                            working_directory=self.working_directory / "solvent")

#     def run(self, client=None):
#         self.vacuum.run(client=client, update_molecules=False)
#         self.solvent.run(client=client, update_molecules=False)

#     @property
#     def charges(self):
#         try:
#             return self.get_charges()
#         except ValueError:
#             return None

#     def get_charges(self, delta=0.6):
#         if self.solvent.charges is None or self.vacuum.charges is None:
#             raise ValueError("Neither `self.solvent.charges` "
#                              "nor `self.vacuum.charges` should be `None`. "
#                              "Perhaps you need to call `.run()` ?")

#         return delta * self.solvent.charges + (1 - delta) * self.vacuum.charges
