from .resp import Resp
from .multiresp import MultiResp
from .due import due, Doi


@due.dcite(
    Doi("10.1021/j100142a004"),
    description="RESP-A1 model",
    path="psiresp.configs",
)
class RespA1(Resp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.conformers:
            mol.qm_options.basis = "6-31g*"
            mol.qm_options.method = "scf"
            mol.qm_options.solvent = None
            mol.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=True,
                           hyp_a1=0.0005,
                           hyp_a2=0.001,
                           hyp_b=0.1,
                           restrained=True,
                           ihfree=True)


@due.dcite(
    Doi("10.1039/c0cp00111b"),
    description="RESP-A2 model",
    path="psiresp.configs",
)
class RespA2(Resp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.conformers:
            mol.qm_options.basis = "6-31g*"
            mol.qm_options.method = "scf"
            mol.qm_options.solvent = None
            mol.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.01,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=True,
                           ihfree=True)


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    description="ESP-A1 model",
    path="psiresp.configs",
)
class EspA1(Resp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.conformers:
            mol.qm_options.basis = "6-31g*"
            mol.qm_options.method = "scf"
            mol.qm_options.solvent = None
            mol.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.0,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=False,
                           ihfree=True)


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    Doi("10.1039/c0cp00111b"),
    description="ESP-A2 model",
    path="psiresp.configs",
)
class EspA2(Resp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.conformers:
            mol.qm_options.basis = "sto-3g"
            mol.qm_options.method = "scf"
            mol.qm_options.solvent = None
            mol.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.0,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=False,
                           ihfree=True)


@due.dcite(
    Doi("10.1021/ct200196m"),
    description="ATB model",
    path="psiresp.configs",
)
class ATBResp(Resp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.conformers:
            mol.qm_options.basis = "6-31g*"
            mol.qm_options.method = "b3lyp"
            mol.qm_options.solvent = "water"
            mol.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.0,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=False,
                           ihfree=False)


@due.dcite(
    Doi("10.1021/j100142a004"),
    description="RESP-A1 multi-molecule model",
    path="psiresp.configs",
)
class MultiRespA1(MultiResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.molecules:
            for conf in mol.conformers:
                conf.qm_options.basis = "6-31g*"
                conf.qm_options.method = "scf"
                conf.qm_options.solvent = None
                conf.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=True,
                           hyp_a1=0.0005,
                           hyp_a2=0.001,
                           hyp_b=0.1,
                           restrained=True,
                           ihfree=True)


@due.dcite(
    Doi("10.1039/c0cp00111b"),
    description="RESP-A2 multi-molecule model",
    path="psiresp.configs",
)
class MultiRespA2(MultiResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.molecules:
            for conf in mol.conformers:
                conf.qm_options.basis = "6-31g*"
                conf.qm_options.method = "scf"
                conf.qm_options.solvent = None
                conf.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.01,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=True,
                           ihfree=True)


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    description="ESP-A1 multi-molecule model",
    path="psiresp.configs",
)
class MultiEspA1(MultiResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.molecules:
            for conf in mol.conformers:
                conf.qm_options.basis = "6-31g*"
                conf.qm_options.method = "scf"
                conf.qm_options.solvent = None
                conf.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.0,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=False,
                           ihfree=True)


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    Doi("10.1039/c0cp00111b"),
    description="ESP-A2 multi-molecule model",
    path="psiresp.configs",
)
class MultiEspA2(MultiResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.molecules:
            for conf in mol.conformers:
                conf.qm_options.basis = "sto-3g"
                conf.qm_options.method = "scf"
                conf.qm_options.solvent = None
                conf.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.0,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=False,
                           ihfree=True)


@due.dcite(
    Doi("10.1021/ct200196m"),
    description="ATB multi-molecule model",
    path="psiresp.configs",
)
class ATBMultiResp(MultiResp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for mol in self.molecules:
            for conf in mol.conformers:
                conf.qm_options.basis = "6-31g*"
                conf.qm_options.method = "b3lyp"
                conf.qm_options.solvent = "water"
                conf.esp_options.use_radii = "msk"

    def run(self, executor=None, charge_constraint_options=None, tol: float = 1e-6, maxiter: int = 50):
        return super().run(executor=executor,
                           charge_constraint_options=charge_constraint_options,
                           tol=tol,
                           maxiter=maxiter,
                           stage_2=False,
                           hyp_a1=0.0,
                           hyp_a2=0.0,
                           hyp_b=0.1,
                           restrained=False,
                           ihfree=False)
