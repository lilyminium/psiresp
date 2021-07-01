from .resp import Resp, RespBase
from .multiresp import MultiResp
from .utils.due import due, Doi


class BaseRespConfig(RespBase):
    qm_method = "scf"
    solvent = None
    use_radii = "msk"
    ihfree = True
    hyp_b = 0.1


@due.dcite(
    Doi("10.1021/j100142a004"),
    description="RESP-A1 model",
    path="psiresp.configs",
)
class RespA1(BaseRespConfig, Resp):
    qm_basis = "6-31g*"
    stage_2 = True
    hyp_a1 = 0.0005
    hyp_a2 = 0.001
    restrained = True


@due.dcite(
    Doi("10.1021/j100142a004"),
    description="RESP-A1 multi-molecule model",
    path="psiresp.configs",
)
class MultiRespA1(BaseRespConfig, MultiResp):
    qm_basis = "6-31g*"
    stage_2 = True
    hyp_a1 = 0.0005
    hyp_a2 = 0.001
    restrained = True


@due.dcite(
    Doi("10.1039/c0cp00111b"),
    description="RESP-A2 model",
    path="psiresp.configs",
)
class RespA2(BaseRespConfig, Resp):
    qm_basis = "6-31g*"
    stage_2 = False
    hyp_a1 = 0.01
    hyp_a2 = 0.0
    restrained = True


@due.dcite(
    Doi("10.1039/c0cp00111b"),
    description="RESP-A2 multimolecule model",
    path="psiresp.configs",
)
class MultiRespA2(BaseRespConfig, MultiResp):
    qm_basis = "6-31g*"
    stage_2 = False
    hyp_a1 = 0.01
    hyp_a2 = 0.0
    restrained = True


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    description="ESP-A1 model",
    path="psiresp.configs",
)
class EspA1(BaseRespConfig, Resp):
    qm_basis = "6-31g*"
    stage_2 = False
    hyp_a1 = 0.0
    hyp_a2 = 0.0
    restrained = False


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    description="ESP-A1 multimolecule model",
    path="psiresp.configs",
)
class MultiEspA1(BaseRespConfig, MultiResp):
    qm_basis = "6-31g*"
    stage_2 = False
    hyp_a1 = 0.0
    hyp_a2 = 0.0
    restrained = False


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    Doi("10.1039/c0cp00111b"),
    description="ESP-A2 model",
    path="psiresp.configs",
)
class EspA2(BaseRespConfig, Resp):
    qm_basis = "sto-3g"
    stage_2 = False
    hyp_a1 = 0.0
    hyp_a2 = 0.0
    restrained = False


@due.dcite(
    Doi("10.1002/jcc.540050204"),
    Doi("10.1039/c0cp00111b"),
    description="ESP-A2 multimolecule model",
    path="psiresp.configs",
)
class MultiEspA2(BaseRespConfig, MultiResp):
    qm_basis = "sto-3g"
    stage_2 = False
    hyp_a1 = 0.0
    hyp_a2 = 0.0
    restrained = False


@due.dcite(
    Doi("10.1021/ct200196m"),
    description="ATB model",
    path="psiresp.configs",
)
class ATBResp(Resp):
    qm_basis = "6-31g*"
    qm_method = "b3lyp"
    solvent = "solvent"
    use_radii = "msk"
    stage_2 = False
    hyp_a1 = 0.0
    hyp_a2 = 0.0
    hyp_b = 0.1
    restrained = False
    ihfree = False


@due.dcite(
    Doi("10.1021/ct200196m"),
    description="ATB multimolecule model",
    path="psiresp.configs",
)
class MultiATBResp(MultiResp):
    qm_basis = "6-31g*"
    qm_method = "b3lyp"
    solvent = "solvent"
    use_radii = "msk"
    stage_2 = False
    hyp_a1 = 0.0
    hyp_a2 = 0.0
    hyp_b = 0.1
    restrained = False
    ihfree = False
