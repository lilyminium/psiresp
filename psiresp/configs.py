from typing import Optional
from pydantic import Field

from .mixins import RespOptions
from .mixins.qm import QMMethod, QMBasisSet, QMSolvent
from .vdwradii import VdwRadiiSet
from .resp import Resp
from .multiresp import MultiResp
from .utils.due import due, Doi


class BaseRespConfig(RespOptions):
    qm_method: QMMethod = Field("hf", const=True)
    solvent: QMSolvent = Field(None, const=True)
    use_radii: VdwRadiiSet = Field("msk", const=True)
    ihfree: bool = Field(True, const=True)
    hyp_b: float = Field(0.1, const=True)


@due.dcite(
    Doi("10.1021/j100142a004"),
    description="RESP-A1 model",
    path="psiresp.configs",
)
class RespA1(BaseRespConfig, Resp):
    """RespA1 config"""
    qm_basis: QMBasisSet = "6-31g*"
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
    """RespA1 config"""
    qm_basis: QMBasisSet = "6-31g*"
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
    """RespA2 config"""
    qm_basis: QMBasisSet = "6-31g*"
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
    """RespA2 config"""
    qm_basis: QMBasisSet = "6-31g*"
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
    """EspA1 config"""
    qm_basis: QMBasisSet = "6-31g*"
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
    """EspA1 config"""
    qm_basis: QMBasisSet = "6-31g*"
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
    """EspA2 config"""
    qm_basis: QMBasisSet = "sto-3g"
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
    """EspA2 config"""
    qm_basis: QMBasisSet = "sto-3g"
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
    """ATBResp config"""
    qm_basis: QMBasisSet = "6-31g*"
    qm_method: QMMethod = "b3lyp"
    solvent: QMSolvent = "solvent"
    use_radii: VdwRadiiSet = "msk"
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
    """ATBResp config"""
    qm_basis: QMBasisSet = "6-31g*"
    qm_method: QMMethod = "b3lyp"
    solvent: QMSolvent = "solvent"
    use_radii: VdwRadiiSet = "msk"
    stage_2 = False
    hyp_a1 = 0.0
    hyp_a2 = 0.0
    hyp_b = 0.1
    restrained = False
    ihfree = False
