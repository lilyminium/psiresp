"""
psiresp
A RESP plugin for Psi4
"""

# Add imports here
from .orientation import Orientation
from .conformer import Conformer
from .resp import Resp
from .multiresp import MultiResp
from .resp2 import Resp2, MultiResp2
from .configs import (
    RespA1,
    RespA2,
    EspA1,
    EspA2,
    ATBResp,
    MultiRespA1,
    MultiRespA2,
    MultiEspA1,
    MultiEspA2,
)

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
