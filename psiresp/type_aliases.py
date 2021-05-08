from typing import List, Tuple

from typing_extensions import Literal

Psi4Method = Literal["scf", "hf", "mp2", "mp3", "b3lyp", "m062x", "PW6B95"]

Psi4Basis = Literal["", "aug-cc-pV(D+d)Z", "6-31g*"]

VdwRadii = Literal["msk", "alvarez", "mantina", "bondi", "bondi_orig"]

AtomReorient = Tuple[int, int, int]

TranslateReorient = Tuple[float, float, float]