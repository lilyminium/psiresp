from dataclasses import dataclass, field
from typing import List

from . import base
from .options import RespStageOptions, ChargeOptions

@dataclass
class RespStageCharges(base.ContainsOptionsBase):
    resp_stage_options: RespStageOptions = field(default_factory=RespStageOptions)
    charge_options: ChargeOptions = field(default_factory=ChargeOptions)
    symbols: List[str] = []

    def __post_init__(self):
        self._unrestrained_charges = None
        self._restrained_charges = None

    @property
    def restrained_charges(self):
        if self._restrained_charges is not None:
            return self._restrained_charges[:self.n_atoms]
    
    @property
    def unrestrained_charges(self):
        if self._unrestrained_charges is not None:
            return self._unrestrained_charges[:self.n_atoms]

    @property
    def charges(self):
        restrained = self.restrained_charges
        if restrained is None:
            return self.unrestrained_charges
        return restrained

        
    def fit(self, a_matrix, b_matrix):
        a, b = self.charge_options.get_constraint_matrix(a_matrix, b_matrix)
        q1 = self.resp_stage_options._solve_a_b(a, b)
        self._unrestrained_charges = q1
        if self.resp_stage_options.restrained:
            q2 = self.resp_stage_options.iter_solve(q1, self.symbols, a, b)
            self._restrained_charges = q2
        return self.charges


@dataclass
class RespCharges(base.MoleculeBase):
    charge_options: ChargeOptions = field(default_factory=ChargeOptions)
    resp_options: RespOptions = field(default_factory=RespOptions)

    def __post_init__(self):

        self.stage_1_charges = None
        self.stage_2_charges = None
    
    @property
    def charges(self):
        if self.stage_2_charges is not None:
            return self.stage_2_charges.charges
        
        values = self.stage_2_charges
        if values is None:
            return self.stage_1_charges
        return values
    

