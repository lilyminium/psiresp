from pkg_resources import resource_filename

from .utils import datafile

ETOH_PDB = datafile("ethanol.pdb")
ETOH_MOL2 = datafile("ethanol.mol2")
ETOH_GRO = datafile("ethanol.gro")
ETOH_XYZ = datafile("ethanol.xyz")

ABMAT = datafile("nme2ala2_abmat.dat")

ETOH_RESP2_GAS_C1_GRID = resource_filename(__name__, "data/test_resp2/resp2_ethanol_gas_c001_o001_grid.dat")