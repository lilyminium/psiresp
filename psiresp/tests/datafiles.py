from pkg_resources import resource_filename

# from .utils import datapath


def datapath(path):
    return resource_filename(__name__, f"data/{path}")


POSTGRES_SERVER_BACKUP = datapath("database.psql")

DMSO = datapath("molecules/dmso_opt_c1.xyz")
DMSO_O1 = datapath("molecules/dmso_opt_c1_o1.xyz")
DMSO_O2 = datapath("molecules/dmso_opt_c1_o2.xyz")
DMSO_O3 = datapath("molecules/dmso_opt_c1_o3.xyz")
DMSO_O4 = datapath("molecules/dmso_opt_c1_o4.xyz")

DMSO_GRID = datapath("surfaces/dmso_opt_c1_grid.npy")
DMSO_O1_GRID = datapath("surfaces/dmso_opt_c1_o1_grid.npy")
DMSO_O2_GRID = datapath("surfaces/dmso_opt_c1_o2_grid.npy")


METHYLAMMONIUM_OPT = datapath("molecules/methylammonium_opt_c1.xyz")

UNIT_SPHERE_3 = datapath("surfaces/surface_n3.dat")
UNIT_SPHERE_64 = datapath("surfaces/surface_n64.dat")

NME2ALA2_OPT_C1 = datapath("molecules/nme2ala2_opt_c1.xyz")
NME2ALA2_OPT_C2 = datapath("molecules/nme2ala2_opt_c2.xyz")


DMSO_ESP = resource_filename(__name__, "data/dmso_opt_c1.esp")
DMSO_RINV = resource_filename(__name__, "data/dmso_opt_c1_r_inv.dat")
# DMSO_SHELL_D1 = resource_filename(__name__, "data/dmso_shell_d1.npy")
# DMSO_SHELL_D2 = resource_filename(__name__, "data/dmso_shell_d2.npy")

DMSO_STAGE_2_A = resource_filename(__name__, "data/stage_2_constraints_a.dat")
DMSO_STAGE_2_B = resource_filename(__name__, "data/stage_2_constraints_b.dat")

ESP_PATH = resource_filename(__name__, "data/esps/*_esp.dat")
GRID_PATH = resource_filename(__name__, "data/esps/*_grid.dat")

MANUAL_JOBS_WKDIR = resource_filename(__name__, "data/manual_jobs/psiresp_working_directory")

DMSO_RESPA2_CHARGES = resource_filename(__name__, "data/charges/dmso_c1_o2_respA2.dat")
DMSO_RESPA1_CHARGES = resource_filename(__name__, "data/charges/dmso_c1_o2_respA1.dat")
DMSO_ESPA1_CHARGES = resource_filename(__name__, "data/charges/dmso_c1_o2_espA1.dat")


# DMSO_QMRA = resource_filename(__name__, "data/molecules/dmso_opt_c1_qmra.xyz")
# DMSO_QMRA_RESPA2_CHARGES = resource_filename(__name__, "data/charges/dmso_c1_o0_respA2.dat")
# DMSO_QMRA_RESPA1_CHARGES = resource_filename(__name__, "data/charges/dmso_c1_o0_respA1.dat")
# DMSO_QMRA_ESPA1_CHARGES = resource_filename(__name__, "data/charges/dmso_c1_o0_espA1.dat")

# DMSO_O1 = resource_filename(__name__, "data/molecules/dmso_opt_c1_o1.xyz")
# DMSO_O2 = resource_filename(__name__, "data/molecules/dmso_opt_c1_o2.xyz")
# DMSO_O3 = resource_filename(__name__, "data/molecules/dmso_opt_c1_o3.xyz")
# DMSO_O4 = resource_filename(__name__, "data/molecules/dmso_opt_c1_o4.xyz")

# DMSO_TPL = resource_filename(__name__, "data/molecules/dmso_orientations.tpl")
# DMSO_ORIENTATION_COORDINATES = resource_filename(__name__, "data/molecules/dmso_orientations.npy")

DMSO_O1_ESP = resource_filename(__name__, "data/dmso_opt_c1_o1.esp")
DMSO_O2_ESP = resource_filename(__name__, "data/dmso_opt_c1_o2.esp")
DMSO_O1_RINV = resource_filename(__name__, "data/dmso_opt_c1_o1_r_inv.dat")
DMSO_O2_RINV = resource_filename(__name__, "data/dmso_opt_c1_o2_r_inv.dat")


# OPT_LOGFILE = resource_filename(__name__, "data/opt_logfile.log")
# OPT_XYZFILE = resource_filename(__name__, "data/opt_logfile.xyz")

# DMSO_PDB = resource_filename(__name__, "data/molecules/dmso_opt_c1.pdb")
# DMSO_GRO = resource_filename(__name__, "data/molecules/dmso_opt_c1.gro")


# NME2ALA2_C1 = resource_filename(__name__, "data/molecules/nme2ala2_c1.xyz")
# NME2ALA2_C1_ABMAT = resource_filename(__name__, "data/nme2ala2_abmat.dat")

# NME2ALA2_OPT_C1 = resource_filename(__name__, "data/molecules/nme2ala2_opt_c1.xyz")
# NME2ALA2_OPT_C2 = resource_filename(__name__, "data/molecules/nme2ala2_opt_c2.xyz")

# NME2ALA2_OPT_RESPA2_CHARGES = resource_filename(__name__, "data/charges/nme2ala2_c2_o4_respA2.dat")
# NME2ALA2_OPT_RESPA1_CHARGES = resource_filename(__name__, "data/charges/nme2ala2_c2_o4_respA1.dat")
# NME2ALA2_OPT_ESPA1_CHARGES = resource_filename(__name__, "data/charges/nme2ala2_c2_o4_espA1.dat")

# METHYLAMMONIUM_OPT = resource_filename(__name__, "data/molecules/methylammonium_opt_c1.xyz")

# ETHANOL_C1 = resource_filename(__name__, "data/molecules/ethanol_opt_c1.xyz")
# ETHANOL_C2 = resource_filename(__name__, "data/molecules/ethanol_opt_c2.xyz")
# ETHANOL_RESPA2_CHARGES = resource_filename(__name__, "data/charges/ethanol_c2_o4_respA2.dat")
# ETHANOL_RESPA1_CHARGES = resource_filename(__name__, "data/charges/ethanol_c2_o4_respA1.dat")
# ETHANOL_ESPA1_CHARGES = resource_filename(__name__, "data/charges/ethanol_c2_o4_espA1.dat")

AMM_NME_OPT_RESPA2_CHARGES = resource_filename(__name__, "data/charges/amm_dimethyl_respA2.dat")
AMM_NME_OPT_RESPA1_CHARGES = resource_filename(__name__, "data/charges/amm_dimethyl_respA1.dat")
AMM_NME_OPT_ESPA1_CHARGES = resource_filename(__name__, "data/charges/amm_dimethyl_espA1.dat")

ETHANOL_RESP2_C1 = resource_filename(__name__, "data/molecules/ethanol_resp2_opt_c1.xyz")
ETHANOL_RESP2_C2 = resource_filename(__name__, "data/molecules/ethanol_resp2_opt_c2.xyz")
# ETHANOL_RESP2_GAS_C1_O1_GRID_ESP = resource_filename(__name__,
#                                                      "data/test_resp2/resp2_ethanol_gas/"
#                                                      "resp2_ethanol_gas_c001/"
#                                                      "resp2_ethanol_gas_c001_o001/"
#                                                      "resp2_ethanol_gas_c001_o001_grid_esp.dat")
# ETHANOL_RESP2_GAS_C1_O1_GRID = resource_filename(__name__,
#                                                  "data/test_resp2/resp2_ethanol_gas/"
#                                                  "resp2_ethanol_gas_c001/"
#                                                  "resp2_ethanol_gas_c001_o001/"
#                                                  "resp2_ethanol_gas_c001_o001_grid.dat")
# ETHANOL_RESP2_GAS_STAGE1_MATRICES = resource_filename(__name__,
#                                                       "data/test_resp2/resp2_ethanol_c2_gas_stg1_abmat.dat")
# ETHANOL_RESP2_GAS_C1_STAGE1_MATRICES = resource_filename(__name__,
#                                                          "data/test_resp2/resp2_ethanol_c2_gas_c1_abmat.dat")

# TEST_RESP_DATA = resource_filename(__name__, "data/test_resp")
# TEST_MULTIRESP_DATA = resource_filename(__name__, "data/test_multiresp")
# TEST_RESP2_DATA = resource_filename(__name__, "data/test_resp2")
# TEST_MULTIRESP2_DATA = resource_filename(__name__, "data/test_multiresp2")

# # ETOH_PDB = datapath("ethanol.pdb")
# # ETOH_MOL2 = datapath("ethanol.mol2")
# # ETOH_GRO = datapath("ethanol.gro")
# # ETOH_XYZ = datapath("ethanol.xyz")

# # ABMAT = datapath("nme2ala2_abmat.dat")

# # ETOH_RESP2_GAS_C1_GRID = resource_filename(__name__, "data/test_resp2/resp2_ethanol_gas_c001_o001_grid.dat")

# # RESP_YAML = resource_filename()
