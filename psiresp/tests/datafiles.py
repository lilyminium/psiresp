from pkg_resources import resource_filename


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


DMSO_ESP = datapath("dmso_opt_c1.esp")
DMSO_RINV = datapath("dmso_opt_c1_r_inv.dat")
# DMSO_SHELL_D1 = datapath("dmso_shell_d1.npy")
# DMSO_SHELL_D2 = datapath("dmso_shell_d2.npy")

DMSO_STAGE_2_A = datapath("stage_2_constraints_a.dat")
DMSO_STAGE_2_B = datapath("stage_2_constraints_b.dat")

ESP_PATH = datapath("esps/*_esp.dat")
GRID_PATH = datapath("esps/*_grid.dat")

MANUAL_JOBS_WKDIR = datapath("manual_jobs/psiresp_working_directory")

DMSO_RESPA2_CHARGES = datapath("charges/dmso_c1_o2_respA2.dat")
DMSO_RESPA1_CHARGES = datapath("charges/dmso_c1_o2_respA1.dat")
DMSO_ESPA1_CHARGES = datapath("charges/dmso_c1_o2_espA1.dat")


# DMSO_QMRA = datapath("molecules/dmso_opt_c1_qmra.xyz")
# DMSO_QMRA_RESPA2_CHARGES = datapath("charges/dmso_c1_o0_respA2.dat")
# DMSO_QMRA_RESPA1_CHARGES = datapath("charges/dmso_c1_o0_respA1.dat")
# DMSO_QMRA_ESPA1_CHARGES = datapath("charges/dmso_c1_o0_espA1.dat")

# DMSO_O1 = datapath("molecules/dmso_opt_c1_o1.xyz")
# DMSO_O2 = datapath("molecules/dmso_opt_c1_o2.xyz")
# DMSO_O3 = datapath("molecules/dmso_opt_c1_o3.xyz")
# DMSO_O4 = datapath("molecules/dmso_opt_c1_o4.xyz")

# DMSO_TPL = datapath("molecules/dmso_orientations.tpl")
# DMSO_ORIENTATION_COORDINATES = datapath("molecules/dmso_orientations.npy")

DMSO_O1_ESP = datapath("dmso_opt_c1_o1.esp")
DMSO_O2_ESP = datapath("dmso_opt_c1_o2.esp")
DMSO_O1_RINV = datapath("dmso_opt_c1_o1_r_inv.dat")
DMSO_O2_RINV = datapath("dmso_opt_c1_o2_r_inv.dat")


# OPT_LOGFILE = datapath("opt_logfile.log")
# OPT_XYZFILE = datapath("opt_logfile.xyz")

# DMSO_PDB = datapath("molecules/dmso_opt_c1.pdb")
# DMSO_GRO = datapath("molecules/dmso_opt_c1.gro")


# NME2ALA2_C1 = datapath("molecules/nme2ala2_c1.xyz")
# NME2ALA2_C1_ABMAT = datapath("nme2ala2_abmat.dat")

# NME2ALA2_OPT_C1 = datapath("molecules/nme2ala2_opt_c1.xyz")
# NME2ALA2_OPT_C2 = datapath("molecules/nme2ala2_opt_c2.xyz")

# NME2ALA2_OPT_RESPA2_CHARGES = datapath("charges/nme2ala2_c2_o4_respA2.dat")
# NME2ALA2_OPT_RESPA1_CHARGES = datapath("charges/nme2ala2_c2_o4_respA1.dat")
# NME2ALA2_OPT_ESPA1_CHARGES = datapath("charges/nme2ala2_c2_o4_espA1.dat")

# METHYLAMMONIUM_OPT = datapath("molecules/methylammonium_opt_c1.xyz")

# ETHANOL_C1 = datapath("molecules/ethanol_opt_c1.xyz")
# ETHANOL_C2 = datapath("molecules/ethanol_opt_c2.xyz")
# ETHANOL_RESPA2_CHARGES = datapath("charges/ethanol_c2_o4_respA2.dat")
# ETHANOL_RESPA1_CHARGES = datapath("charges/ethanol_c2_o4_respA1.dat")
# ETHANOL_ESPA1_CHARGES = datapath("charges/ethanol_c2_o4_espA1.dat")

AMM_NME_OPT_RESPA2_CHARGES = datapath("charges/amm_dimethyl_respA2.dat")
AMM_NME_OPT_RESPA1_CHARGES = datapath("charges/amm_dimethyl_respA1.dat")
AMM_NME_OPT_ESPA1_CHARGES = datapath("charges/amm_dimethyl_espA1.dat")

ETHANOL_RESP2_C1 = datapath("molecules/ethanol_resp2_opt_c1.xyz")
ETHANOL_RESP2_C2 = datapath("molecules/ethanol_resp2_opt_c2.xyz")
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

# TEST_RESP_DATA = datapath("test_resp")
# TEST_MULTIRESP_DATA = datapath("test_multiresp")
# TEST_RESP2_DATA = datapath("test_resp2")
# TEST_MULTIRESP2_DATA = datapath("test_multiresp2")

# # ETOH_PDB = datapath("ethanol.pdb")
# # ETOH_MOL2 = datapath("ethanol.mol2")
# # ETOH_GRO = datapath("ethanol.gro")
# # ETOH_XYZ = datapath("ethanol.xyz")

# # ABMAT = datapath("nme2ala2_abmat.dat")

# # ETOH_RESP2_GAS_C1_GRID = datapath("test_resp2/resp2_ethanol_gas_c001_o001_grid.dat")

# # RESP_YAML = resource_filename()
