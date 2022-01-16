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

DMSO_STAGE_2_A = datapath("stage_2_constraints_a.dat")
DMSO_STAGE_2_B = datapath("stage_2_constraints_b.dat")

ESP_PATH = datapath("esps/*_esp.dat")
GRID_PATH = datapath("esps/*_grid.dat")

MANUAL_JOBS_WKDIR = datapath("manual_jobs/psiresp_working_directory")

DMSO_RESPA2_CHARGES = datapath("charges/dmso_c1_o2_respA2.dat")
DMSO_RESPA1_CHARGES = datapath("charges/dmso_c1_o2_respA1.dat")
DMSO_ESPA1_CHARGES = datapath("charges/dmso_c1_o2_espA1.dat")


DMSO_O1_ESP = datapath("dmso_opt_c1_o1.esp")
DMSO_O2_ESP = datapath("dmso_opt_c1_o2.esp")
DMSO_O1_RINV = datapath("dmso_opt_c1_o1_r_inv.dat")
DMSO_O2_RINV = datapath("dmso_opt_c1_o2_r_inv.dat")

AMM_NME_OPT_RESPA2_CHARGES = datapath("charges/amm_dimethyl_respA2.dat")
AMM_NME_OPT_RESPA1_CHARGES = datapath("charges/amm_dimethyl_respA1.dat")
AMM_NME_OPT_ESPA1_CHARGES = datapath("charges/amm_dimethyl_espA1.dat")

ETHANOL_RESP2_C1 = datapath("molecules/ethanol_resp2_opt_c1.xyz")
ETHANOL_RESP2_C2 = datapath("molecules/ethanol_resp2_opt_c2.xyz")

ETHANE_JSON = datapath("molecules/ethane.json")
