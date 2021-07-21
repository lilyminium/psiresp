"""
Creating command-line clients for running psiresp
"""

import argparse

# import psiresp

resp = argparse.ArgumentParser("Run a PsiRESP RESP from the command line")
resp.add_argument("molfile", type=str,
                  help=("Molecule file defining the molecule for calculating RESP. "
                        "Formats accepted include XYZ, PDB, MOL, MOL2, and GRO. "
                        "If multiple molecules are given in the file, "
                        "they are treated as conformers and all are kept."))
