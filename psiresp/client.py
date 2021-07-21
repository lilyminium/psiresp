"""
Creating command-line clients for running psiresp
"""

import argparse

import yaml
import psiresp

parser = argparse.ArgumentParser("Run a PsiRESP RESP from the command line")
parser.add_argument("jobfile", type=str,
                    help="input job file")
