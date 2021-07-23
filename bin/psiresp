#!/usr/bin/env python

import argparse
import os

import yaml
import psiresp

parser = argparse.ArgumentParser("Run a PsiRESP RESP job using a job file")
parser.add_argument("jobfile", type=str,
                    help="input job file")
parser.add_argument("-np", "--nprocs", default=1, type=int,
                    help="number of processes")
parser.add_argument("-nt", "--nthreads", default=1, type=int,
                    help="number of threads per process")
parser.add_argument("-mem", "--memory", default="500mb", type=str,
                    help="memory per process")
parser.add_argument("--write_all_charges", default=False,
                    action="store_true",
                    help=("Whether to write charges at different "
                          "stages of the fit. If True, the following "
                          "files may be written, depending on the "
                          "kind of RESP job: "
                          "stage_1_unrestrained, stage_1_restrained, "
                          "stage_2_unrestrained, stage_2_restrained"))


if __name__ == "__main__":
    args = parser.parse_args()
    run_options = vars(args)
    path = os.path.abspath(run_options.pop("jobfile"))
    with open(path, "r") as f:
        options = yaml.full_load(f)
    if "molecules" in options:
        cls = psiresp.MultiResp
    else:
        cls = psiresp.Resp
    obj = cls.from_yaml(path)

    obj.run(**run_options)
