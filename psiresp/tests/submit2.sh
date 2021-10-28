#! /usr/bin/bash
#SBATCH --partition=free
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4gb
#SBATCH --account lilyw7
#SBATCH --export ALL
#SBATCH --requeue
#SBATCH --output=%x.log-%A_%a


. "/data/homezvol0/lilyw7/miniconda3/etc/profile.d/conda.sh"
conda activate psiresp-3.8

pytest test_grid.py
pytest test_m*.py
pytest test_o*.py
pytest test_rdutils.py