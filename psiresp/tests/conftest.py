from pkg_resources import resource_filename

from io import StringIO
import os
import re
import psi4
import numpy as np


def coordinates_from_xyz(file):
    return np.loadtxt(file skiprows=2, usecols=(1, 2, 3), comments='!')
