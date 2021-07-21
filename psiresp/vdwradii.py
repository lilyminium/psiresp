from typing_extensions import Literal


MSK_RADII = {  # from FMOPRP in GAMESS fmoio.src
    'H': 1.20,                                                                      'He': 1.20,
    'Li': 1.37, 'Be': 1.45,  'B': 1.45,  'C': 1.50, 'N': 1.50, 'O': 1.40, 'F': 1.35, 'Ne': 1.30,
    'Na': 1.57, 'Mg': 1.36, 'Al': 1.24, 'Si': 1.17, 'P': 1.80, 'S': 1.75, 'Cl': 1.70
}

#: Bondi, A. van der Waals Volumes and Radii. J. Phys. Chem. 68, 441–451 (1964).
BONDI_RADII = {
    'H': 1.20,                                                                         'He': 1.40,
    'Li': 1.82,                          'C': 1.70,  'N': 1.55,  'O': 1.52,  'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73,             'Si': 2.10,  'P': 1.80,  'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    'K': 1.75,             'Ga': 1.87,             'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
    'In': 1.93, 'Sn': 2.17,             'Te': 2.06,  'I': 1.98, 'Xe': 2.16,
    'Tl': 1.96, 'Pb': 2.02,
    # other metals
    'Ni': 1.63, 'Cu': 1.4, 'Zn': 1.39,
    'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
    'Pt': 1.72, 'Au': 1.66, 'Hg': 1.55,
    'U': 1.86,
}

#: Mantina, M., Chamberlin, A. C., Valero, R., Cramer, C. J. & Truhlar, D. G. Consistent van der Waals Radii for the Whole Main Group. J Phys Chem A 113, 5806–5812 (2009).
MANTINA_RADII = {
    'H': 1.10,                                                                         'He': 1.40,
    'Li': 1.82, 'Be': 1.53,  'B': 1.92,  'C': 1.70,  'N': 1.55,  'O': 1.52,  'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10,  'P': 1.80,  'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    'K': 1.75, 'Ca': 2.31, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
    'Rb': 3.03, 'Sr': 2.49, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06,  'I': 1.98, 'Xe': 2.16,
    'Cs': 3.43, 'Ba': 2.68, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20,
    'Fr': 3.48, 'Ra': 2.83,
    # other metals
    'Ni': 1.63, 'Cu': 1.4, 'Zn': 1.39,
    'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
    'Pt': 1.72, 'Au': 1.66, 'Hg': 1.55,
    'U': 1.86,
}

HMANTINA_RADII = {  # H  is still from BONDI radii; follows GAMESS
    'H': 1.20,                                                                         'He': 1.40,
    'Li': 1.82, 'Be': 1.53,  'B': 1.92,  'C': 1.70,  'N': 1.55,  'O': 1.52,  'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10,  'P': 1.80,  'S': 1.80, 'Cl': 1.75, 'Ar': 1.88,
    'K': 1.75, 'Ca': 2.31, 'Ga': 1.87, 'Ge': 2.11, 'As': 1.85, 'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
    'Rb': 3.03, 'Sr': 2.49, 'In': 1.93, 'Sn': 2.17, 'Sb': 2.06, 'Te': 2.06,  'I': 1.98, 'Xe': 2.16,
    'Cs': 3.43, 'Ba': 2.68, 'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07, 'Po': 1.97, 'At': 2.02, 'Rn': 2.20,
    'Fr': 3.48, 'Ra': 2.83,
    # other metals
    'Ni': 1.63, 'Cu': 1.4, 'Zn': 1.39,
    'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58,
    'Pt': 1.72, 'Au': 1.66, 'Hg': 1.55,
    'U': 1.86,
}

#: Alvarez, S. A cartography of the van der Waals territories. Dalton Transactions 42, 8617–8636 (2013).
ALVAREZ_RADII = {
    'H': 1.20,                                                                         'He': 1.43,
    'Li': 2.12, 'Be': 1.98,  'B': 1.91,  'C': 1.77,  'N': 1.66,  'O': 1.50,  'F': 1.46, 'Ne': 1.58,
    'Na': 2.50, 'Mg': 2.51, 'Al': 2.25, 'Si': 2.19,  'P': 1.90,  'S': 1.89, 'Cl': 1.82, 'Ar': 1.83,
    'K': 2.73, 'Ca': 2.62, 'Ga': 2.32, 'Ge': 2.29, 'As': 1.88, 'Se': 1.82, 'Br': 1.86, 'Kr': 2.25,
    'Rb': 3.21, 'Sr': 2.84, 'In': 2.43, 'Sn': 2.42, 'Sb': 2.47, 'Te': 1.99,  'I': 2.04, 'Xe': 2.06,
    'Cs': 3.48, 'Ba': 3.03, 'Tl': 2.47, 'Pb': 2.60, 'Bi': 2.54,
    # other metals
    'Sc': 2.58, 'Ti': 2.46,  'V': 2.42, 'Cr': 2.45, 'Mn': 2.45, 'Fe': 2.44, 'Co': 2.40, 'Ni': 2.40, 'Cu': 2.38, 'Zn': 2.39,
    'Y': 2.75, 'Zr': 2.52, 'Nb': 2.56, 'Mo': 2.45, 'Tc': 2.44, 'Ru': 2.46, 'Rh': 2.44, 'Pd': 2.15, 'Ag': 2.53, 'Cd': 2.49,
    'La': 2.98, 'Ce': 2.88, 'Pr': 2.92, 'Nd': 2.95, 'Sm': 2.90, 'Eu': 2.87, 'Gd': 2.83,
    'Tb': 2.79, 'Dy': 2.87, 'Ho': 2.81, 'Er': 2.83, 'Tm': 2.79, 'Yb': 2.80, 'Lu': 2.74,
    'Hf': 2.63, 'Ta': 2.53,  'W': 2.57, 'Re': 2.49, 'Os': 2.48, 'Ir': 2.41, 'Pt': 2.29, 'Au': 2.32, 'Hg': 2.45,
    'Ac': 2.8, 'Th': 2.93, 'Pa': 2.88,  'U': 2.71, 'Np': 2.82, 'Pu': 2.81, 'Am': 2.83,
    'Cm': 3.05, 'Bk': 3.4, 'Cf': 3.05, 'Es': 2.7,
}

#: radii keywords and options
options = {
    'bondi_orig': BONDI_RADII,
    'bondi': HMANTINA_RADII,
    'mantina': MANTINA_RADII,
    'alvarez': ALVAREZ_RADII,
    'msk': MSK_RADII,
}

VdwRadiiSet = Literal[(*options,)]
