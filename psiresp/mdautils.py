import io

try:
    import MDAnalysis as mda
except ImportError:
    raise ImportError(
        "MDAnalysis is a necessary dependency for this functionality. "
        "Please install it with "
        "`conda install -c conda-forge mdanalysis`"
    )


def molecule_to_mdanalysis(molecule):
    xyzstr = molecule.qcmol.to_string(dtype="xyz")
    if not molecule.conformers:
        u = mda.Universe(io.StringIO(xyzstr), format="XYZ")
    else:
        conformers = [conf.coordinates for conf in molecule.conformers]
        u = mda.Universe(io.StringIO(xyzstr), conformers,
                         topology_format="XYZ")

    if molecule.charges is not None:
        u.add_TopologyAttr("charges", molecule.charges)
    return u
