import functools

from .resp import Resp
from .multiresp import MultiResp
from .due import due, Doi


def clean_kwargs(func):
    kws = ('stage_2', 'hyp_a1', 'hyp_a2', 'hyp_b', 'restraint', 'basis', 'method', 'ihfree',
           'use_radii', 'solvent')

    def wrapper(*args, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in kws}
        return func(*args, **kwargs)
    return wrapper


def resp_config(stage_2=True, hyp_a1=0.0005, hyp_a2=0.001,
                hyp_b=0.1, restraint=True, use_radii='msk',
                basis='6-31g*', method='scf', ihfree=True,
                solvent=None):
    """Make analogous classes to R.E.D. options."""
    def wrapper(cls):
        @functools.wraps(cls)
        class Config(cls):
            @clean_kwargs
            def run(self, opt=False, save_opt_geometry=False,
                    chrconstr=[], chrequiv=[],
                    psi4_options={}, n_orient=0, orient=[],
                    n_rotate=0, rotate=[], n_translate=0, translate=[],
                    equal_methyls=False, **kwargs):
                """
                Perform a Resp fit.

                Parameters
                ----------
                opt: bool (optional)
                    Whether to optimise the geometry of each conformer
                save_opt_geometry: bool (optional)
                    if ``True``, writes optimised geometries to an XYZ file
                chrconstr: list or dict (optional)
                    Intramolecular charge constraints in the form of
                    {charge: atom_number_list} or [(charge, atom_number_list)].
                    The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
                    mean that atoms 1 and 2 together have a charge of 0.
                chrequiv: list (optional)
                    lists of atoms with equivalent charges.
                    e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
                    charges, and atoms 3, 4, and 5 have equal charges.
                psi4_options: dict (optional)
                    additional Psi4 options
                n_orient: int (optional)
                    If this is greater than 0 and ``orient`` is not given,
                    ``n_orient`` orientations are automatically generated for the
                    molcule. Heavy atoms are prioritised.
                orient: list of tuples of ints (optional)
                    List of reorientations. Corresponds to REMARK REORIENT in R.E.D.
                    e.g. [(1, 5, 9), (9, 5, 1)] creates two reorientations: the first
                    around the first, fifth and ninth atom; and the second in reverse
                    order.
                n_rotate: int (optional)
                    If this is greater than 0 and ``rotate`` is not given,
                    ``n_rotate`` rotations are automatically generated for the
                    molecule. Heavy atoms are prioritised.
                rotate: list of tuples of ints (optional)
                    List of rotations. Corresponds to REMARK ROTATE in R.E.D.
                    e.g. [(1, 5, 9), (9, 5, 1)] creates two rotations: the first
                    around the first, fifth and ninth atom; and the second in reverse
                    order.
                n_translate: int (optional)
                    If this is greater than 0 and ``translate`` is not given,
                    ``n_translate`` translations are randomly generated for the
                    molcule in the domain [0, 1).
                translate: list of tuples of floats (optional)
                    List of translations. Corresponds to REMARK TRANSLATE in R.E.D.
                    e.g. [(1.0, 0, -0.5)] creates a translation that adds 1.0 to the
                    x coordinates, 0 to the y coordinates, and -0.5 to the z coordinates.
                equal_methyls: bool (optional)
                    if ``True``, all carbons in methyl groups are constrained to be
                    equivalent; all carbons in methylenes are equivalent; all hydrogens
                    in methyls are equivalent; and all hydrogens in methylenes are
                    equivalent. Ignored if ``chrequiv`` constraints are provided.
                **kwargs:
                    arguments passed to Resp.fit

                Returns
                -------
                charges: ndarray
                """
                return super(Config, self).run(opt=opt, save_opt_geometry=save_opt_geometry,
                                            chrconstr=chrconstr, chrequiv=chrequiv,
                                            psi4_options=psi4_options,
                                            n_orient=n_orient, orient=orient,
                                            n_rotate=n_rotate, rotate=rotate,
                                            n_translate=n_translate, translate=translate,
                                            equal_methyls=equal_methyls, stage_2=stage_2,
                                            hyp_a1=hyp_a1, hyp_a2=hyp_a2, hyp_b=hyp_b,
                                            restraint=restraint, basis=basis,
                                            method=method, ihfree=ihfree,
                                            solvent=solvent, **kwargs)

            @clean_kwargs
            def optimize_geometry(self, psi4_options={}, save_opt_geometry=False, **kwargs):
                """
                Optimise geometry for all conformers.

                Parameters
                ----------
                psi4_options: dict (optional)
                    additional Psi4 options
                save_opt_geometry: bool (optional)
                    if ``True``, saves optimised geometries to XYZ file
                """
                return super(Config, self).optimize_geometry(method=method, basis=basis,
                                                            psi4_options=psi4_options, save_opt_geometry=save_opt_geometry)

            @clean_kwargs
            def fit(self, hyp_a=0.0005, tol=1e-6, maxiter=50, **kwargs):
                """
                Perform the Resp fits.

                Parameters
                ----------
                tol: float (optional)
                    threshold for convergence
                maxiter: int (optional)
                    maximum number of iterations
                **kwargs:
                    arguments passed to Resp.get_constraint_matrices

                Returns
                -------
                charges: ndarray
                """
                return super(Config, self).fit(hyp_a=hyp_a, hyp_b=hyp_b, restraint=restraint, basis=basis,
                                            method=method, ihfree=ihfree, **kwargs)

            @clean_kwargs
            def iter_solve(self, q, a, b, hyp_a=0.0005, tol=1e-6, maxiter=50, **kwargs):
                """
                Fit the charges iteratively, as required for the hyperbola penalty
                function.

                Parameters
                ----------
                q: ndarray
                    partial atomic charges
                a: ndarray
                    unrestrained matrix A
                b: ndarray
                    matrix B
                hyp_a: float (optional)
                    scale factor of asymptote limits of hyperbola
                tol: float (optional)
                    threshold for convergence
                maxiter: int (optional)
                    maximum number of iterations

                Returns
                -------
                charges: ndarray
                """

                return super(Config, self).iter_solve(q, a, b, hyp_a=hyp_a,
                                                    hyp_b=hyp_b,
                                                    ihfree=ihfree,
                                                    tol=tol,
                                                    maxiter=maxiter)

            @clean_kwargs
            def get_constraint_matrices(self, chrconstr=[], chrequiv=[],
                                        weights=1,
                                        vdw_scale_factors=(1.4, 1.6, 1.8, 2.0),
                                        vdw_point_density=1.0, vdw_radii={},
                                        rmin=0, rmax=-1,
                                        psi4_options={}, save_files=False, **kwargs):
                """
                Get A and B matrices to solve for charges, including charge constraints.

                Parameters
                ----------
                chrconstr: list or dict (optional)
                    Intramolecular charge constraints in the form of
                    {charge: atom_number_list} or [(charge, atom_number_list)].
                    The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
                    mean that atoms 1 and 2 together have a charge of 0.
                chrequiv: list (optional)
                    lists of atoms with equivalent charges.
                    e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
                    charges, and atoms 3, 4, and 5 have equal charges.
                weights: iterable of ints (optional)
                    weights of each conformer
                vdw_scale_factors: iterable of floats (optional)
                    scale factors
                vdw_point_density: float (optional)
                    point density
                use_radii: str (optional)
                    which set of van der Waals' radii to use
                vdw_radii: dict (optional)
                    van der Waals' radii. If elements in the molecule are not
                    defined in the chosen ``use_radii`` set, they must be given here.
                rmin: float (optional)
                    inner boundary of shell to keep grid points from
                rmax: float (optional)
                    outer boundary of shell to keep grid points from. If < 0,
                    all points are selected.
                psi4_options: dict (optional)
                    additional Psi4 options
                save_files: bool (optional)
                    if ``True``, Psi4 files are saved and the computed Esp
                    and grids are written to files.

                Returns
                -------
                a: ndarray
                b: ndarray
                """
                return super(Config, self).get_constraint_matrices(method=method, basis=basis,
                                                                chrconstr=chrconstr,
                                                                chrequiv=chrequiv,
                                                                weights=weights, use_radii=use_radii,
                                                                vdw_point_density=vdw_point_density,
                                                                vdw_scale_factors=vdw_scale_factors,
                                                                vdw_radii=vdw_radii, rmin=rmin,
                                                                rmax=rmax, solvent=solvent,
                                                                psi4_options=psi4_options,
                                                                save_files=save_files)
        return Config
    return wrapper


@due.dcite(
    Doi('10.1021/j100142a004'),
    description='RESP-A1 model',
    path='psiresp.configs',
)
@resp_config(stage_2=True, hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1,
             restraint=True, basis='6-31g*', method='scf',
             ihfree=True, use_radii='msk', solvent=None)
class RespA1(Resp):
    """
    Class to manage a 2-stage RESP fit for one 
    molecule of multiple conformers. This is called the RESP-A1 
    model in R.E.D.

    Please cite [Bayly1993]_ if you use this class.

    Parameters
    ----------
    conformers: iterable of Conformers
        conformers of the molecule. They must all have the same atoms
        in the same order.
    name: str (optional)
        name of the molecule. This is used to name output files. If not
        given, defaults to 'Mol'.
    chrconstr: list or dict (optional)
        Intramolecular charge constraints in the form of
        {charge: atom_number_list} or [(charge, atom_number_list)].
        The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
        mean that atoms 1 and 2 together have a charge of 0.
    chrequiv: list (optional)
        lists of atoms with equivalent charges, indexed from 1.
        e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
        charges, and atoms 3, 4, and 5 have equal charges.

    Attributes
    ----------
    name: str
        name of the molecule. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    indices: ndarray of ints
        indices of each atom, indexed from 0
    heavy_ids: ndarray of ints
        atom numbers of heavy atoms, indexed from 1
    h_ids: ndarray of ints
        atom numbers of hydrogen atoms, indexed from 1
    symbols: ndarray
        element symbols
    charge: int or float
        overall charge of molecule
    conformers: list of Conformers
    n_conf: int
        number of conformers
    unrestrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    restrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1039/c0cp00111b'),
    description='RESP-A2 model',
    path='psiresp.configs',
)
@resp_config(stage_2=False, hyp_a1=0.01, hyp_a2=0.0, hyp_b=0.1,
             restraint=True, basis='6-31g*', method='scf',
             ihfree=True, use_radii='msk', solvent=None)
class RespA2(Resp):
    """
    Class to manage a 1-stage RESP fit for one 
    molecule of multiple conformers. This is called the RESP-A2
    model in R.E.D.

    Please cite [Dupradeau2010]_ if you use this class.

    Parameters
    ----------
    conformers: iterable of Conformers
        conformers of the molecule. They must all have the same atoms
        in the same order.
    name: str (optional)
        name of the molecule. This is used to name output files. If not
        given, defaults to 'Mol'.
    chrconstr: list or dict (optional)
        Intramolecular charge constraints in the form of
        {charge: atom_number_list} or [(charge, atom_number_list)].
        The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
        mean that atoms 1 and 2 together have a charge of 0.
    chrequiv: list (optional)
        lists of atoms with equivalent charges, indexed from 1.
        e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
        charges, and atoms 3, 4, and 5 have equal charges.

    Attributes
    ----------
    name: str
        name of the molecule. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    indices: ndarray of ints
        indices of each atom, indexed from 0
    heavy_ids: ndarray of ints
        atom numbers of heavy atoms, indexed from 1
    h_ids: ndarray of ints
        atom numbers of hydrogen atoms, indexed from 1
    symbols: ndarray
        element symbols
    charge: int or float
        overall charge of molecule
    conformers: list of Conformers
    n_conf: int
        number of conformers
    unrestrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    restrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1002/jcc.540050204'),
    description='ESP-A1 model',
    path='psiresp.configs',
)
@resp_config(stage_2=False, hyp_a1=0.0, hyp_a2=0.0, hyp_b=0.1,
             restraint=False, basis='6-31g*', method='scf',
             ihfree=True, use_radii='msk', solvent=None)
class EspA1(Resp):
    """
    Class to manage a 1-stage unrestrained fit for one 
    molecule of multiple conformers. This is called the ESP-A1
    model in R.E.D.

    Please cite [Singh1984]_ if you use this class.

    Parameters
    ----------
    conformers: iterable of Conformers
        conformers of the molecule. They must all have the same atoms
        in the same order.
    name: str (optional)
        name of the molecule. This is used to name output files. If not
        given, defaults to 'Mol'.
    chrconstr: list or dict (optional)
        Intramolecular charge constraints in the form of
        {charge: atom_number_list} or [(charge, atom_number_list)].
        The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
        mean that atoms 1 and 2 together have a charge of 0.
    chrequiv: list (optional)
        lists of atoms with equivalent charges, indexed from 1.
        e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
        charges, and atoms 3, 4, and 5 have equal charges.

    Attributes
    ----------
    name: str
        name of the molecule. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    indices: ndarray of ints
        indices of each atom, indexed from 0
    heavy_ids: ndarray of ints
        atom numbers of heavy atoms, indexed from 1
    h_ids: ndarray of ints
        atom numbers of hydrogen atoms, indexed from 1
    symbols: ndarray
        element symbols
    charge: int or float
        overall charge of molecule
    conformers: list of Conformers
    n_conf: int
        number of conformers
    unrestrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1002/jcc.540050204'),
    Doi('10.1039/c0cp00111b'),
    description='ESP-A2 model',
    path='psiresp.configs',
)
@resp_config(stage_2=False, hyp_a1=0.0, hyp_a2=0.0, hyp_b=0.1,
             restraint=False, basis='sto-3g', method='scf',
             ihfree=True, use_radii='msk', solvent=None)
class EspA2(Resp):
    """
    Class to manage a 1-stage unrestrained fit for one 
    molecule of multiple conformers, using HF/STO-3G. 
    This is called the ESP-A2 model in R.E.D.

    Please cite [Singh1984]_ and [Dupradeau2010]_ if you use this class.

    Parameters
    ----------
    conformers: iterable of Conformers
        conformers of the molecule. They must all have the same atoms
        in the same order.
    name: str (optional)
        name of the molecule. This is used to name output files. If not
        given, defaults to 'Mol'.
    chrconstr: list or dict (optional)
        Intramolecular charge constraints in the form of
        {charge: atom_number_list} or [(charge, atom_number_list)].
        The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
        mean that atoms 1 and 2 together have a charge of 0.
    chrequiv: list (optional)
        lists of atoms with equivalent charges, indexed from 1.
        e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
        charges, and atoms 3, 4, and 5 have equal charges.

    Attributes
    ----------
    name: str
        name of the molecule. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    indices: ndarray of ints
        indices of each atom, indexed from 0
    heavy_ids: ndarray of ints
        atom numbers of heavy atoms, indexed from 1
    h_ids: ndarray of ints
        atom numbers of hydrogen atoms, indexed from 1
    symbols: ndarray
        element symbols
    charge: int or float
        overall charge of molecule
    conformers: list of Conformers
    n_conf: int
        number of conformers
    unrestrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1021/ct200196m'),
    description='ATB model',
    path='psiresp.configs',
)
#: TODO: autodetect symmetry?
@resp_config(stage_2=False, hyp_a1=0.0, hyp_a2=0.0, hyp_b=0.1,
             restraint=False, basis='6-31g*', method='b3lyp',
             ihfree=False, use_radii='msk', solvent='water')
class ATBResp(Resp):
    """
    Class to manage a 1-stage unrestrained fit using the method of 
    Singh and Kollman 1984 at B3LYP/6-31G* with implicit solvent, as 
    described for the Automated Topology Builder.

    Please cite [Malde2011]_ and [Singh1984]_ if you use this class.

    Parameters
    ----------
    conformers: iterable of Conformers
        conformers of the molecule. They must all have the same atoms
        in the same order.
    name: str (optional)
        name of the molecule. This is used to name output files. If not
        given, defaults to 'Mol'.
    chrconstr: list or dict (optional)
        Intramolecular charge constraints in the form of
        {charge: atom_number_list} or [(charge, atom_number_list)].
        The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
        mean that atoms 1 and 2 together have a charge of 0.
    chrequiv: list (optional)
        lists of atoms with equivalent charges, indexed from 1.
        e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
        charges, and atoms 3, 4, and 5 have equal charges.

    Attributes
    ----------
    name: str
        name of the molecule. This is used to name output files.
    n_atoms: int
        number of atoms in each conformer
    indices: ndarray of ints
        indices of each atom, indexed from 0
    heavy_ids: ndarray of ints
        atom numbers of heavy atoms, indexed from 1
    h_ids: ndarray of ints
        atom numbers of hydrogen atoms, indexed from 1
    symbols: ndarray
        element symbols
    charge: int or float
        overall charge of molecule
    conformers: list of Conformers
    n_conf: int
        number of conformers
    unrestrained_charges: ndarray of floats
        partial atomic charges (only exists after calling fit or run)
    """


def multiresp_config(stage_2=True, hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1,
                     restraint=True, basis='6-31g*', method='scf', use_radii='msk',
                     ihfree=True, solvent=None):
    """Make analogous classes to R.E.D. options"""
    def wrapper(cls):
        class Config(cls):
            @clean_kwargs
            def run(self, opt=False, save_opt_geometry=False,
                    intra_chrconstr=[], intra_chrequiv=[], inter_chrconstr=[],
                    inter_chrequiv=[], n_orient=0, orient=[], n_rotate=0, rotate=[],
                    n_translate=0, translate=[], equal_methyls=False, psi4_options={},
                    **kwargs):
                """
                Parameters
                ----------
                opt: bool (optional)
                    Whether to optimise the geometry of each conformer
                save_opt_geometry: bool (optional)
                    if ``True``, writes optimised geometries to an XYZ file
                intra_chrconstr: list of lists or dicts (optional)
                    Intramolecular charge constraints for each molecule in the 
                    form of {charge: atom_number_list} or [(charge, atom_number_list)]. 
                    The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]] 
                    mean that atoms 1 and 2 together have a charge of 0.
                intra_chrequiv: list of lists (optional)
                    Lists of atoms with equivalent charges within each molecule.
                    e.g. [
                        [[1, 2], [3, 4, 5]],
                        [[1, 3, 5, 7]]
                        ] mean that atoms 1 and 2 in the first molecule have equal 
                    charges; atoms 3, 4, and 5 in the first molecule have 
                    equal charges; atoms 1, 3, 5, 7 in the second molecule have equal 
                    charges.
                inter_chrconstr: list of lists (optional)
                    Intermolecular charge constraints in the form of 
                    {charge: [(mol_number, atom_number)]} or 
                    [(charge, [(mol_number, atom_number)])]. 
                    The numbers are indexed from 1.
                    e.g. {0: [(1, 3), (2, 1)]} or [(0, [(1, 3), (2, 1)])] mean that
                    the third atom of the first molecule, and the first atom of the 
                    second molecule, combine to have a charge of 0.
                inter_chrequiv: list of lists (optional)
                    Lists of atoms with equivalent charges between each molecule, 
                    in the form [(mol_number, atom_number)]. 
                    e.g. [[(1, 2), (2, 2), (3, 4)]] mean that atom 2 of molecule 1, 
                    atom 2 of molecule 2, and atom 4 of molecule 3, all have equal 
                    charges.
                n_orient: int (optional)
                    If this is greater than 0 and ``orient`` is not given, 
                    ``n_orient`` orientations are automatically generated for each 
                    molcule. Heavy atoms are prioritised.
                orient: list of lists (optional)
                    List of lists of reorientations. 
                    Corresponds to REMARK REORIENT in R.E.D.
                    e.g. [(1, 5, 9), (9, 5, 1)] creates two reorientations: the first
                    around the first, fifth and ninth atom; and the second in reverse 
                    order.
                n_rotate: int (optional)
                    If this is greater than 0 and ``rotate`` is not given, 
                    ``n_rotate`` rotations are automatically generated for each 
                    molecule. Heavy atoms are prioritised.
                rotate: list of lists (optional)
                    List of lists of rotations. 
                    Corresponds to REMARK ROTATE in R.E.D.
                    e.g. [(1, 5, 9), (9, 5, 1)] creates two rotations: the first
                    around the first, fifth and ninth atom; and the second in reverse 
                    order.
                n_translate: int (optional)
                    If this is greater than 0 and ``translate`` is not given, 
                    ``n_translate`` translations are randomly generated for each 
                    molcule in the domain [0, 1).
                translate: list of lists (optional)
                    List of lists of translations. 
                    Corresponds to REMARK TRANSLATE in R.E.D.
                    e.g. [(1.0, 0, -0.5)] creates a translation that adds 1.0 to the 
                    x coordinates, 0 to the y coordinates, and -0.5 to the z 
                    coordinates.
                psi4_options: dict (optional)
                    additional Psi4 options
                **kwargs:
                    arguments passed to MultiResp.fit

                Returns
                -------
                charges: list of ndarrays
                """
                super(Config, self).run(opt=opt, save_opt_geometry=save_opt_geometry,
                                        intra_chrconstr=intra_chrconstr,
                                        intra_chrequiv=intra_chrequiv,
                                        inter_chrconstr=inter_chrconstr,
                                        inter_chrequiv=inter_chrequiv,
                                        psi4_options=psi4_options,
                                        n_orient=n_orient, orient=orient,
                                        n_rotate=n_rotate, rotate=rotate,
                                        n_translate=n_translate, translate=translate,
                                        equal_methyls=equal_methyls, stage_2=stage_2,
                                        hyp_a1=hyp_a1, hyp_a2=hyp_a2, hyp_b=hyp_b,
                                        restraint=restraint, basis=basis, solvent=solvent,
                                        method=method, ihfree=ihfree, **kwargs)

            def optimize_geometry(self, psi4_options={}, save_opt_geometry=False):
                """
                Optimise geometry for all conformers.

                Parameters
                ----------
                psi4_options: dict (optional)
                    additional Psi4 options
                save_opt_geometry: bool (optional)
                    if ``True``, saves optimised geometries to XYZ file
                """
                super(Config, self).optimize_geometry(method=method, basis=basis,
                                                    psi4_options=psi4_options,
                                                    save_opt_geometry=save_opt_geometry)

            @clean_kwargs
            def fit(self, tol=1e-6, maxiter=50, **kwargs):
                """
                Perform the Resp fits.

                Parameters
                ----------
                tol: float (optional)
                    threshold for convergence
                maxiter: int (optional)
                    maximum number of iterations
                **kwargs:
                    arguments passed to Resp.get_constraint_matrices

                Returns
                -------
                charges: ndarray
                """
                super(Config, self).fit(hyp_b=hyp_b, restraint=restraint, basis=basis,
                                        method=method, ihfree=ihfree, **kwargs)

            @clean_kwargs
            def get_constraint_matrices(self, intra_chrconstr=[], intra_chrequiv=[],
                                        inter_chrconstr=[], inter_chrequiv=[],
                                        weights=1, **kwargs):
                """
                Get A and B matrices to solve for charges, including charge constraints.

                Parameters
                ----------
                intra_chrconstr: list of lists or dicts (optional)
                    Intramolecular charge constraints for each molecule in the 
                    form of, where the list of constraints has the form 
                    [{charge: atom_number_list}] or [[(charge, atom_number_list)]]. 
                    The numbers are indexed from 1. 
                    e.g. [{0: [1, 2]}, {0.2: [3, 4, 5]}]
                    mean that atoms 1 and 2 together have a charge of 0 in the first
                    molecule, whereas atoms 3, 4, and 5 combine to a charge of 0.2 in 
                    the second molecule.
                intra_chrequiv: list of lists (optional)
                    Lists of atoms with equivalent charges within each molecule.
                    e.g. [
                        [[1, 2], [3, 4, 5]],
                        [[1, 3, 5, 7]]
                        ] mean that atoms 1 and 2 in the first molecule have equal 
                    charges; atoms 3, 4, and 5 in the first molecule have 
                    equal charges; atoms 1, 3, 5, 7 in the second molecule have equal 
                    charges.
                inter_chrconstr: list of lists (optional)
                    Intermolecular charge constraints in the form of 
                    {charge: [(mol_number, atom_number)]} or 
                    [(charge, [(mol_number, atom_number)])]. 
                    The numbers are indexed from 1.
                    e.g. {0: [(1, 3), (2, 1)]} or [(0, [(1, 3), (2, 1)])] mean that
                    the third atom of the first molecule, and the first atom of the 
                    second molecule, combine to have a charge of 0.
                inter_chrequiv: list of lists (optional)
                    Lists of atoms with equivalent charges between each molecule, 
                    in the form [(mol_number, atom_number)]. 
                    e.g. [[(1, 2), (2, 2), (3, 4)]] mean that atom 2 of molecule 1, 
                    atom 2 of molecule 2, and atom 4 of molecule 3, all have equal 
                    charges.
                weights: iterable (optional)
                    List of weights for each molecule, or list of lists of weights for
                    each conformer. e.g. [1, 2, [3, 4]] weights all conformers in the 
                    first molecule by 1; all the conformers in the second molecule by 2;
                    the first conformer in the third molecule by 3; and the second 
                    conformer in the third molecule by 4. If only one value is given, it
                    is repeated for every molecule.
                **kwargs
                    Arguments passed to Resp.get_constraint_matrices.

                Returns
                -------
                A: ndarray
                B: ndarray
                edges: list of tuples
                    list of start, end indices of the molecule atoms in `a` and `b`.
                    For example, ``i, j = edges[3]`` such that ``a[i:j, i:j]`` is 
                    the A matrix for the 4th molecule.
                nmol: ndarray
                    array containing the number of orientations in each Resp, with 
                    the same shape as B
                """

                super(Config, self).get_constraint_matrices(intra_chrconstr=intra_chrconstr,
                                                            intra_chrequiv=intra_chrequiv,
                                                            inter_chrconstr=inter_chrconstr,
                                                            inter_chrequiv=inter_chrequiv,
                                                            solvent=solvent, weights=weights,
                                                            method=method, basis=basis,
                                                            use_radii=use_radii,
                                                            **kwargs)

        return Config
    return wrapper


@due.dcite(
    Doi('10.1021/j100142a004'),
    description='RESP-A1 multi-molecule model',
    path='psiresp.configs',
)
@multiresp_config(stage_2=True, hyp_a1=0.0005, hyp_a2=0.001, hyp_b=0.1,
                  restraint=True, basis='6-31g*', method='scf',
                  ihfree=True, use_radii='msk', solvent=None)
class MultiRespA1(MultiResp):
    """
    Class to manage a 2-stage RESP fit for multiple molecules of 
    multiple conformers. This is called the RESP-A1 model in R.E.D.

    Please cite [Bayly1993]_ if you use this class.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.

    Attributes
    ----------
    molecules: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    n_molecules: int
        number of molecule Resp classes
    n_structures: int
        number of structures in entire MultiResp fit, where one structure 
        is one orientation of one conformer
    charges: list of ndarray
        partial atomic charges for each molecule
        (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1039/c0cp00111b'),
    description='RESP-A2 multi-molecule model',
    path='psiresp.configs',
)
@multiresp_config(stage_2=False, hyp_a1=0.01, hyp_a2=0.0, hyp_b=0.1,
                  restraint=True, basis='6-31g*', method='scf',
                  ihfree=True, use_radii='msk', solvent=None)
class MultiRespA2(MultiResp):
    """
    Class to manage a 1-stage RESP fit for multiple molecules of 
    multiple conformers. This is called the RESP-A2 
    model in R.E.D.

    Please cite [Dupradeau2010]_ if you use this class.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.

    Attributes
    ----------
    molecules: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    n_molecules: int
        number of molecule Resp classes
    n_structures: int
        number of structures in entire MultiResp fit, where one structure 
        is one orientation of one conformer
    charges: list of ndarray
        partial atomic charges for each molecule
        (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1002/jcc.540050204'),
    description='ESP-A1 multi-molecule model',
    path='psiresp.configs',
)
@multiresp_config(stage_2=False, hyp_a1=0.0, hyp_a2=0.0, hyp_b=0.1,
                  restraint=True, basis='6-31g*', method='scf',
                  ihfree=True, use_radii='msk', solvent=None)
class MultiEspA1(MultiResp):
    """
    Class to manage a 1-stage unrestrained fit for multiple molecules of 
    multiple conformers. This is called the ESP-A1 model in R.E.D.

    Please cite [Singh1984]_ if you use this class.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.

    Attributes
    ----------
    molecules: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    n_molecules: int
        number of molecule Resp classes
    n_structures: int
        number of structures in entire MultiResp fit, where one structure 
        is one orientation of one conformer
    charges: list of ndarray
        partial atomic charges for each molecule
        (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1002/jcc.540050204'),
    Doi('10.1039/c0cp00111b'),
    description='ESP-A2 multi-molecule model',
    path='psiresp.configs',
)
@multiresp_config(stage_2=False, hyp_a1=0.0, hyp_a2=0.0, hyp_b=0.1,
                  restraint=True, basis='sto-3g', method='scf',
                  ihfree=True, use_radii='msk', solvent=None)
class MultiEspA2(MultiResp):
    """
    Class to manage a 1-stage unrestrained fit for multiple molecules of 
    multiple conformers. This is called the ESP-A2 model in R.E.D.

    Please cite [Singh1984]_ and [Dupradeau2010]_ if you use this class.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.

    Attributes
    ----------
    molecules: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    n_molecules: int
        number of molecule Resp classes
    n_structures: int
        number of structures in entire MultiResp fit, where one structure 
        is one orientation of one conformer
    charges: list of ndarray
        partial atomic charges for each molecule
        (only exists after calling fit or run)
    """


@due.dcite(
    Doi('10.1021/ct200196m'),
    description='ATB multi-molecule model',
    path='psiresp.configs',
)
#: TODO: autodetect symmetry?
@multiresp_config(stage_2=False, hyp_a1=0.0, hyp_a2=0.0, hyp_b=0.1,
                  restraint=False, basis='6-31g*', method='b3lyp',
                  ihfree=False, use_radii='msk', solvent='water')
class ATBMultiResp(MultiResp):
    """
    Class to manage a 1-stage unrestrained fit using the method of 
    Singh and Kollman 1984 at B3LYP/6-31G* with implicit solvent, as 
    described for the Automated Topology Builder.

    Please cite [Malde2011]_ and [Singh1984]_ if you use this class.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.

    Attributes
    ----------
    molecules: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.
    n_molecules: int
        number of molecule Resp classes
    n_structures: int
        number of structures in entire MultiResp fit, where one structure 
        is one orientation of one conformer
    charges: list of ndarray
        partial atomic charges for each molecule
        (only exists after calling fit or run)
    """
