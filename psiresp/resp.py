from __future__ import division, absolute_import
import warnings
import itertools
import logging

import numpy as np

from .conformer import Conformer
from . import utils

log = logging.getLogger(__name__)


class Resp(object):
    """
    Class to manage R/ESP for one molecule of multiple conformers.

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
        partial atomic charges (only exists after calling fit or run
        with restraint=True)
    """

    @classmethod
    def from_molecules(cls, molecules, charge=0, multiplicity=1, name=None,
                       orient=[], rotate=[], translate=[],
                       grid_name='grid.dat', esp_name='grid_esp.dat',
                       mat_name='abmat.dat',
                       load_files=False, **kwargs):
        """
        Create Resp class from Psi4 molecules.

        Parameters
        ----------
        molecules: iterable of Psi4 molecules
            conformers of the molecule. They must all have the same atoms
            in the same order.
        charge: int (optional)
            overall charge of the molecule.
        multiplicity: int (optional)
            multiplicity of the molecule
        name: str (optional)
            name of the molecule. This is used to name output files. If not
            given, defaults to 'Mol'.
        orient: list of tuples of ints (optional)
            List of reorientations to add to each conformer.
            Corresponds to REMARK REORIENT in R.E.D.
            e.g. [(1, 5, 9), (9, 5, 1)] creates two reorientations: the first
            around the first, fifth and ninth atom; and the second in reverse
            order.
        rotate: list of tuples of ints (optional)
            List of rotations to add to each conformer.
            Corresponds to REMARK ROTATE in R.E.D.
            e.g. [(1, 5, 9), (9, 5, 1)] creates two rotations: the first
            around the first, fifth and ninth atom; and the second in reverse
            order.
        translate: list of tuples of floats (optional)
            List of translations to add to each conformer.
            Corresponds to REMARK TRANSLATE in R.E.D.
            e.g. [(1.0, 0, -0.5)] creates a translation that adds 1.0 to the
            x coordinates, 0 to the y coordinates, and -0.5 to the z coordinates.
        grid_name: str (optional)
            template for grid filename for each Orientation.
        esp_name: str (optional)
            template for ESP filename for each Orientation.
        load_files: bool (optional)
            If ``True``, tries to load data from file for each Orientation.
        **kwargs:
            arguments passed to ``Resp.__init__()``.

        Returns
        -------
        resp: Resp
        """
        molecules = utils.asiterable(molecules)
        if name is not None:
            names = ['{}_c{}'.format(name, i+1) for i in range(len(molecules))]
        else:
            molnames = [m.name() for m in molecules]
            gennames = ['Mol_c{}'.format(i+1) for i in range(len(molecules))]
            # default Psi4 name is 'default'...
            # Hope no one actually wants to call their molecules default!
            names = np.where(molnames == 'default', molnames, gennames)

        conformers = []
        for mol, n in zip(molecules, names):
            conformers.append(Conformer(mol.clone(), name=n, charge=charge,
                                        multiplicity=multiplicity,
                                        orient=orient, rotate=rotate,
                                        translate=translate,
                                        grid_name=grid_name,
                                        esp_name=esp_name,
                                        mat_name=mat_name,
                                        load_files=load_files))

        return cls(conformers, name=name, load_files=load_files,
                   **kwargs)

    def __init__(self, conformers, name=None, chrconstr=[], chrequiv=[],
                 load_files=False):
        if name is None:
            name = 'Resp'
        if not conformers:
            raise ValueError('Resp must be created with at least one conformer')
        self.name = name
        self._load_files = load_files
        self.conformers = utils.asiterable(conformers)
        self.n_conf = len(self.conformers)
        self.charge = self._conf.charge
        self.symbols = self._conf.symbols
        self.n_atoms = self._conf.n_atoms
        self.indices = np.arange(self.n_atoms)
        self.atom_ids = self.indices+1
        self.heavy_ids = np.where(self.symbols != 'H')[0]+1
        self.h_ids = np.where(self.symbols == 'H')[0]+1
        self._orient_combs = self._gen_orientation_atoms()
        self.chrconstr = []
        self.chrequiv = []
        if chrconstr is not None:
            self.add_charge_constraints(chrconstr)
        if chrequiv is not None:
            self.add_charge_equivalences(chrequiv)

        log.debug(f'Resp(name={self.name}) created with '
                  f'{self.n_conf} conformers, {len(self.chrconstr)} charge '
                  f'constraints and {len(self.chrequiv)} charge equivalences')

    @property
    def _conf(self):
        return self.conformers[0]

    @property
    def n_structures(self):
        return sum([c.n_orientations for c in self.conformers])

    def clone(self, name=None):
        """Clone into another instance of Resp

        Parameters
        ----------
        name: str (optional)
            If not given, the new Resp instance has '_copy' appended 
            to its name

        Returns
        -------
        resp: Resp
        """
        if name is None:
            name = self.name+'_copy'
        mols = [c.molecule.clone() for c in self.conformers]
        charge = self.charge
        mult = self._conf.multiplicity
        new = type(self).from_molecules(mols, name=name, charge=charge,
                                        multiplicity=mult, load_files=self._load_files)
        for nc, mc in zip(new.conformers, self.conformers):
            nc.add_orientations(orient=mc._orient, rotate=mc._rotate,
                                translate=mc._translate)
        return new

    def add_charge_constraint(self, charge, atom_ids):
        """
        Add charge constraint

        Parameters
        ----------
        charge: float
        atom_ids: iterable of ints
            Atom numbers involved in charge constraint, indexed from 1.
            All the atoms specified here combine for a charge of ``charge``. 
        """
        atom_ids = utils.asiterable(atom_ids)
        if not all(isinstance(x, int) for x in atom_ids):
            raise ValueError('atom_ids must be an iterable of integer atom numbers')
        for a in atom_ids:
            if a not in self.atom_ids:
                raise ValueError('Atom number not found: {}'.format(a))
        self.chrconstr.append([charge, np.asarray(atom_ids)])

    def add_charge_constraints(self, chrconstr=[]):
        """Add charge constraints

        Parameters
        ----------
        chrconstr: list or dict (optional)
            Intramolecular charge constraints in the form of
            {charge: atom_number_list} or [(charge, atom_number_list)].
            The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
            mean that atoms 1 and 2 together have a charge of 0.
        """
        if isinstance(chrconstr, dict):
            chrconstr = list(chrconstr.items())
        for q, ids in chrconstr:
            self.add_charge_constraint(q, ids)

    def add_charge_equivalence(self, atom_ids):
        """
        Add constraint for equivalent charges.

        Parameters
        ----------
        atom_ids: iterable of ints
            Atom numbers involved in charge constraint, indexed from 1.
            All the atoms specified here are constrained to the same charge.
        """
        for a in atom_ids:
            if a not in self.atom_ids:
                raise ValueError('Atom number not found: {}'.format(a))
        if len(atom_ids) < 2:
            raise ValueError('Cannot add equivalence constraint for <2 atoms')
        self.chrequiv.append(np.asarray(atom_ids))

    def add_charge_equivalences(self, chrequiv=[]):
        """Add charge equivalence constraints

        Parameters
        ----------
        chrequiv: list (optional)
            lists of atoms with equivalent charges, indexed from 1.
            e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
            charges, and atoms 3, 4, and 5 have equal charges.
        """
        for ids in chrequiv:
            self.add_charge_equivalence(ids)

    def optimize_geometry(self, method='scf', basis='6-31g*',
                          psi4_options={}, save_opt_geometry=False,
                          save_files=False):
        """
        Optimise geometry for all conformers.

        Parameters
        ----------
        basis: str (optional)
            Basis set to optimise geometry
        method: str (optional)
            Method to optimise geometry
        psi4_options: dict (optional)
            additional Psi4 options
        save_opt_geometry: bool (optional)
            if ``True``, saves optimised geometries to XYZ file
        save_files: bool (optional)
            if ``True``, Psi4 files are saved. This does not affect 
            writing optimised geometries to files. 
        """
        for conf in self.conformers:
            conf.optimize_geometry(method=method, basis=basis,
                                   psi4_options=psi4_options,
                                   save_opt_geometry=save_opt_geometry,
                                   save_files=save_files)

    def set_user_constraints(self, chrconstr=[], chrequiv=[]):
        """
        Get A and B matrices for charge constraints.

        Parameters
        ----------
        chrconstr: list or dict (optional)
            Intramolecular charge constraints in the form of
            {charge: atom_number_list} or [(charge, atom_number_list)].
            The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
            mean that atoms 1 and 2 together have a charge of 0.
        chrequiv: list (optional)
            lists of atoms with equivalent charges, indexed from 1.
            e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
            charges, and atoms 3, 4, and 5 have equal charges.

        Returns
        -------
        A: ndarray
        B: ndarray
        """
        if isinstance(chrconstr, dict):
            chrconstr = list(chrconstr.items())
        else:
            chrconstr = list(chrconstr)

        if chrequiv is None:
            chrequiv = []
        else:
            chrequiv = list(chrequiv)

        chrconstr = self.chrconstr + chrconstr
        chrequiv = self.chrequiv + chrequiv

        equiv = [np.asarray(x)-1 for x in chrequiv if len(x) >= 2]
        edges = np.r_[0, np.cumsum([len(x)-1 for x in equiv])].astype(int)
        n_equiv = edges[-1]
        n_constr = len(chrconstr)
        ndim = n_equiv+n_constr+self.n_atoms+1

        AB = np.zeros((ndim+1, ndim))
        A = AB[:ndim]
        B = AB[-1]

        for i, (q, ids) in enumerate(chrconstr, self.n_atoms+1):
            B[i] = q
            ix = np.asarray(ids)-1
            A[i, ix] = A[ix, i] = 1

        for i, indices in enumerate(equiv):
            x = np.arange(edges[i], edges[i+1])+self.n_atoms+1+n_constr
            A[(x, indices[:-1])] = A[(indices[:-1], x)] = -1
            A[(x, indices[1:])] = A[(indices[1:], x)] = 1

        return AB

    def iter_solve(self, q, a, b, hyp_a=0.0005, hyp_b=0.1, ihfree=True,
                   tol=1e-6, maxiter=50):
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
        hyp_b: float (optional)
            tightness of hyperbola at its minimum
        ihfree: bool (optional)
            if `True`, exclude hydrogens from restraint
        tol: float (optional)
            threshold for convergence
        maxiter: int (optional)
            maximum number of iterations

        Returns
        -------
        charges: ndarray
        """
        if not hyp_a:  # i.e. no restraint
            return q

        mask = np.ones(self.n_atoms, dtype=bool)
        if ihfree:
            mask[self.h_ids-1] = False
        diag = np.diag_indices(self.n_atoms)
        ix = (diag[0][mask], diag[1][mask])
        indices = np.where(mask)[0]
        b2 = hyp_b**2

        niter, delta = 0, 2*tol
        while delta > tol and niter < maxiter:
            q_last, a_i = q.copy(), a.copy()
            a_i[ix] = a[ix] + hyp_a/np.sqrt(q[indices]**2 + b2) * self.n_structures
            q = np.linalg.solve(a_i, b)
            delta = np.max((q-q_last)[:self.n_atoms]**2) ** 0.5
            niter += 1

        if delta > tol:
            err = 'Charge fitting did not converge with maxiter={}'
            warnings.warn(err.format(maxiter))

        return q

    def get_constraint_matrices(self, chrconstr=[], chrequiv=[],
                                weights=1, use_radii='msk',
                                vdw_scale_factors=(1.4, 1.6, 1.8, 2.0),
                                vdw_point_density=1.0, vdw_radii={},
                                rmin=0, rmax=-1, basis='6-31g*', method='scf',
                                solvent=None, psi4_options={},
                                save_files=False, load_files=False,
                                mat_name=''):
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
            lists of atoms with equivalent charges, indexed from 1.
            e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
            charges, and atoms 3, 4, and 5 have equal charges.
        weights: iterable (optional)
            weights of each conformer. If only one number is given, this is 
            repeated for each conformer.
        use_radii: str (optional)
            which set of van der Waals' radii to use
        vdw_scale_factors: iterable of floats (optional)
            scale factors
        vdw_point_density: float (optional)
            point density
        vdw_radii: dict (optional)
            van der Waals' radii. If elements in the molecule are not
            defined in the chosen ``use_radii`` set, they must be given here.
        rmin: float (optional)
            inner boundary of shell to keep grid points from
        rmax: float (optional)
            outer boundary of shell to keep grid points from. If < 0,
            all points are selected.
        basis: str (optional)
            Basis set to compute ESP
        method: str (optional)
            Method to compute ESP
        solvent: str (optional)
            Solvent for computing in implicit solvent
        psi4_options: dict (optional)
            additional Psi4 options
        save_files: bool (optional)
            if ``True``, Psi4 files are saved and the computed ESP
            and grids are written to files.

        Returns
        -------
        a: ndarray
        b: ndarray
        """
        mat_filename = utils.prepend_name_to_file(self.name, mat_name)
        AB = None

        if load_files and self._load_files and mat_name:
            try:
                AB = np.loadtxt(mat_filename)
            except OSError:
                warnings.warn(f'Could not read data from {mat_filename}')
            else:
                log.info(f'Read matrix from {mat_filename}')
                A, B = AB[:-1], AB[-1]
                return A, B

        AB = self.set_user_constraints(chrconstr=chrconstr,
                                       chrequiv=chrequiv)
        A, B = AB[:-1], AB[-1]
        log.debug(f'Computing {self.name} Resp constraint matrices '
                  f'of dimension {len(B)}')

        if not utils.isiterable(weights):
            weights = itertools.repeat(weights)
        elif len(weights) != self.n_conf:
            err = ('weights must be an iterable of the same length '
                   'as number of conformers')
            raise ValueError(err)

        # don't compute anything unless we have to
        if any(conf._grid_needs_computing for conf in self.conformers):
            vdw_points = utils.gen_connolly_shells(self.symbols,
                                                   vdw_radii=vdw_radii,
                                                   use_radii=use_radii,
                                                   scale_factors=vdw_scale_factors,
                                                   density=vdw_point_density)
        else:
            vdw_points = []

        # get molecule weights
        n_a = self.n_atoms
        for conf, w in zip(self.conformers, weights):
            a, b = conf.get_esp_matrices(weight=w, vdw_points=vdw_points,
                                         rmin=0, rmax=-1, basis=basis,
                                         method=method, solvent=solvent,
                                         psi4_options=psi4_options,
                                         save_files=save_files, load_files=load_files)
            A[:n_a, :n_a] += a
            B[:n_a] += b

        A[n_a, :n_a] = A[:n_a, n_a] = 1
        B[n_a] = self.charge

        if save_files and mat_name:
            np.savetxt(mat_filename, AB)
            log.info(f'Saved matrix to {mat_filename}')

        return A, B

    def fit(self, restraint=True, hyp_a=0.0005, hyp_b=0.1, ihfree=True,
            tol=1e-6, maxiter=50, **kwargs):
        """
        Perform the R/ESP fits.

        Parameters
        ----------
        restraint: bool (optional)
            whether to perform a restrained fit
        hyp_a: float (optional)
            scale factor of asymptote limits of hyperbola
        hyp_b: float (optional)
            tightness of hyperbola at its minimum
        ihfree: bool (optional)
            if `True`, exclude hydrogens from restraint
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
        a, b = self.get_constraint_matrices(**kwargs)
        q = np.linalg.solve(a, b)
        self.unrestrained_charges = q[:self.n_atoms]
        if restraint:
            q = self.iter_solve(q, a, b, hyp_a=hyp_a, hyp_b=hyp_b,
                                ihfree=ihfree, tol=tol, maxiter=maxiter)
            self.restrained_charges = q[:self.n_atoms]

        self.charges = q[:self.n_atoms]
        return self.charges

    def get_sp3_ch_ids(self):
        """
        Get atom numbers of hydrogens bonded to sp3 carbons. Numbers
        are indexed from 1.

        Returns
        -------
        groups: dict
            dict of {C atom number: array of bonded H numbers}.
        """
        import psi4

        groups = {}
        bonds = psi4.qcdb.parker._bond_profile(self._conf.molecule)
        bonds = np.asarray(bonds)[:, :2]  # [[i, j, bond_order]]
        for i in self.indices[self.symbols == 'C']:
            cbonds = bonds[np.any(bonds == i, axis=1)]
            partners = np.ravel(cbonds[cbonds != i])
            groups[i+1] = partners[self.symbols[partners] == 'H']+1
        return groups

    def get_methyl_constraints(self, chrconstr=None):
        """
        Get charge equivalence arrays when all methyls are treated as
        equivalent, and all methylenes are equivalent. Toggle this with
        ``equal_methyls=True`` in ``run()``.

        Parameters
        ----------
        chrconstr: list or dict (optional)
            Intramolecular charge constraints in the form of
            {charge: atom_number_list} or [(charge, atom_number_list)].
            The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
            mean that atoms 1 and 2 together have a charge of 0.

        Returns
        -------
        equivalence arrays: list of lists of ints
            List of equivalence arrays. First array contains methyl carbons;
            second contains methylene carbons; third contains methyl hydrogens;
            last contains methylene hydrogens.
        """
        cs, hs = zip(*self.get_sp3_ch_ids().items())
        if chrconstr is None:
            flat_constr = []
        else:
            flat_constr = np.ravel(chrconstr)
        equivs = [c3s, c2s, h3s, h2s] = [[], [], [], []]
        for c, h in zip(cs, hs):
            if c in flat_constr or any(x in flat_constr for x in h):
                continue
            if len(h) == 3:
                c3s.append(c)
                h3s.extend(h)
            else:
                c2s.append(c)
                h2s.extend(h)
        return equivs

    def get_stage2_constraints(self, q, equal_methyls=False,
                               chrequiv=[], chrconstr=[]):
        """
        Create constraints for RESP stage 2. Atom numbers are indexed from 1.

        If charge equivalence constraints are provided, these are used in stage
        2 fitting. Otherwise, if ``equal_methyls=True``, the carbons in
        methyl groups are constrained to be equivalent, as are the carbons in
        methylenes, and the hydrogens in them. If ``equal_methyls=False``,
        equivalence constraints are created for each group of hydrogens bonded
        to sp3 carbons, separately.

        Any atoms involved in charge constraints are not included in the
        equivalence constraints.

        Parameters
        ----------
        q: ndarray
            charges from stage 1
        equal_methyls: bool (optional)
            if ``True``, all carbons in methyl groups are constrained to be
            equivalent; all carbons in methylenes are equivalent; all hydrogens
            in methyls are equivalent; and all hydrogens in methylenes are
            equivalent. Ignored if ``chrequiv`` constraints are provided.
        chrconstr: list or dict (optional)
            Intramolecular charge constraints in the form of
            {charge: atom_number_list} or [(charge, atom_number_list)].
            The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
            mean that atoms 1 and 2 together have a charge of 0.
        chrequiv: list (optional)
            lists of atoms with equivalent charges, indexed from 1.
            e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
            charges, and atoms 3, 4, and 5 have equal charges.

        Returns
        -------
        chrconstr: list of tuples
            Charge constraints in the form of [(charge, atom_number_list)]
        chrequiv: list
            Lists of atoms with equivalent charges in the form
            [atom_number_list]
        """
        cs = []
        if chrequiv:  # constraints are fitted in stage 2
            equivs = chrequiv
        elif equal_methyls:
            equivs = self.get_methyl_constraints(chrconstr)
        else:
            cs, equivs = zip(*self.get_sp3_ch_ids().items())

        # TODO: look into this
        # sort through equivs; cannot have 2 equivalent atoms, that
        # are constrained to different charges in chrconstr; and
        # if they are constrained to the same charges I think it makes
        # a singular matrix?
        # if isinstance(chrconstr, dict):
        #     chrconstr = list(chrconstr.items())
        # charge_mapping = {at[0]: q for q, at in chrconstr if len(at) == 1}
        # final_equivs = []
        # for eq in equivs:
        #     if not len(eq) >= 2:
        #         continue
        #     charges = [charge_mapping[a] for a in eq if a in charge_mapping]
        #     if len(set(charges)) <= 1:
        #         final_equivs.append(eq)
        final_equivs = equivs

        chs = np.r_[cs, np.concatenate(final_equivs)]

        q = np.asarray(q)
        ids = self.indices+1
        mask = ~np.in1d(ids, chs)
        constraints = [(q, [a]) for q, a in zip(q[mask], ids[mask])]
                    #    if a not in charge_mapping]
        return constraints, final_equivs

    def _gen_orientation_atoms(self):
        """
        Generate potential orientations from heavy atoms first,
        then hydrogens.

        Returns
        -------
        all_combinations: list of tuples of ints
            e.g. [(1, 5, 9)]
        """
        comb = list(itertools.combinations(self.heavy_ids, 3))
        comb += list(itertools.combinations(self.h_ids, 3))

        combr = [x[::-1] for x in comb]
        all_comb = [x for items in zip(comb, combr) for x in items]
        return all_comb

    def add_orientations(self, orient=[], n_orient=0, translate=[],
                         n_translate=0, rotate=[], n_rotate=0,
                         load_files=False):
        """
        Add orientations to conformers.

        Parameters
        ----------
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
        load_files: bool (optional)
            If ``True``, tries to load ESP and grid data from file.
        """
        if not orient and n_orient:
            orient = self._orient_combs[:n_orient]

        if not rotate and n_rotate:
            rotate = self._orient_combs[:n_rotate]

        if not translate and n_translate:
            translate = np.random.rand(n_translate, 3)

        for conf in self.conformers:
            conf.add_orientations(orient=orient, rotate=rotate,
                                  translate=translate, load_files=load_files)

    def run(self, stage_2=True, opt=False, save_opt_geometry=False,
            chrconstr=[], chrequiv=[], basis='6-31g*', method='scf',
            psi4_options={}, hyp_a1=0.0005, hyp_a2=0.001, n_orient=0, orient=[],
            n_rotate=0, rotate=[], n_translate=0, translate=[],
            equal_methyls=False, restraint=True, load_files=False,
            **kwargs):
        """
        Perform a 1- or 2-stage ESP or RESP fit.

        Parameters
        ----------
        stage_2: bool (optional)
            Whether to perform a 2-stage RESP fit
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
            lists of atoms with equivalent charges, indexed from 1.
            e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
            charges, and atoms 3, 4, and 5 have equal charges.
        basis: str (optional)
            Basis set for QM
        method: str (optional)
            Method for QM
        psi4_options: dict (optional)
            additional Psi4 options
        hyp_a1: float (optional)
            scale factor of asymptote limits of hyperbola for first stage.
        hyp_a2: float (optional)
            scale factor of asymptote limits of hyperbola for second stage.
        restraint: bool (optional)
            whether to perform a restrained fit
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
            equivalent. 
        load_files: bool (optional)
            If ``True``, tries to load ESP and grid data from file.
        **kwargs:
            arguments passed to Resp.fit

        Returns
        -------
        charges: ndarray
        """

        if opt:
            self.optimize_geometry(method=method, basis=basis,
                                   psi4_options=psi4_options,
                                   save_opt_geometry=save_opt_geometry)

        self.add_orientations(n_orient=n_orient, orient=orient,
                              n_rotate=n_rotate, rotate=rotate,
                              n_translate=n_translate,
                              translate=translate, load_files=load_files)

        if stage_2:  # do constraints in stage 2 only
            stage_2_equiv = chrequiv
            chrequiv = []
        else:
            stage_2_equiv = []

            if equal_methyls:
                methyls = self.get_methyl_constraints(chrconstr)
                chrequiv = list(chrequiv) + methyls

        q = self.fit(chrconstr=chrconstr,
                     chrequiv=chrequiv,
                     basis=basis, method=method,
                     hyp_a=hyp_a1, restraint=restraint,
                     load_files=load_files,
                     mat_name='stg1_abmat.dat',
                     **kwargs)

        if stage_2:
            if restraint:
                self.stage_1_charges = [self.unrestrained_charges,
                                        self.restrained_charges]
            else:
                self.stage_1_charges = self.unrestrained_charges
            cs = self.get_stage2_constraints(q, chrequiv=stage_2_equiv,
                                             equal_methyls=equal_methyls)
            constr, equiv = cs

            q = self.fit(chrconstr=constr, chrequiv=equiv,
                         basis=basis, method=method,
                         hyp_a=hyp_a2, restraint=restraint,
                         load_files=load_files,
                         mat_name='stg2_abmat.dat',
                         **kwargs)
        return q
