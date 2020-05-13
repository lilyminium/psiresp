from __future__ import division, absolute_import
import warnings
import itertools

import numpy as np

from .resp import Resp
from . import utils


class MultiResp(object):
    """
    Class to manage R/ESP for multiple molecules of multiple conformers.

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

    def __init__(self, resps):
        self.molecules = resps
        self._unrestrained_charges = None
        self._restrained_charges = None

    def clone(self, suffix='_copy'):
        names = [r.name+suffix for r in self.molecules]
        resps = [m.clone(name=n) for n, m in zip(names, self.molecules)]
        return type(self)(resps)

    @property
    def n_structures(self):
        return sum([mol.n_structures for mol in self.molecules])

    @property
    def n_molecules(self):
        return len(self.molecules)

    @property
    def restrained_charges(self):
        return self._restrained_charges

    @restrained_charges.setter
    def restrained_charges(self, charges):
        for mol, q in zip(self.molecules, charges):
            mol.restrained_charges = q
        self._restrained_charges = charges

    @property
    def unrestrained_charges(self):
        return self._unrestrained_charges

    @unrestrained_charges.setter
    def unrestrained_charges(self, charges):
        for mol, q in zip(self.molecules, charges):
            mol.unrestrained_charges = q
        self._unrestrained_charges = charges

    def _nmol_values(self, values, name, element):
        """
        Validate given values to check they are either a single value 
        or the same length as ``n_molecules``. Raises an error message 
        with the ``name`` of the variable and ``element`` description 
        if the lengths mismatch.
        """
        values = utils.iter_single(values)
        try:
            if len(values) != self.n_molecules:
                err = ('{} must be a list with the '
                       'same length as the number of molecules, '
                       'where each element is a {}')
                raise ValueError(err.format(name, element))
        except TypeError:  # itertools repeat
            pass
        return values

    def optimize_geometry(self, method='scf', basis='6-31g*',
                          psi4_options={}, save_opt_geometry=False,
                          save_files=False):
        """
        Optimise geometry for all molecules.

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
        for mol in self.molecules:
            mol.optimize_geometry(method=method, basis=basis,
                                  psi4_options=psi4_options,
                                  save_opt_geometry=save_opt_geometry,
                                  save_files=save_files)

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
        intra_chrconstr = self._nmol_values(intra_chrconstr,
                                            'intra_chrconstr',
                                            'list or dict of constraints')
        intra_chrequiv = self._nmol_values(intra_chrequiv,
                                           'intra_chrequiv',
                                           'list of equivalence constraints')

        if utils.empty(inter_chrconstr):
            inter_chrconstr = []
        if utils.empty(inter_chrequiv):
            inter_chrequiv = []

        if not utils.isiterable(weights):
            weights = itertools.repeat(weights)
        elif len(weights) != self.n_molecules:
            err = ('weights must be an iterable of the same length '
                   'as number of molecules')
            raise ValueError(err)

        # gather individual A and B matrices for each Resp class
        a_s, b_s, nmol = [], [], []
        for m, constr, equiv, w in zip(self.molecules, intra_chrconstr,
                                       intra_chrequiv, weights):
            a, b = m.get_constraint_matrices(chrconstr=constr, chrequiv=equiv,
                                             weights=w, **kwargs)
            a_s.append(a)
            b_s.append(b)
            nmol.extend([m.n_mol]*len(b))

        mol_edges = np.r_[0, np.cumsum([len(b) for b in b_s])].astype(int)
        edges = list(zip(mol_edges[:-1], mol_edges[1:]))
        n_intra = mol_edges[-1]

        # allow for inter constraints to be given either as (nmol, natom)
        # or (nmol, [n_atom])
        def groups_to_indices(groups):
            indices = []
            for a, b in groups:
                offset = mol_edges[a-1]
                if isinstance(b, (tuple, list)):
                    indices.extend([offset+x-1 for x in b])
                else:
                    indices.append(offset+x-1)
            return np.array(indices)

        # set up intermolecular constraints
        if isinstance(inter_chrconstr, dict):
            inter_chrconstr = list(inter_chrconstr.items())
        inter_constr = [(q, groups_to_indices(e)) for q, e in inter_chrconstr]
        n_constr = len(inter_constr)

        inter_equiv = [groups_to_indices(e) for e in inter_chrequiv]
        inter_equiv = [e for e in inter_equiv if len(e) >= 2]
        lengths = np.cumsum([len(x)-1 for x in inter_equiv])
        equiv_edges = np.r_[0, lengths].astype(int)
        n_equiv = equiv_edges[-1]

        ndim = n_intra+n_equiv+n_constr
        nmol.extend([self.n_mol]*(n_equiv+n_constr))

        A = np.zeros((ndim, ndim))
        B = np.zeros(ndim)

        for (i, j), a, b in zip(edges, a_s, b_s):
            A[i:j, i:j] = a
            B[i:j] = b

        for i, (q, ix) in enumerate(inter_constr, n_intra):
            B[i] = q
            A[i, ix] = A[ix, i] = 1

        for i, ix in enumerate(inter_equiv):
            x = np.arange(equiv_edges[i], equiv_edges[i+1])+n_intra+n_constr
            A[(x, ix[:-1])] = A[(ix[:-1], x)] = -1
            A[(x, ix[1:])] = A[(ix[1:], x)] = 1

        edges = [[i, i+m.n_atoms] for (i, _), m in zip(edges, self.molecules)]
        return A, B, np.array(edges), np.array(nmol)

    def fit(self, restraint=True, hyp_a=0.0005, hyp_b=0.1, ihfree=True,
            tol=1e-5, maxiter=50, **kwargs):
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
            arguments passed to MultiResp.get_constraint_matrices

        Returns
        -------
        charges: list of ndarray
        """
        a, b, edges, nmol = self.get_constraint_matrices(**kwargs)
        q = np.linalg.solve(a, b)
        self.unrestrained_charges = [q[i:j] for i, j in edges]
        if restraint:
            q = self.iter_solve(q, a, b, edges, nmol, hyp_a=hyp_a, hyp_b=hyp_b,
                                ihfree=ihfree, tol=tol, maxiter=maxiter)
            self.restrained_charges = [q[i:j] for i, j in edges]
        self.charges = [q[i:j] for i, j in edges]
        for mol, q in zip(self.molecules, self.charges):
            mol.charges = q
        return self.charges

    def iter_solve(self, q, a, b, edges, nmol, hyp_a=0.0005, hyp_b=0.1,
                   ihfree=True, tol=1e-5, maxiter=50):
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
        edges: list of tuples
            list of start, end indices for the atoms in ``a`` and ``b``
        nmol: ndarray
            array containing the number of orientations in each Resp, with 
            the same shape as ``b``
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
        mask = np.zeros(len(b), dtype=bool)
        for i, j in edges:
            mask[i:j] = True
        if ihfree:
            for mol, (i, j) in zip(self.molecules, edges):
                mask[i:j] = mol.symbols != 'H'
        diag = np.diag_indices(len(b))
        ix = (diag[0][mask], diag[1][mask])
        b2 = hyp_b**2
        n_mol = nmol[mask]

        niter, delta = 0, 2*tol
        while delta > tol and niter < maxiter:
            q_last, a_i = q.copy(), a.copy()
            a_i[ix] = a[ix] + hyp_a/np.sqrt(q[mask]**2 + b2) * n_mol
            q = np.linalg.solve(a_i, b)
            delta = np.max((q-q_last)[mask]**2) ** 0.5
            niter += 1

        if delta > tol:
            err = 'Charge fitting did not converge with maxiter={}'
            warnings.warn(err.format(maxiter))

        return q

    def get_stage2_constraints(self, qs, equal_methyls=False,
                               intra_chrequiv=[], inter_chrconstr=[],
                               intra_chrconstr=[]):
        """
        Create constraints for Resp stage 2. Atom numbers are indexed from 1.

        Parameters
        ----------
        qs: ndarray
            list of charges for each molecule from stage 1
        equal_methyls: bool (optional)
            if ``True``, all carbons in methyl groups are constrained to be
            equivalent; all carbons in methylenes are equivalent; all hydrogens
            in methyls are equivalent; and all hydrogens in methylenes are
            equivalent. Ignored if ``chrequiv`` constraints are provided.
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
        inter_chrequiv: list of lists (optional)
            Original intermolecular charge equivalence constraints from 
            stage 1

        Returns
        -------
        intra_chrconstr: list
            Intramolecular charge constraints for each molecule in the 
            form of [[[charge, atom_number_list]]]
        intra_chrequiv: list
            Intramolecular charge equivalence constraints for each 
            molecule in the form of [[[atom_number_list]]]
        """

        intra_chrconstr = self._nmol_values(intra_chrconstr,
                                            'intra_chrconstr',
                                            'list or dict of constraints')
        intra_chrequiv = self._nmol_values(intra_chrequiv,
                                           'intra_chrequiv',
                                           'list of equivalence constraints')

        iconstr = []
        iequiv = []

        # add inter constraints too
        if inter_chrconstr is None:
            inter_chrconstr = []
        elif isinstance(inter_chrconstr, dict):
            inter_chrconstr = list(inter_chrconstr.items())

        # anything involved in a charge constraint cannot fluctuate freely
        chrconstr = [list() for i in range(self.n_molecules)]
        for _, groups in inter_chrconstr:
            for molid, atoms in groups:
                if isinstance(atoms, (tuple, list)):
                    chrconstr[molid-1].extend(atoms)
                else:
                    chrconstr[molid-1].append(atoms)

        for i, atoms in enumerate(intra_chrconstr):
            chrconstr[i].extend(atoms)  # avoid any tuple+list hijinks

        rows = zip(self.molecules, qs, intra_chrequiv, chrconstr)
        for i, (mol, q, eq, cr) in enumerate(rows):
            c, e = mol.get_stage2_constraints(q, chrequiv=eq, chrconstr=cr,
                                              equal_methyls=equal_methyls)
            iconstr.append(c)
            iequiv.append(e)

        return iconstr, iequiv

    def get_methyl_constraints(self):
        """
        Get charge equivalence arrays when all methyls are treated as 
        equivalent, and all methylenes are equivalent. Toggle this with 
        ``equal_methyls=True`` in ``run()``.

        Returns
        -------
        equivalence arrays: list of lists of ints
            List of equivalence arrays. First array contains methyl carbons;
            second contains methylene carbons; third contains methyl hydrogens;
            last contains methylene hydrogens.
        """
        equivs = [[], [], [], []]
        for mol in self.molecules:
            meqs = mol.get_methyl_constraints()
            for j, eqs in enumerate(meqs):
                equivs[j].append(eqs)
        return equivs

    def add_orientations(self, orient=[], n_orient=0, rotate=[],
                         n_rotate=0, translate=[], n_translate=0,
                         load_files=False):
        """
        Add orientations to each conformer of each Resp molecule. 

        Parameters
        ----------
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
        load_files: bool (optional)
            If ``True``, tries to load ESP and grid data from file.
        """

        orient = utils.iter_single(orient)
        rotate = utils.iter_single(rotate)
        translate = utils.iter_single(translate)
        for mol, o, r, t in zip(self.molecules, orient, rotate, translate):
            mol.add_orientations(orient=o, n_orient=0, rotate=r,
                                 n_rotate=0, translate=t, n_translate=0,
                                 load_files=load_files)

    def run(self, stage_2=True, opt=False, save_opt_geometry=False,
            intra_chrconstr=[], intra_chrequiv=[], inter_chrconstr=[],
            inter_chrequiv=[], n_orient=0, orient=[], n_rotate=0, rotate=[],
            n_translate=0, translate=[], equal_methyls=False, basis='6-31g*',
            method='scf', psi4_options={}, hyp_a1=0.0005, hyp_a2=0.001,
            load_files=False, **kwargs):
        """
        Parameters
        ----------
        stage_2: bool (optional)
            Whether to perform a 2-stage RESP fit
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
        equal_methyls: bool (optional)
            if ``True``, all carbons in methyl groups are constrained to be
            equivalent; all carbons in methylenes are equivalent; all hydrogens
            in methyls are equivalent; and all hydrogens in methylenes are
            equivalent.
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
        load_files: bool (optional)
            If ``True``, tries to load ESP and grid data from file.
        **kwargs:
            arguments passed to MultiResp.fit

        Returns
        -------
        charges: list of ndarrays
        """

        if opt:
            for mol in self.molecules:
                mol.optimize_geometry(method=method, basis=basis,
                                      psi4_options=psi4_options,
                                      save_opt_geometry=save_opt_geometry)

        self.add_orientations(n_orient=n_orient, orient=orient,
                              n_rotate=n_rotate, rotate=rotate,
                              n_translate=n_translate,
                              translate=translate, load_files=load_files)

        intra_chrconstr = self._nmol_values(intra_chrconstr,
                                            'intra_chrconstr',
                                            'list or dict of constraints')
        intra_chrequiv = self._nmol_values(intra_chrequiv,
                                           'intra_chrequiv',
                                           'list of equivalence constraints')

        if stage_2:  # do intra-constraints in stage 2 only
            stage_2_equiv = intra_chrequiv
            intra_chrequiv = [[] for i in range(self.n_molecules)]
        else:
            stage_2_equiv = [[] for i in range(self.n_molecules)]

        if equal_methyls and not stage_2:
            equivs = self.get_methyl_constraints()
            intra_chrequiv = [x+y for x, y in zip(intra_chrequiv, equivs)]

        qs = self.fit(intra_chrconstr=intra_chrconstr,
                      intra_chrequiv=intra_chrequiv,
                      inter_chrconstr=inter_chrconstr,
                      inter_chrequiv=inter_chrequiv,
                      basis=basis, method=method, hyp_a=hyp_a1,
                      **kwargs)
        if stage_2:
            cs = self.get_stage2_constraints(qs, equal_methyls=equal_methyls,
                                             intra_chrequiv=stage_2_equiv, inter_chrconstr=inter_chrconstr,
                                             intra_chrconstr=intra_chrconstr)
            intra_c, intra_e = cs
            qs = self.fit(intra_chrconstr=intra_c,
                          intra_chrequiv=intra_e,
                          basis=basis, method=method,
                          hyp_a=hyp_a2, **kwargs)
        return qs
