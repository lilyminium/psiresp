import numpy as np

from .conformer import Conformer
from .resp import Resp
from .multiresp import MultiResp
from . import utils
from .due import due, Doi

@due.dcite(
    Doi('10.1038/s42004-020-0291-4'),
    description='RESP2',
    path='psiresp.resp2',
)
class Resp2(object):
    """
    Class to manage Resp2 for one molecule of multiple conformers.

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
    gas: Resp
        Resp class of molecule in gaseous phase
    solv: Resp
        Resp class of molecule in aqueous phase
    gas_charges: ndarray of floats
        RESP charges in gas phase (only exists after calling run)
    solv_charges: ndarray of floats
        RESP charges in aqueous phase (only exists after calling run)
    charges: ndarray of floats
        Resp2 charges (only exists after calling run)
    """

    @classmethod
    def from_molecules(cls, molecules, charge=0, multiplicity=1, name=None,
                       orient=[], rotate=[], translate=[], chrconstr=[],
                       chrequiv=[]):
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
            names = np.where(molnames == 'default', molnames, gennames)

        conformers = []
        for mol, name in zip(molecules, names):
            conformers.append(Conformer(mol.clone(), name=name, charge=charge,
                                        multiplicity=multiplicity,
                                        orient=orient, rotate=rotate,
                                        translate=translate))

        return cls(conformers, name=name,
                   chrconstr=chrconstr, chrequiv=chrequiv)

    def __init__(self, conformers, name=None, chrconstr=[],
                 chrequiv=[]):
        if name is None:
            name = 'Resp'
        self.name = name
        self.gas = Resp(conformers).clone(name=name+'_gas')
        self.solv = self.gas.clone(name=name+'_solv')
        self._gas_charges = self._solv_charges = self._charges = None
        if chrconstr is not None:
            self.gas.add_charge_constraints(chrconstr)
        if chrequiv is not None:
            self.solv.add_charge_equivalences(chrequiv)

    @property
    def gas_charges(self):
        if self._gas_charges is None:
            raise ValueError('No gas charges available. Call run()')
        return self._gas_charges

    @property
    def solv_charges(self):
        if self._solv_charges is None:
            raise ValueError('No solv charges available. Call run()')
        return self._solv_charges

    @property
    def charges(self):
        if self._charges is None:
            raise ValueError('No Resp2 charges available. Call run()')
        return self._charges

    def run(self, opt=False, save_opt_geometry=False, delta=0.6,
            chrconstr=[], chrequiv=[], weights=1, method='PW6B95',
            basis='aug-cc-pV(D+d)Z', vdw_radii={}, psi4_options={},
            rmin=1.3, rmax=2.1, save_files=False, n_orient=0, orient=[],
            n_rotate=0, rotate=[], n_translate=0, translate=[],
            equal_methyls=False, tol=1e-6, maxiter=50, load_files=False):
        """
        Perform a 2-stage RESP2 fit.

        Parameters
        ----------
        opt: bool (optional)
            Whether to optimise the geometry of each conformer
        save_opt_geometry: bool (optional)
            if ``True`` , writes optimised geometries to an XYZ file
        delta: float (optional)
            mixing parameter for aqueous and gaseous charges. delta=0.6 
            generates charges from 60% aqueous and 40% gaseous charges.
        chrconstr: list or dict (optional)
            Intramolecular charge constraints in the form of
            {charge: atom_number_list} or [(charge, atom_number_list)].
            The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]]
            mean that atoms 1 and 2 together have a charge of 0.
        chrequiv: list (optional)
            lists of atoms with equivalent charges.
            e.g. [[1, 2], [3, 4, 5]] mean that atoms 1 and 2 have equal
            charges, and atoms 3, 4, and 5 have equal charges.
        weights: iterable (optional)
            weights of each conformer. If only one number is given, this is 
            repeated for each conformer.
        method: str (optional)
            Method to compute ESP
        basis: str (optional)
            Basis set to compute ESP
        vdw_radii: dict (optional)
            van der Waals' radii. If elements in the molecule are not
            defined in the chosen ``use_radii`` set, they must be given here.
        psi4_options: dict (optional)
            additional Psi4 options
        rmin: float (optional)
            inner boundary of shell to keep grid points from
        rmax: float (optional)
            outer boundary of shell to keep grid points from. If < 0,
            all points are selected.
        save_files: bool (optional)
            if ``True``, Psi4 files are saved and the computed ESP
            and grids are written to files.
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
        tol: float (optional)
            threshold for fitting convergence
        maxiter: int (optional)
            maximum number of iterations in fitting
        load_files: bool (optional)
            If ``True``, tries to load ESP and grid data from file.

        Returns
        -------
        charges: ndarray
        """
        if opt:
            for m, b in [('hf', '6-31g*'),
                         ('hf', basis),
                         (method, basis)]:
                self.gas.optimize_geometry(method=m, basis=b,
                                           psi4_options=psi4_options,
                                           save_opt_geometry=save_opt_geometry)

            self.solv = self.gas.clone(name=self.name+'_solv')

        self.gas.add_orientations(n_orient=n_orient, orient=orient,
                                  n_rotate=n_rotate, rotate=rotate,
                                  n_translate=n_translate,
                                  translate=translate)

        self.solv.add_orientations(n_orient=n_orient, orient=orient,
                                   n_rotate=n_rotate, rotate=rotate,
                                   n_translate=n_translate,
                                   translate=translate)

        self._gas_charges = self.gas.run(stage_2=True, opt=False,
                                         chrconstr=chrconstr,
                                         chrequiv=chrequiv,
                                         weights=weights, use_radii='bondi',
                                         vdw_scale_factors=(1.4, 1.6, 1.8, 2.0),
                                         vdw_point_density=2.5,
                                         vdw_radii=vdw_radii,
                                         rmin=rmin, rmax=rmax, method=method,
                                         basis=basis, solvent=None,
                                         restraint=True,
                                         psi4_options=psi4_options,
                                         hyp_a1=0.0005, hyp_a2=0.001,
                                         equal_methyls=equal_methyls,
                                         ihfree=True, tol=tol, maxiter=maxiter,
                                         save_files=save_files,
                                         load_files=load_files)

        self._solv_charges = self.solv.run(stage_2=True, opt=False,
                                           chrconstr=chrconstr,
                                           chrequiv=chrequiv,
                                           weights=weights, use_radii='bondi',
                                           vdw_scale_factors=(1.4, 1.6, 1.8, 2.0),
                                           vdw_point_density=2.5,
                                           vdw_radii=vdw_radii,
                                           rmin=rmin, rmax=rmax,
                                           method=method,
                                           basis=basis, solvent='water',
                                           restraint=True,
                                           psi4_options=psi4_options,
                                           hyp_a1=0.0005, hyp_a2=0.001,
                                           equal_methyls=equal_methyls,
                                           ihfree=True, tol=tol, maxiter=maxiter,
                                           save_files=save_files,
                                           load_files=load_files)

        self._charges = delta*self.solv_charges + (1-delta)*self.gas_charges
        return self._charges

@due.dcite(
    Doi('10.1038/s42004-020-0291-4'),
    description='RESP2 multi-molecule fit',
    path='psiresp.resp2',
)
class MultiResp2(object):
    """
    Class to manage Resp2 for multiple molecules of multiple conformers.

    Parameters
    ----------
    resps: list of Resp
        Molecules for multi-molecule fit, set up in Resp classes.

    Attributes
    ----------
    gas: Resp
        MultiResp class of molecules in gaseous phase
    solv: Resp
        MultiResp class of molecules in aqueous phase
    gas_charges: ndarray of floats
        RESP charges in gas phase (only exists after calling run)
    solv_charges: ndarray of floats
        RESP charges in aqueous phase (only exists after calling run)
    charges: ndarray of floats
        Resp2 charges (only exists after calling run)
    """

    def __init__(self, resps):
        cresps = [r.clone(name=r.name+'_gas') for r in resps]
        self.gas = MultiResp(cresps)
        self.solv = self.gas.clone(suffix='_solv')
        self._gas_charges = self._solv_charges = self._charges = None

    @property
    def gas_charges(self):
        if self._gas_charges is None:
            raise ValueError('No gas charges available. Call run()')
        return self._gas_charges

    @property
    def solv_charges(self):
        if self._solv_charges is None:
            raise ValueError('No solv charges available. Call run()')
        return self._solv_charges

    @property
    def charges(self):
        if self._charges is None:
            raise ValueError('No Resp2 charges available. Call run()')
        return self._charges

    def run(self, opt=False, save_opt_geometry=False, delta=0.6,
            intra_chrconstr=[], intra_chrequiv=[], inter_chrconstr=[],
            inter_chrequiv=[], weights=1, method='PW6B95',
            basis='aug-cc-pV(D+d)Z', vdw_radii={}, psi4_options={},
            rmin=1.3, rmax=2.1, save_files=False, n_orient=0, orient=[],
            n_rotate=0, rotate=[], n_translate=0, translate=[],
            equal_methyls=False, tol=1e-6, maxiter=50, load_files=False):
        """
        Perform a 2-stage RESP2 fit.

        Parameters
        ----------
        opt: bool (optional)
            Whether to optimise the geometry of each conformer
        save_opt_geometry: bool (optional)
            if ``True``, writes optimised geometries to an XYZ file
        delta: float (optional)
            mixing parameter for aqueous and gaseous charges. delta=0.6 
            generates charges from 60% aqueous and 40% gaseous charges.
        intra_chrconstr: list of lists or dicts (optional)
            Intramolecular charge constraints for each molecule in the 
            form of {charge: atom_number_list} or [(charge, atom_number_list)]. 
            The numbers are indexed from 1. e.g. {0: [1, 2]} or [[0, [1, 2]]] 
            mean that atoms 1 and 2 together have a charge of 0.
        intra_chrequiv: list of lists (optional)
            Lists of atoms with equivalent charges within each molecule. e.g. ::
            
                [
                  [[1, 2], [3, 4, 5]],
                  [[1, 3, 5, 7]]
                 ] 
                 
            mean that atoms 1 and 2 in the first molecule have equal 
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
            weights of each conformer. If only one number is given, this is 
            repeated for each conformer.
        method: str (optional)
            Method to compute ESP
        basis: str (optional)
            Basis set to compute ESP
        vdw_radii: dict (optional)
            van der Waals' radii. If elements in the molecule are not
            defined in the chosen ``use_radii`` set, they must be given here.
        psi4_options: dict (optional)
            additional Psi4 options
        rmin: float (optional)
            inner boundary of shell to keep grid points from
        rmax: float (optional)
            outer boundary of shell to keep grid points from. If < 0,
            all points are selected.
        save_files: bool (optional)
            if ``True`` , Psi4 files are saved and the computed ESP
            and grids are written to files.
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
        tol: float (optional)
            threshold for fitting convergence
        maxiter: int (optional)
            maximum number of iterations in fitting
        load_files: bool (optional)
            If ``True``, tries to load ESP and grid data from file.

        Returns
        -------
        charges: ndarray
        """

        if opt:
            for m, b in [('hf', '6-31g*'),
                         ('hf', basis),
                         (method, basis)]:
                self.gas.optimize_geometry(method=m, basis=b,
                                           psi4_options=psi4_options,
                                           save_opt_geometry=save_opt_geometry)

            self.solv = self.gas.clone(suffix='_solv')

        self.add_orientations(n_orient=n_orient, orient=orient,
                              n_rotate=n_rotate, rotate=rotate,
                              n_translate=n_translate, translate=translate)

        self._gas_charges = self.gas.run(stage_2=True, opt=False,
                                         intra_chrconstr=intra_chrconstr,
                                         intra_chrequiv=intra_chrequiv,
                                         inter_chrconstr=inter_chrconstr,
                                         inter_chrequiv=inter_chrequiv,
                                         weights=weights, use_radii='bondi',
                                         vdw_scale_factors=(1.4, 1.6, 1.8, 2.0),
                                         vdw_point_density=2.5, vdw_radii=vdw_radii,
                                         rmin=rmin, rmax=rmax, method=method,
                                         basis=basis, solvent=None, restraint=True,
                                         psi4_options=psi4_options, hyp_a1=0.0005,
                                         hyp_a2=0.001, equal_methyls=equal_methyls,
                                         ihfree=True, tol=tol, maxiter=maxiter,
                                         save_files=save_files,
                                         load_files=load_files)

        self._solv_charges = self.solv.run(stage_2=True, opt=False,
                                           intra_chrconstr=intra_chrconstr,
                                           intra_chrequiv=intra_chrequiv,
                                           inter_chrconstr=inter_chrconstr,
                                           inter_chrequiv=inter_chrequiv,
                                           weights=weights, use_radii='bondi',
                                           vdw_scale_factors=(1.4, 1.6, 1.8, 2.0),
                                           vdw_point_density=2.5, vdw_radii=vdw_radii,
                                           rmin=rmin, rmax=rmax, method=method,
                                           basis=basis, solvent='water', restraint=True,
                                           psi4_options=psi4_options, hyp_a1=0.0005,
                                           hyp_a2=0.001, equal_methyls=equal_methyls,
                                           ihfree=True, tol=tol, maxiter=maxiter,
                                           save_files=save_files,
                                           load_files=load_files)
        charges = []
        for solv, gas, sresp, gresp in zip(self.solv_charges, self.gas_charges,
                                           self.gas.molecules, self.solv.molecules):
            sresp.charges = gresp.charges = q = delta*solv + (1-delta)*gas
            charges.append(q)

        self._charges = charges
        return charges
