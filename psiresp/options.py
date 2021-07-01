from collections import defaultdict
from typing import List, Dict

from .charge_constraints import ChargeConstraint, ChargeEquivalence
from .charge_constraints import AtomId
from . import mixins, base
from .generators import OrientationGenerator


class OrientationOptions(mixins.IOMixin):
    pass


class ConformerOptions(mixins.IOMixin):
    optimize_geometry: bool = False
    weight: float = 1
    orientation_options: OrientationOptions = OrientationOptions()
    orientation_generator: OrientationGenerator = OrientationGenerator()


class ChargeConstraintOptions(base.Model):
    """Options for setting charge constraints and charge equivalence constraints.

    Parameters
    ----------
    charge_constraints: list of dicts, tuples, or ChargeConstraints
        This is a list of all inputs accepted by the
        :class:`ChargeConstraint` class.
        It will be used to create constraints for a group of atoms to
        the given charge.
    charge_equivalences: list of lists, tuples, or ChargeEquivalences
        This is a list of all inputs accepted by the
        :class:`ChargeEquivalence` class.
        It will be used to create constraints so that each atom in the
        given group is constrained to the same charge.

    """
    charge_constraints: List[ChargeConstraint] = []
    charge_equivalences: List[ChargeEquivalence] = []
    symmetric_methyls: bool = True
    symmetric_methylenes: bool = True

    # def __post_init__(self):
    #     self.clean_charge_constraints()
    #     self.clean_charge_equivalences()

    @property
    def n_charge_constraints(self):
        return len(self.charge_constraints)

    @property
    def n_charge_equivalences(self):
        return len(self.charge_equivalences)

    def iterate_over_constraints(self):
        "Iterate first over charge constraints and then charge equivalences"
        for item in self.charge_constraints:
            yield item
        for item in self.charge_equivalences:
            yield item

    def _unite_overlapping_equivalences(self):
        """Join ChargeEquivalence constraints with overlapping atoms"""
        equivalences = defaultdict(set)
        for chrequiv in self.charge_equivalences:
            atom_set = set(chrequiv.atom_ids)
            for atom in atom_set:
                equivalences[atom] |= atom_set

        chrequivs = {tuple(sorted(x)) for x in equivalences.values()}
        self.charge_equivalences = [ChargeEquivalence(x) for x in chrequivs]

    def _get_single_atom_charge_constraints(self) -> Dict[AtomId, float]:
        """Get ChargeConstraints with only one atom as a dict"""
        single_charges = {}
        for constr in self.charge_constraints:
            if len(constr.atom_ids) == 1:
                atom_id = constr.atom_ids[0]
                if atom_id in single_charges:
                    err = ("Found conflicting charge constraints for "
                           f"atom {atom_id}, constrained to both "
                           f"{single_charges[atom_id]} and {constr.charge}")
                    raise ValueError(err)
                single_charges[atom_id] = constr.charge
        return single_charges

    def _remove_incompatible_and_redundant_equivalent_atoms(self):
        """Remove atoms from charge equivalences if they are constrained
        to different charges, and remove charge equivalence constraints
        if all atoms are constrained to the same charge (so it is redundant)
        """
        # if a charge equivalence has multiple atoms constrained to
        # single, different, charges, remove those.
        single_charges = self._get_single_atom_charge_constraints()
        redundant = []
        for i_eq, chrequiv in enumerate(self.charge_equivalences):

            indices, charges = zip(*[(i, single_charges[x])
                                     for i, x in enumerate(chrequiv.atom_ids)
                                     if x in single_charges])

            if len(charges) > 1:
                for i in indices[::-1]:
                    # TODO: silently delete or raise an error?
                    del chrequiv[i]
            # every atom in the equivalence is constrained to the same charge
            # this is redundant and should get removed
            # or can result in singular matrices
            elif len(charges) == len(chrequiv.atom_ids):
                redundant.append(i_eq)
        for i in redundant[::-1]:
            del self.equivalences[i]

    def clean_charge_equivalences(self):
        """Clean the ChargeEquivalence constraints.

        1. Join charge equivalence constraints with overlapping atoms
        2. Remove atoms from charge equivalences if they are constrained
        to different charges, and remove charge equivalence constraints
        if all atoms are constrained to the same charge (so it is redundant)
        """
        self._unite_overlapping_equivalences()
        self._remove_incompatible_and_redundant_equivalent_atoms()

    def clean_charge_constraints(self):
        """Clean the ChargeConstraints.

        1. Removes duplicates
        2. Checks that there are no atoms constrained to two different charges
        """
        # remove duplicates
        self.charge_constraints = list(set(self.charge_constraints))
        # this will check for duplicate conflicting charges as a side effect
        self._get_single_atom_charge_constraints()

    def get_constraint_matrix(self, a_matrix, b_matrix):
        """Create full constraint matrix from input matrices and charge constraints.

        A and B are the matrices used to solve Ax = B.

        Parameters
        ----------
        a_matrix: numpy.ndarray
            Matrix of shape (N, N)
        b_matrix: numpy.ndarray
            Matrix of shape (N,)

        Returns
        -------
        A: numpy.ndarray
            Overall matrix of constraints, shape (M, M).
            M = N + number_of_charge_constraints + number_of_equivalent_atom_pairs
        B: numpy.ndarray
            Overall solution vector, shape (M,)

        """
        n_dim = a_matrix.shape[0]
        col_constraints = [c.to_coo_cols(n_dim)
                           for c in self.iterate_over_constraints()]
        col_block = scipy.sparse.hstack(col_constraints, format="coo")
        a_block = scipy.sparse.coo_matrix(a_matrix)
        a_sparse = scipy.sparse.bmat([[a_block, col_block],
                                      [col_block.transpose(), None]])
        b_dense = np.r_[b_matrix, [c.charge for c in self.charge_constraints]]
        b_sparse = np.zeros(a_sparse.shape[0])
        b_sparse[:len(b_dense)] = b_dense
        return a_sparse, b_sparse

    def add_sp3_equivalences(self, sp3_ch_ids: Dict[int, List[int]] = {}):
        """
        Add ChargeEquivalences for the hydrogens attached to sp3 carbons

        This will add methyls if ``symmetric_methyls`` is True,
        and methylenes if ``symmetric_methylenes`` is True.

        Parameters
        ----------
        sp3_ch_ids: dictionary of {int: list[int]}
            A dictionary of atom numbers.
            Atom numbers are indices, indexed from 1.
            Keys are the atom numbers of carbons with 4 bonds.
            Values are the numbers of the hydrogens bonded to these carbons.
        """
        accepted = []
        if self.symmetric_methyls:
            accepted.append(3)
        if self.symmetric_methylenes:
            accepted.append(2)
        if not accepted:
            return
        for hs in sp3_ch_ids.values():
            if len(hs) in accepted:
                self.charge_equivalences.append(ChargeEquivalence(hs))

    def add_stage_2_constraints(self, charges=[]):
        """Add ChargeConstraints restraining atoms to the given charges,
        if they are not in charge equivalence constraints.

        Parameters
        ----------
        charges: iterable of floats
            Charges
        """
        charges = np.asarray(charges)
        equivalent_atom_ids = np.array([i for eq in self.charge_equivalences
                                        for i in eq.atom_ids])
        all_atom_ids = np.arange(len(charges), dtype=int) + 1
        mask = ~np.in1d(all_atom_ids, equivalent_atom_ids)

        for q, a in zip(charges[mask], all_atom_ids[mask]):
            constr = ChargeConstraint(charge=q, atom_ids=[a])
            self.charge_constraints.append(constr)

        self.clean_charge_constraints()
        self.clean_charge_equivalences()
