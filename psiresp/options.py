from collections import UserDict, UserList
from typing import Dict, List, Optional
import itertools
import textwrap
import os
import warnings
import functools

import numpy as np
from . import utils


class AttrDict(UserDict):
    def __init__(self, **kwargs):
        self.__dict__["data"] = {}
        self.update(kwargs)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


class IOOptions(AttrDict):
    def __init__(self,
                 force=False,
                 verbose: bool = False,
                 write_to_files: bool = False,
                 load_from_files: bool = False):
        super().__init__(force=force, verbose=verbose, write_to_files=write_to_files, load_from_files=load_from_files)

    def try_load_data(self, path):
        suffix = path.split(".")[-1]

        if not self.force and self.load_from_files:
            if suffix == "csv":
                loader = utils.read_csv
            elif suffix in ("dat", "txt"):
                loader = np.loadtxt
            elif suffix in ("npy", "npz"):
                loader = np.load
            elif suffix in ("xyz", "pdb", "mol2"):
                loader = utils.load_text
            else:
                raise ValueError(f"Can't find loader for {suffix} file")

            try:
                data = loader(path)
            except Exception:
                if self.verbose:
                    print(f"Could not load data from {path}: (re)running.")
            else:
                if self.verbose:
                    print(f"Loaded from {path}.")
                return data, path
        return None, path

    def save_data(self, data, path, comments=None):
        if self.write_to_files:
            suffix = path.split(".")[-1]

            if suffix == "csv":
                data.to_csv(path)
            elif suffix in ("dat", "txt"):
                np.savetxt(path, data, comments=comments)
            elif suffix == "npy":
                np.save(path, data)
            elif suffix == "npz":
                np.savez(path, **data)
            elif suffix == "xyz":
                if isinstance(data, str):
                    with open(path, "w") as f:
                        f.write(data)
                else:
                    data.save_xyz_file(path, True)
            else:
                raise ValueError(f"Can't find saver for {suffix} file")
            if self.verbose:
                print("Saved to", path)


class QMOptions(AttrDict):
    def __init__(self,
                 method: str = "scf",
                 basis: str = "6-31g*",
                 solvent: Optional[str] = None,
                 geom_maxiter: int = 200,
                 full_hess_every: int = 10,
                 g_convergence: str = "gau_tight",
                 **kwargs):
        super().__init__(method=method,
                         basis=basis,
                         solvent=solvent,
                         geom_maxiter=geom_maxiter,
                         full_hess_every=full_hess_every,
                         g_convergence=g_convergence)

    def write_opt_file(self, psi4mol, destination_dir=".", filename="opt.in"):
        opt_file = utils.create_psi4_molstr(psi4mol)
        opt_file += textwrap.dedent(f"""
        set {{
            basis {self.basis}
            geom_maxiter {self.geom_maxiter}
            full_hess_every {self.full_hess_every}
            g_convergence {self.g_convergence}
        }}

        optimize('{self.method}')
        """)

        inpath = os.path.join(destination_dir, filename)

        with open(inpath, "w") as f:
            f.write(opt_file)

        outfile = Path(inpath).stem + ".out"
        return outfile

    def write_esp_file(self, psi4mol, destination_dir=".", filename="esp.in"):
        esp_file = utils.create_psi4_molstr(psi4mol)

        esp_file += f"set basis {self.basis}\n"

        if self.solvent:
            esp_file += textwrap.dedent(f"""
            set {{
                pcm true
                pcm_scf_type total
            }}

            pcm = {{
                Units = Angstrom
                Medium {{
                    SolverType = CPCM
                    Solvent = {self.solvent}
                }}

                Cavity {{
                    RadiiSet = bondi # Bondi | UFF | Allinger
                    Type = GePol
                    Scaling = True # radii for spheres scaled by 1.2
                    Area = 0.3
                    Mode = Implicit
                }}
            }}

            """)

        esp_file += textwrap.dedent(f"""\
        E, wfn = prop('{self.method}', properties=['GRID_ESP'], return_wfn=True)
        esp = wfn.oeprop.Vvals()
            """)

        with open(os.path.join(destination_dir, filename), "w") as f:
            f.write(esp_file)

        outfile = os.path.join(destination_dir, "grid_esp.dat")
        return outfile


class ESPOptions(AttrDict):
    def __init__(self,
                 rmin: float = 0,
                 rmax: float = -1,
                 use_radii: str = "msk",
                 vdw_radii: Dict[str, float] = {},
                 vdw_scale_factors: List[float] = [1.4, 1.6, 1.8, 2.0],
                 vdw_point_density: float = 1.0):
        super().__init__(rmin=rmin,
                         rmax=rmax,
                         use_radii=use_radii,
                         vdw_scale_factors=vdw_scale_factors,
                         vdw_radii=vdw_radii,
                         vdw_point_density=vdw_point_density)


class OrientationOptions(AttrDict):
    def __init__(self,
                 n_reorientations: int = 0,
                 reorientations=[],
                 n_translations: int = 0,
                 translations=[],
                 n_rotations: int = 0,
                 rotations=[],
                 keep_original=True):
        reorientations = list(reorientations)
        translations = list(translations)
        rotations = list(rotations)
        super().__init__(n_reorientations=n_reorientations,
                         n_translations=n_translations,
                         n_rotations=n_rotations,
                         reorientations=reorientations,
                         translations=translations,
                         rotations=rotations,
                         keep_original=keep_original)

    @property
    def n_specified_orientations(self):
        return sum(map(len, [self.reorientations, self.rotations, self.translations]))

    @property
    def n_orientations(self):
        return sum([self.n_specified_orientations, self.n_reorientations, self.n_translations, self.n_rotations])

    def generate_atom_combinations(self, symbols: List[str]):
        symbols = np.asarray(symbols)
        is_H = symbols == "H"
        h_atoms = list(np.flatnonzero(is_H) + 1)
        heavy_atoms = list(np.flatnonzero(~is_H) + 1)

        comb = list(itertools.combinations(heavy_atoms, 3))
        h_comb = itertools.combinations(heavy_atoms + h_atoms, 3)
        comb += [x for x in h_comb if x not in comb]
        backwards = [x[::-1] for x in comb]
        new_comb = [x for items in zip(comb, backwards) for x in items]
        return new_comb

    def generate_orientations(self, symbols: List[str]):
        atom_combinations = self.generate_atom_combinations(symbols)
        for kw in ("reorientations", "rotations"):
            n = max(self[f"n_{kw}"] - len(self[kw]), 0)
            self[kw].extend(atom_combinations[:n])
        n_trans = self.n_translations - len(self.translations)
        if n_trans > 0:
            new_translations = (np.random.rand(n_trans, 3) - 0.5) * 10
            self.translations.extend(new_translations)

    @staticmethod
    def id_to_indices(atom_ids):
        return [a - 1 if a > 0 else a for a in atom_ids]

    def to_indices(self):
        dct = {"translations": self.translations}
        dct["reorientations"] = [self.id_to_indices(x) for x in self.reorientations]
        dct["rotations"] = [self.id_to_indices(x) for x in self.rotations]
        return dct


@functools.total_ordering
class AtomId:
    def __init__(self, molecule_id=1, atom_id=None, atom_increment=0):
        if isinstance(molecule_id, AtomId):
            atom_id = molecule_id.atom_id
            atom_increment = molecule_id.atom_increment
            molecule_id = molecule_id.molecule_id
        else:
            if atom_id is None:
                if isinstance(molecule_id, (list, tuple)) and len(molecule_id) == 2:
                    atom_id = molecule_id[1]
                    molecule_id = molecule_id[0]
                else:
                    atom_id = molecule_id
                    molecule_id = 1
        self.atom_id = atom_id
        self.molecule_id = molecule_id
        self.atom_increment = atom_increment

    def __lt__(self, other):
        if isinstance(other, AtomId):
            other = other.absolute_atom_id
        return self.absolute_atom_id < other

    def __eq__(self, other):
        if isinstance(other, AtomId):
            other = other.absolute_atom_id
        return self.absolute_atom_id == other

    def __hash__(self):
        return hash((self.atom_id, self.molecule_id, self.atom_increment))

    @property
    def absolute_atom_id(self):
        return self.atom_increment + self.atom_id

    @property
    def absolute_atom_index(self):
        return self.absolute_atom_id - 1

    def copy_with_molecule_id(self, molecule_id=1, atom_increment=0):
        new = type(self)(molecule_id=molecule_id, atom_id=self.atom_id)
        new.atom_increment = atom_increment
        return new


class BaseChargeConstraint(UserList):
    def __init__(self, atom_ids: list = []):
        atom_ids = [AtomId(x) for x in atom_ids]
        atom_ids = sorted(set(atom_ids))
        super().__init__(atom_ids)

    @property
    def atom_ids(self):
        return self.data

    @property
    def absolute_atom_ids(self):
        return np.array([x.absolute_atom_id for x in self.data])

    @property
    def indices(self):
        return self.absolute_atom_ids - 1

    def __len__(self):
        return len(self.atom_ids)

    def copy_atom_ids_to_molecule(self, molecule_id=1, atom_increment=0):
        atom_ids = []
        for aid in self.atom_ids:
            atom_ids.append(aid.copy_with_molecule_id(molecule_id=molecule_id, atom_increment=atom_increment))
        return atom_ids


class ChargeConstraint(BaseChargeConstraint):
    def __repr__(self):
        return f"<ChargeConstraint charge={self.charge}, indices={self.indices}>"

    @classmethod
    def from_obj(cls, obj):
        if isinstance(obj, dict):
            if len(obj) != 1:
                raise ValueError("dict must have only one key-value pair " "in charge: [atom_ids] format.")
            obj = list(obj.items())[0]
        elif isinstance(obj, ChargeConstraint):
            obj = [obj.charge, obj.atom_ids]
        return cls(charge=obj[0], atom_ids=obj[1])

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        args = (self.charge, tuple(sorted(self.atom_ids)))
        return hash(args)

    def __init__(self, charge: float = 0, atom_ids: list = []):
        self.charge = charge
        super().__init__(atom_ids=atom_ids)

    def copy_with_molecule_id(self, molecule_id=1, atom_increment=0):
        atom_ids = self.copy_atom_ids_to_molecule(molecule_id=molecule_id, atom_increment=atom_increment)
        return type(self)(charge=self.charge, atom_ids=atom_ids)


class ChargeEquivalence(BaseChargeConstraint):
    def __repr__(self):
        return f"<ChargeEquivalence indices={self.indices}>"

    def __init__(self, atom_ids: list = []):
        super().__init__(atom_ids=atom_ids)
        if not len(self.atom_ids) >= 2:
            raise ValueError("Must have at least 2 different atoms in a " "charge equivalence constraint")

    def __add__(self, other):
        return type(self)(np.concatenate([self.atom_ids, other.atom_ids]))

    def __radd__(self, other):
        if other == 0:
            return self
        return other.__add__(self)

    def copy_with_molecule_id(self, molecule_id=1, atom_increment=0):
        atom_ids = self.copy_atom_ids_to_molecule(molecule_id=molecule_id, atom_increment=atom_increment)
        return type(self)(atom_ids=atom_ids)


class ChargeOptions(AttrDict):

    # @classmethod
    # def from_multiresp(cls, resps, charge_constraints=[], charge_equivalences=[]):

    def __init__(self,
                 charge_constraints=[],
                 charge_equivalences=[],
                 equivalent_methyls=False,
                 equivalent_sp3_hydrogens=True):
        if isinstance(charge_constraints, dict):
            charge_constraints = list(charge_constraints.items())
        chrconstr = [ChargeConstraint.from_obj(x) for x in charge_constraints]
        chrequiv = [ChargeEquivalence(x) for x in charge_equivalences]
        super().__init__(charge_constraints=chrconstr,
                         charge_equivalences=chrequiv,
                         equivalent_methyls=equivalent_methyls,
                         equivalent_sp3_hydrogens=equivalent_sp3_hydrogens)
        self.clean_charge_constraints()
        self.clean_charge_equivalences()

    def iterate_over_constraints(self):
        for item in self.charge_constraints:
            yield item
        for item in self.charge_equivalences:
            yield item

    def clean_charge_equivalences(self):
        atom_ids = []
        equivalences = []

        # first collapse items with overlapping atoms
        for equiv in self.charge_equivalences:
            atom_set = set(equiv.atom_ids)
            for i, seen_set in enumerate(atom_ids):
                if atom_set.intersection(seen_set):
                    equivalences[i].append(equiv)
                    seen_set.update(atom_set)
                    break
            else:
                atom_ids.append(atom_set)
                equivalences.append([equiv])
        chrequivs = [sum(x) for x in equivalences]

        # then check charge constraints
        # if an equivalence has >1 atom that is restrained
        # to a charge (not in a group) and they're different, remove those atoms
        single_charges = {}
        for constr in self.charge_constraints:
            if len(constr.atom_ids) == 1:
                single_charges[constr.atom_ids[0]] = constr.charge

        for equiv in chrequivs:
            single_atoms = [i for i, x in enumerate(equiv.atom_ids) if x in single_charges]
            charges = [single_charges[equiv[i]] for i in single_atoms]
            if len(set(charges)) > 1:
                for i in single_atoms[::-1]:
                    del equiv[i]

        self.charge_equivalences = chrequivs

    def clean_charge_constraints(self):
        # remove duplicates
        unique = set(self.charge_constraints)
        # error on conflicting constraints
        single_atoms = {}
        for constr in unique:
            if len(constr.atom_ids) == 1:
                atom_id = constr.atom_ids[0]
                if atom_id in single_atoms:
                    raise ValueError("Found conflicting charge constraints for "
                                     f"atom {atom_id} to {single_atoms[atom_id]} "
                                     f"and {constr.charge}")
        self.charge_constraints = list(unique)

    def get_constraint_matrix(self, a_matrix, b_matrix):
        # preprocessing
        equiv = [x.indices for x in self.charge_equivalences]
        edges = np.r_[0, np.cumsum([len(x) - 1 for x in equiv])].astype(int)

        # get dimensions
        n_chrequiv = edges[-1]
        n_chrconstr = len(self.charge_constraints)
        n_conf_dim = a_matrix.shape[0]
        ndim = n_chrequiv + n_chrconstr + n_conf_dim

        A = np.zeros((ndim, ndim))
        B = np.zeros(ndim)
        A[:n_conf_dim, :n_conf_dim] = a_matrix
        B[:n_conf_dim] = b_matrix

        for i, chrconstr in enumerate(self.charge_constraints, n_conf_dim):
            B[i] = chrconstr.charge
            A[i, chrconstr.indices] = A[chrconstr.indices, i] = 1

        row_inc = n_conf_dim + n_chrconstr
        for i, indices in enumerate(equiv):
            x = np.arange(edges[i], edges[i + 1]) + row_inc
            A[(x, indices[:-1])] = A[(indices[:-1], x)] = -1
            A[(x, indices[1:])] = A[(indices[1:], x)] = 1

        return A, B

    def add_methyl_equivalences(self, sp3_ch_ids={}):
        c3s = []
        c2s = []
        h3s = []
        h2s = []
        for c, hs in sp3_ch_ids.items():
            if len(hs) == 3:
                c3s.append(c)
                h3s.extend(hs)
            elif len(hs) == 2:
                c2s.append(c)
                h2s.extend(hs)

        equivs = [c3s, c2s, h3s, h2s]
        for x in equivs:
            if len(x) > 1:
                self.charge_equivalences.append(ChargeEquivalence(x))
        self.clean_charge_equivalences()

    def add_stage_2_constraints(self, charges=[], sp3_ch_ids={}):
        charges = np.asarray(charges)
        atom_ids = [i for eq in self.charge_equivalences for i in eq.atom_ids]
        if self.equivalent_sp3_hydrogens:
            hs = [y for x in sp3_ch_ids.values() for y in x]
            cs = list(sp3_ch_ids.keys())
            atom_ids = atom_ids + hs + cs
        atom_ids = np.array(atom_ids)
        ids = np.arange(len(charges)) + 1
        mask = ~np.in1d(ids, atom_ids)

        for q, a in zip(charges[mask], ids[mask]):
            constr = ChargeConstraint(charge=q, atom_ids=[a])
            self.charge_constraints.append(constr)

        if not self.equivalent_methyls and self.equivalent_sp3_hydrogens:
            # if self.equivalent_methyls, this is redundant
            # equivs = []
            for hs in sp3_ch_ids.values():
                if len(hs) > 1:
                    self.charge_equivalences.append(ChargeEquivalence(hs))
            # equivs = [ChargeEquivalence(list(hs)) for hs in sp3_ch_ids.values()]
            # self.charge_equivalences.extend(equivs)
            # self.charge_equivalences.extend([ChargeEquivalence(list) for x in sp3_ch_ids.values()])
            self.clean_charge_equivalences()
        self.clean_charge_constraints()


class RespOptions(AttrDict):
    def __init__(self,
                 restrained: bool = True,
                 hyp_a: float = 0.0005,
                 hyp_b: float = 0.1,
                 ihfree: bool = True,
                 tol: float = 1e-6,
                 maxiter: int = 50):
        super().__init__(restrained=restrained, hyp_a=hyp_a, hyp_b=hyp_b, ihfree=ihfree, tol=tol, maxiter=maxiter)


class RespCharges:
    def __init__(self, resp_options=RespOptions(), symbols=[], n_structures: int = 1):

        self.resp_options = resp_options
        self.symbols = symbols
        self.n_structures = n_structures
        self.n_atoms = len(symbols)
        self.restrained_charges = None
        self.unrestrained_charges = None

    def iter_solve(self, charges, a_matrix, b_matrix):
        """
        Fit the charges iteratively, as required for the hyperbola penalty
        function.

        Parameters
        ----------
        charges: ndarray
            partial atomic charges
        a_matrix: ndarray
            unrestrained matrix A
        b_matrix: ndarray
            matrix B

        Returns
        -------
        charges: ndarray
        """
        if not self.resp_options.hyp_a:  # i.e. no restraint
            return charges

        mask = np.ones(self.n_atoms, dtype=bool)
        if self.resp_options.ihfree:
            h_indices = np.where(self.symbols == "H")[0]
            mask[h_indices] = False
        diag = np.diag_indices(self.n_atoms)
        ix = (diag[0][mask], diag[1][mask])
        indices = np.where(mask)[0]
        b2 = self.resp_options.hyp_b**2
        n_structures = self.n_structures[mask]

        niter, delta = 0, 2 * self.resp_options.tol
        while delta > self.resp_options.tol and niter < self.resp_options.maxiter:
            q_last, a_i = charges.copy(), a_matrix.copy()
            a_i[ix] = a_matrix[ix] + self.resp_options.hyp_a / np.sqrt(charges[indices]**2 + b2) * n_structures
            charges = np.linalg.solve(a_i, b_matrix)
            delta = np.max((charges - q_last)[:self.n_atoms]**2)**0.5
            niter += 1

        if delta > self.resp_options.tol:
            err = "Charge fitting did not converge with maxiter={}"
            warnings.warn(err.format(self.resp_options.maxiter))

        return charges

    def fit(self, a_matrix, b_matrix):
        q1 = np.linalg.solve(a_matrix, b_matrix)
        self.unrestrained_charges = q1[:self.n_atoms]
        if self.resp_options.restrained:
            q2 = self.iter_solve(q1, a_matrix, b_matrix)
            self.restrained_charges = q2[:self.n_atoms]
        return self.charges

    @property
    def charges(self):
        if self.resp_options.restrained:
            return self.restrained_charges
        return self.unrestrained_charges

    def copy(self, start_index=0, end_index=None, n_structures=None):
        if end_index is None:
            end_index = self.n_atoms
        if n_structures is None:
            n_structures = self.n_structures
        new = type(self)(resp_options=self.resp_options,
                         symbols=self.symbols[start_index:end_index],
                         n_structures=n_structures)
        if self.unrestrained_charges is not None:
            new.unrestrained_charges = self.unrestrained_charges[start_index:end_index]
        if self.restrained_charges is not None:
            new.restrained_charges = self.restrained_charges[start_index:end_index]
        return new
