import numpy as np

from . import base, mixins, options, utils


class Orientation(options.OrientationOptions, mixins.MoleculeMixin):
    """
    Class to manage one orientation of a conformer. This should
    not usually be created or interacted with by a user. Instead,
    users are expected to work primarily with
    :class:`psiresp.conformer.Conformer` or :class:`psiresp.resp.Resp`.

    Parameters
    ----------
    conformer: psiresp.Conformer
        The conformer that owns this orientation
    psi4mol: psi4.core.Molecule
        Psi4 molecule that forms the basis of this orientation
    name: str (optional)
        The name of this orientation

    Attributes
    ----------
    psi4mol: psi4.core.Molecule
        Psi4 molecule that forms the basis of this orientation
    conformer: psiresp.Conformer
        The conformer that owns this orientation
    name: str (optional)
        The name of this orientation
    grid: numpy.ndarray
        The grid of points on which to compute the ESP
    esp: numpy.ndarray
        The computed ESP at grid points
    r_inv: numpy.ndarray
        inverse distance from each grid point to each atom, in
        atomic units.
    """

    conformer: "Conformer"

    def __post_init__(self):
        super().__post_init__()
        self._grid = None
        self._esp = None
        self._r_inv = None

    @property
    def path(self):
        return self.conformer.path / self.name

    @property
    def resp(self):
        return self.conformer.resp

    @property
    def grid(self):
        if self._grid is None:
            self._grid = self.compute_grid()
        return self._grid

    @property
    def esp(self):
        if self._esp is None:
            self._esp = self.compute_esp()
        return self._esp

    @property
    def r_inv(self):
        if self._r_inv is None:
            self._r_inv = self.compute_r_inv()
        return self._r_inv

    def compute_r_inv(self) -> npt.NDArray:
        """Get inverse r"""
        points = self.grid.reshape((len(self.grid), 1, 3))
        disp = self.coordinates - points
        inverse = 1 / np.sqrt(np.einsum("ijk, ijk->ij", disp, disp))
        return inverse * utils.BOHR_TO_ANGSTROM

    def get_esp_mat_a(self) -> npt.NDArray:
        """Get A matrix for solving"""
        return np.einsum("ij, ik->jk", self.r_inv, self.r_inv)

    def get_esp_mat_b(self) -> npt.NDArray:
        """Get B matrix for solving"""
        return np.einsum("i, ij->j", self.esp, self.r_inv)

    @mixins.io.datafile(filename="grid.dat")
    def compute_grid(self):
        return self.resp.resp.generate_vdw_grid(self.symbols, self.coordinates)

    @mixins.io.datafile(filename="grid_esp.dat")
    def compute_esp(self):
        grid = self.grid
        if self.psi4mol_geometry_in_bohr:
            grid = grid * constants.ANGSTROM_TO_BOHR
        with self.directory() as tmpdir:
            # ... this dies unless you write out grid.dat
            np.savetxt("grid.dat", grid)
            infile = self.resp.resp.write_esp_file(self.psi4mol)
            self.resp.resp.try_run_qm(infile, cwd=tmpdir)
            esp = np.loadtxt("grid_esp.dat")
        self._esp = esp
        return esp
