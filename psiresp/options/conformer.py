from .. import mixins, base, rdutils
from .orientation import OrientationOptions, OrientationGenerator
class ConformerOptions(mixins.IOMixin):
    optimize_geometry: bool = False
    weight: float = 1
    orientation_options: OrientationOptions = OrientationOptions()
    orientation_generator: OrientationGenerator = OrientationGenerator()


class ConformerGenerator(base.Model):
    conformer_geometries: npt.ArrayLike = []
    max_generated_conformers: int = 0
    min_conformer_rmsd: float = 1.5
    minimize_conformer_geometries: bool = False
    minimize_max_iter: int = 2000
    keep_original_resp_geometry: bool = True
    name_template: str = "{resp.name}_{counter:03d}"

    def generate_conformer_geometries(self, psi4mol: psi4.core.Molecule):
        rdmol = rdutils.rdmol_from_psi4mol(psi4mol)
        self._generate_conformers_from_rdmol(rdmol)

    def _generate_conformers_from_rdmol(self, rdmol: rdkit.Chem.Mol):
        if not self.keep_original_resp_geometry:
            rdmol.RemoveAllConformers()
        
        for coordinates in self.conformer_geometries:
            rdutils.add_conformer_from_coordinates(rdmol, coordinates)

        rdutils.generate_conformers(rdmol,
                                    n_conformers=self.max_generated_conformers,
                                    rmsd_threshold=self.min_conformer_rmsd)
        if self.minimize_conformer_geometries:
            rdutils.minimize_conformer_geometries(rdmol,
                                                  self.minimize_max_iter)
        self.conformer_geometries = rdutils.get_conformer_coordinates(rdmol)

    
    def format_name(self, **kwargs):
        return self.name_template.format(**kwargs)
