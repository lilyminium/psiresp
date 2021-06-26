from .base import options, OptionsBase

@options
class GridOptions(OptionsBase):
    """Options for setting up the grid for ESP computation

    Parameters
    ----------
    rmin: float
        minimum radius
    rmax: float
        maximum radius
    use_radii: str
        Name of the radius set to use
    vdw_radii: dict of {str: float}
        Dictionary of VDW radii to override the radii in the
        `use_radii` set
    vdw_scale_factors: iterable of floats
        Scale factors for the radii
    vdw_point_density: float
        Point density
    """

    rmin: float = 0
    rmax: float = -1
    use_radii: str = "msk"
    vdw_radii: Dict[str, float] = {}
    vdw_scale_factors: List[float] = [1.4, 1.6, 1.8, 2.0]
    vdw_point_density: float = 1.0