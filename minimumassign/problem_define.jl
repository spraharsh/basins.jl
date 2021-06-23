@doc raw"""
    Defines a minimization problem, given a potential and a seed
    
"""

include("../potentials/pele-interface.jl")
include("../utils/utils.jl")

cell_utils = pyimport("pele.utils.cell_scale")

"""
Makes InversePower Potential given specifications
"""
function make_IP_potential(natoms, dim, phi, power, eps, seed)
    radii_arr = generate_radii(seed, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
    length = get_box_length(radii_arr, phi, dim)
    boxvec = [length, length]
    cell_scale = cell_utils.get_ncellsx_scale(radii_arr, boxvec)
    return pot.InversePower(
        power,
        eps,
        radii_arr,
        ndim = dim,
        boxvec = boxvec,
        use_cell_lists = true,
        ncellx_scale = cell_scale,
    )
end

# Potentials we want to use for comparison
make_canonical_2d_IP_potential = natoms -> make_IP_potential(natoms, 2, 0.9, 2.5, 1.0, 0)

make_canonical_3d_IP_potential = natoms -> make_IP_potential(natoms, 3, 0.7, 2.5, 1.0, 0)
