include("../src/minimumassign/minimumassign.jl")
include("../src/utils/utils.jl")
include("../src/potentials/pele-interface.jl")


natoms = 8
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
dim = 2
phi = 0.9
power = 2.5
eps = 1
length_arr = get_box_length(radii_arr, phi, dim)
coords = generate_random_coordinates(length_arr, natoms, dim)
using PyCall
boxvec = [length_arr, length_arr]
utils = pyimport("pele.utils.cell_scale")
cell_scale = utils.get_ncellsx_scale(radii_arr, boxvec)


pele_wrapped_pot_2 = pot.InversePower(
    2.5,
    1.0,
    radii_arr,
    ndim = 2,
    boxvec = boxvec,
    use_cell_lists = false,
    ncellx_scale = cell_scale,
)
pele_wrapped_python_pot = PythonPotential(pele_wrapped_pot_2)

func = gradient_problem_function_pele!(pele_wrapped_python_pot)


ba = BasinAssigner(10^-8, 10^-8, 10^-5)


res = find_corresponding_minimum(ba, func, coords, 10^5, pele_wrapped_python_pot)
