using DiffEqBase: IterativeSolvers


include("../src/optimizer/newton.jl")
include("../src/minimumassign/mxopt.jl")

# using ProfileView
# using Profile

using StatProfilerHTML



natoms = 256
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
dim = 2
phi = 0.9
power = 2.5
eps = 1.0
length_arr = get_box_length(radii_arr, phi, dim)
coords = generate_random_coordinates(length_arr, natoms, dim)
using PyCall
boxvec = [length_arr, length_arr]
utils = pyimport("pele.utils.cell_scale")
cell_scale = utils.get_ncellsx_scale(radii_arr, boxvec)
println(cell_scale)



@info radii_arr, "radii_arr"
@info boxvec, "boxvec"
@info coords, "coords"
println(radii_arr)

pele_wrapped_pot_2 = pot.InversePower(
    2.5,
    1.0,
    radii_arr,
    ndim = 2,
    boxvec = boxvec,
    use_cell_lists = true,
    ncellx_scale = cell_scale,
)


pele_wrapped_python_pot_2 = PythonPotential(pele_wrapped_pot_2)
using Sundials
solver = CVODE_BDF(linear_solver=:GMRES)
nls = NewtonLinesearch(pele_wrapped_python_pot_2, coords, 1e-8)

mxd = Mixed_Descent(pele_wrapped_python_pot_2, solver, nls, coords, 10, 10^-5, 0.0, 10^-8)

rtol = 1e-4
atol = 1e-4
T = 20
convergence_tolerance = 1e-8
convexity_factor = 2.0
convexity_tolerance = 1e-8


mxd = Mixed_Descent(pele_wrapped_python_pot_2, solver, nls, coords, T, rtol, convexity_tolerance, convergence_tolerance)
mxd2 = Mixed_Descent(pele_wrapped_python_pot_2, solver, nls, coords, T, rtol, convexity_tolerance, convergence_tolerance)

# println(coords)


# one_iteration!(mxd)
# one_iteration!(mxd)
# println(mxd.potential.ngev)
# println(coords-mxd.integrator.u)

run!(mxd, 2000)
@info mxd.n_e_evals, mxd.n_g_evals, mxd.n_h_evals, mxd.iter_number
@profilehtml run!(mxd2, 2000)


# println(mxd.potential.ngev)
# ProfileSVG.save("profile.svg")
