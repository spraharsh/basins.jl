@doc raw"""
Benchmarks are done in ipython notebook since 
since I'm not particularly sure how to set propagated OpenMP threads from julia, through python to C++
within a cell
"""

using DiffEqBase,
    OrdinaryDiffEq,
    Sundials,
    Plots,
    ODEInterface,
    ODEInterfaceDiffEq,
    LSODA,
    ModelingToolkit,
    DiffEqDevTools,
    SparseArrays
using PyCall
using Plots
using LinearSolve
include("src/potentials/inversepower.jl")
include("src/minimumassign/minimumassign.jl")
include("src/utils/utils.jl")
include("src/potentials/pele-interface.jl")

natoms = 1024
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
dim = 2
phi = 0.9
power = 2.5
eps = 1

length_arr = get_box_length(radii_arr, phi, dim)

coords = generate_random_coordinates(length_arr, natoms, dim)

# calculate a hessian




tol = 1e-4
tspan = (0, 100000.0)

print("using OMP NUM THREADS: ")
println(ENV["OMP_NUM_THREADS"])

boxvec = [length_arr, length_arr]

utils = pyimport("pele.utils.cell_scale")
cell_scale = utils.get_ncellsx_scale(radii_arr, boxvec)
println(cell_scale)


pele_wrapped_pot = pot.InversePower(
    2.5,
    1.0,
    radii_arr,
    ndim = 2,
    boxvec = boxvec,
    use_cell_lists = true,
    ncellx_scale = cell_scale,
)

pele_wrapped_pot.getEnergy(coords)


hess = zeros(length(coords), length(coords))
pele_wrapped_python_pot = PythonPotential(pele_wrapped_pot)
system_hessian_pele!(pele_wrapped_python_pot, coords, hess)
sparse_hess = sparse(hess)

odefunc_pele = gradient_problem_function_pele!(pele_wrapped_python_pot)
println(sparse_hess)

prob = ODEProblem{true}(odefunc_pele, coords, tspan, jac_prototype=sparse_hess);

@time sol =
    solve(prob, QNDF(linsolve=KLUFactorization(reuse_symbolic=false), autodiff=false), abstol = 1 / 10^12, reltol = 1 / 10^12);
sol.destats




