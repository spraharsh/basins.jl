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
    DiffEqDevTools
using PyCall
using Plots
include("potentials/inversepower.jl")
include("minimumassign/minimumassign.jl")
include("utils/utils.jl")

natoms = 1024
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
dim = 2.0
phi = 0.9
power = 2.5
eps = 1

length_arr = get_box_length(radii_arr, phi, dim)

coords = generate_random_coordinates(length_arr, natoms, dim)

tol = 1e-4
tspan = (0, 100.0)

print("using OMP NUM THREADS: ")
println(ENV["OMP_NUM_THREADS"])
using PyCall
include("potentials/pele-interface.jl")
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

pele_wrapped_python_pot = PythonPotential(pele_wrapped_pot)
odefunc_pele = gradient_problem_function_pele!(pele_wrapped_python_pot)

prob = ODEProblem{true}(odefunc_pele, coords, tspan);

BLAS.set_num_threads(1);        # for julia blas only
abstols = 1.0 ./ 10.0 .^ (5:8);
reltols = 1.0 ./ 10.0 .^ (5:8);
setups = [
    Dict(:alg => QNDF(autodiff = false)),
    # Dict(:alg=>rodas()),
    Dict(:alg => CVODE_BDF(linear_solver = :Dense)),
    Dict(:alg => CVODE_BDF(linear_solver = :GMRES)),
    Dict(:alg => CVODE_BDF(linear_solver = :BCG)),
    Dict(:alg => CVODE_BDF(linear_solver = :PCG)),
    Dict(:alg => CVODE_BDF(linear_solver = :TFQMR)),
    # Dict(:alg=>Rodas4(autodiff=false)),
    # Dict(:alg=>Rodas5(autodiff=false)),
    # Dict(:alg=>RadauIIA5(autodiff=false)),
    # Dict(:alg=>lsoda()),
];
solnames = [
    "QNDF",
    "CVODE_BDF Dense",
    "CVODE_BDF GMRES",
    "CVODE_BDF BCG",
    "CVODE_BDF PCG",
    "CVODE_BDF TFQMR",
]

@time sol = solve(prob, CVODE_BDF(), abstol = 1 / 10^12, reltol = 1 / 10^12);
sol.destats


wp2 = WorkPrecisionSet(
    prob,
    abstols,
    reltols,
    setups;
    error_estimate = :l2,
    names = solnames,
    appxsol = sol,
    maxiters = Int(1e5),
    numruns = 1,
);
plot(wp2)

savefig("comparison_cvode_cell_lists_OMP_THREADS_" * string(ENV["OMP_NUM_THREADS"]))
