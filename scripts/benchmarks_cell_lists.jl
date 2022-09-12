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
include("../src/potentials/inversepower.jl")
include("../src/minimumassign/minimumassign.jl")
include("../src/utils/utils.jl")
include("../src/potentials/pele-interface.jl")

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
tspan = (0, 1000.0)

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

odefunc_pele = get_ode_func_gradient_problem_pele!(pele_wrapped_python_pot;jac_prototype = sparse_hess)

prob = ODEProblem{true}(odefunc_pele, coords, tspan);

# define preconditioners for solving

using AlgebraicMultigrid
function algebraicmultigrid(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
  if newW === nothing || newW
    Pl = aspreconditioner(ruge_stuben(convert(AbstractMatrix,W)))
  else
    Pl = Plprev
  end
  Pl,nothing
end

function algebraicmultigrid_with_smoothing(W,du,u,p,t,newW,Plprev,Prprev,solverdata)
    if newW === nothing || newW
      A = convert(AbstractMatrix,W)
      Pl = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(A, presmoother = AlgebraicMultigrid.Jacobi(rand(size(A,1))), postsmoother = AlgebraicMultigrid.Jacobi(rand(size(A,1)))))
    else
      Pl = Plprev
    end
    Pl,nothing
end

BLAS.set_num_threads(1);        # for julia blas only
abstols = 1.0 ./ 10.0 .^ (5:11);
reltols = 1.0 ./ 10.0 .^ (5:11);
setups = [
    # Dict(:alg => QNDF(autodiff = false)),
    Dict(:alg => QNDF(linsolve=KrylovJL_GMRES(), autodiff = false)),
    Dict(:alg => QNDF(linsolve=KrylovJL_GMRES(), precs=algebraicmultigrid,concrete_jac=true, autodiff=false)),
    Dict(:alg => QNDF(linsolve=KrylovJL_GMRES(), precs=algebraicmultigrid_with_smoothing,concrete_jac=true, autodiff=false)),
    Dict(:alg => FBDF(linsolve=KrylovJL_GMRES(), autodiff=false)),
    Dict(:alg => FBDF(linsolve=KrylovJL_GMRES(), precs=algebraicmultigrid,concrete_jac=true, autodiff=false)),
    Dict(:alg => FBDF(linsolve=KrylovJL_GMRES(), precs=algebraicmultigrid_with_smoothing,concrete_jac=true, autodiff=false)),
    # Dict(:alg=>rodas()),
    Dict(:alg => CVODE_BDF()),
    # Dict(:alg => CVODE_BDF(linear_solver = :KLU)),
    Dict(:alg => CVODE_BDF(linear_solver = :PCG)),
    # Dict(:alg => CVODE_BDF(linear_solver = :BCG)),
    # Dict(:alg => CVODE_BDF(linear_solver = :PCG)),
    # Dict(:alg => CVODE_BDF(linear_solver = :TFQMR)),
    # Dict(:alg=>Rodas4(autodiff=false)),
    # Dict(:alg=>Rodas5(autodiff=false)),
    # Dict(:alg=>RadauIIA5(autodiff=false)),
    # Dict(:alg=>lsoda()),
];

solnames = [
    #"QNDF",
    "QNDF_KrylovJL_GMRES",
    "QNDF_KrylovJL_GMRES_AMG",
    "QNDF_KrylovJL_GMRES_AMG_with_smoothing",
    "FBDF_KrylovJL_GMRES",
    "FBDF_KrylovJL_GMRES_AMG",
    "FBDF_KrylovJL_GMRES_AMG_with_smoothing",
    "CVODE_BDF Dense",
     "CVODE_BDF PCG",
]
println("solving benchmark solution")
@time sol =
    solve(prob, CVODE_BDF(linear_solver = :PCG), abstol = 1 / 10^13, reltol = 1 / 10^13);
sol.destats
println("running benchmarks")
wp2 = WorkPrecisionSet(
    prob,
    abstols,
    reltols,
    setups;
    error_estimate = :l2,
    names = solnames,
    appxsol = sol,
    maxiters = Int(1e5),
    numruns = 2,
);
println("plotted benchmarks")
plot(wp2)

println("done")
savefig("comparison_cvode_cell_lists_OMP_THREADS_particles_n_"*string(natoms)*"_" * string(ENV["OMP_NUM_THREADS"]))