@doc raw"""
    Mixed descent solver as defined in pele
"""


using DifferentialEquations
using LinearAlgebra
using Sundials
include("../optimizer/optimizer.jl")
include("../potentials/base_potential.jl")
include("../potentials/pele-interface.jl")
include("../utils/utils.jl")

"""
Mixed descent to assign minimum as fast as possible
"""
mutable struct Mixed_Descent
    integrator
    optimizer
    potential
    converged
    iter_number
    conv_tol
    lambda_tol
    T
    N
    switch_to_phase_2::Bool
    use_phase_1::Bool
    hessian_calculated::Bool
    hess::Any
    coords::Vector{Float64}
end



function Mixed_Descent(pot::AbstractPotential, ode_solver, optimizer::AbstractOptimizer, coords,  T, ode_tol, lambda_tol, conv_tol)
    odefunc_pele = gradient_problem_function_pele!(pot)
    tspan = (0, 100.)
    prob = ODEProblem{true}(odefunc_pele, coords, tspan)
    ode_solver = CVODE_BDF()
    println(ode_tol)
    println(prob)
    integrator = init(prob, ode_solver, reltol = ode_tol, abstol=ode_tol)
    converged = false
    iter_number = 0
    switch_to_phase_2 = false
    use_phase_1 = true
    hessian_calculated = false
    N = length(coords)
    hess = (zeros((length(coords), length(coords))))
    coords_in = copy(coords)
    Mixed_Descent(integrator, optimizer, pot, converged, iter_number, conv_tol, lambda_tol, T, N, switch_to_phase_2, use_phase_1, hessian_calculated, hess, coords_in)
end




function one_iteration!(mxd::Mixed_Descent)
    # Convexity check during cvode phase
    println(mxd.iter_number%mxd.T == 0)
    println(mxd.switch_to_phase_2)
    
    if (mxd.iter_number % mxd.T == 0 & (mxd.iter_number>0) & !(mxd.switch_to_phase_2))
        hess = system_hessian_pele(mxd.potential, mxd.integrator.u)
        hess_eigvals = eigvals(hess)
        min_eigval = minimum(hess_eigvals)
        max_eigval = maximum(hess_eigvals)
        if (max_eigval==0)
            max_eigval = 10^(-8)
        end
        print("min_eigval: ")
        println(min_eigval)
        convexity_estimate = abs(min_eigval/max_eigval)
        if ((min_eigval< -10^-8) & (convexity_estimate >= mxd.conv_tol))
            mxd.switch_to_phase_2 = false
        elseif ((min_eigval < -10^-8) & (convexity_estimate<mxd.conv_tol))
            mxd.switch_to_phase_2 =false
        else
            mxd.switch_to_phase_2 = true
            mxd.optimizer.x0 .= mxd.integrator.u
        end   
    end
    converged = false
    # solve the ODE during phase 1
    if !(mxd.switch_to_phase_2)
        step!(mxd.integrator)
        mxd.iter_number +=1
    else
        minimize!(mxd.optimizer)
        converged = true
        mxd.iter_number += mxd.optimizer.nsteps
    end
    converged
end

function run!(mxd::Mixed_Descent, max_steps::Int = 10000)
    for i = 1:max_steps
        println(i)
        converged = one_iteration!(mxd)
        if converged
            break
        end
    end
    (mxd.optimizer.x0)
end



natoms = 64
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
println(cell_scale)

pele_wrapped_pot = pot.InversePower(
    2.5,
    1.0,
    radii_arr,
    ndim = 2,
    boxvec = boxvec,
    use_cell_lists = false,
    ncellx_scale = cell_scale,
)

pele_wrapped_python_pot = PythonPotential(pele_wrapped_pot)



include("../optimizer/newton.jl")
using IterativeSolvers


# Todo remember to provide a good starting guess
function lsolve_lsmr!(x, A, b) 
   println("this")
    print("eigenvalues original")
    min_eigval = minimum(eigvals(A))
    println(eigvals(A))
    if min_eigval<0
        print("minimum eigenvalue")
        println(min_eigval)
        A[diagind(A)] .-=  - 2*min_eigval
    end
    print("eigenvalues:")
    println("-------- Eigenvalues ")
    # print(eigvals(A))
    # IterativeSolvers.lsmr!(x, A, b)
    # print("x out of solver:")
    # println(x)
    # x
    x .= qr!(A, Val(true)) \ b
    println(x)
end


function p_energy(x)
    system_energy_pele(pele_wrapped_python_pot, x)
end

# r_g!(G, x) = 2*x

function p_gradient(G, x)
    system_gradient_pele!(pele_wrapped_python_pot, x, G)
end

function p_hessian(H, x)
    H .= system_hessian_pele(pele_wrapped_python_pot, x)
end



nls = NewtonLinesearch(
    lsolve_lsmr!,
    p_energy,
    p_gradient,
    p_hessian,
    backtracking_line_search!,
    coords
)

using Sundials
solver = CVODE_BDF()

println(pele_wrapped_pot)
println(coords)
println(boxvec)
println(radii_arr)
mxd = Mixed_Descent(pele_wrapped_python_pot, solver, nls, coords,  30, 10^-5, 0.0 , 10^-3)
println(coords)






run!(mxd, 2000)

println(coords)
println(mxd.integrator.u .- coords)
println("hellow world")
println(p_energy(coords))




