@doc raw"""
    Mixed descent solver as defined in pele
"""


using DifferentialEquations
using LinearAlgebra
using Sundials
using DiffEqBase
include("../optimizer/optimizer.jl")
include("../potentials/base_potential.jl")
include("../potentials/pele-interface.jl")
include("../utils/utils.jl")
using Krylov
using IterativeSolvers

"""
Mixed descent to assign minimum as fast as possible
"""
mutable struct Mixed_Descent
    integrator::Sundials.CVODEIntegrator
    optimizer::AbstractOptimizer
    potential::AbstractPotential
    converged::Bool
    iter_number::Int
    conv_tol::Float64
    lambda_tol::Float64
    T::Int
    N::Int
    switch_to_phase_2::Bool
    use_phase_1::Bool
    hess::Matrix{Float64}
    dummy_grad::Vector{Float64} # Used as en extra for the hessian  
    hess_current::Bool  # Checks whether hessian is current
    coords::Vector{Float64}
    # data collection
    n_e_evals::Int  # energy evaluations
    n_g_evals::Int  # gradient evaluations
    n_h_evals::Int  # hessian evaluations
end



function Mixed_Descent(
    pot::AbstractPotential,
    ode_solver::CVODE_BDF,
    optimizer::AbstractOptimizer,
    coords::Vector{Float64},
    T::Int,
    ode_tol::Float64,
    lambda_tol::Float64,
    conv_tol::Float64,
)
    odefunc_pele = gradient_problem_function_with_hessian_pele!(pot)
    tspan = (0, 100.0)
    prob = ODEProblem{true}(odefunc_pele, coords, tspan)
    ode_solver = CVODE_BDF(linear_solver = :Dense)
    println(ode_tol)
    integrator = init(prob, ode_solver, reltol = ode_tol, abstol = ode_tol)
    converged = false
    # @info lambda_tol
    if lambda_tol == 0
        @warn "convexity tolerance is 0, leading to numerical convergence misses. setting to 1e-8"
        lambda_tol = 1e-8
    end
    iter_number = 0
    switch_to_phase_2 = false
    use_phase_1 = true
    hessian_calculated = false
    N = length(coords)
    hess = (zeros((length(coords), length(coords))))
    coords_in = copy(coords)
    Mixed_Descent(
        integrator,
        optimizer,
        pot,
        converged,
        iter_number,
        conv_tol,
        lambda_tol,
        T,
        N,
        switch_to_phase_2,
        use_phase_1,
        hess,
        copy(coords), # dummy gradient
        false,  # hessian not current
        coords_in,
        0,
        0,
        0,
    )
end




function one_iteration!(mxd::Mixed_Descent)
    mxd.hess_current = false # reset
    if (mxd.iter_number % mxd.T == 0 & (mxd.iter_number > 0) & !(mxd.switch_to_phase_2))
        system_grad_hessian_pele!(mxd.potential, mxd.integrator.u, mxd.dummy_grad, mxd.hess)
        mxd.hess_current = true
        hess_eigvals = eigvals(mxd.hess)
        min_eigval = minimum(hess_eigvals)
        max_eigval = maximum(hess_eigvals)
        if (max_eigval == 0)
            max_eigval = 10^(-8)
        end
        convexity_estimate = abs(min_eigval / max_eigval)
        if ((min_eigval < -mxd.lambda_tol) & (convexity_estimate >= mxd.conv_tol))
            mxd.switch_to_phase_2 = false
            mxd.integrator.u = mxd.optimizer.x0
        else
            mxd.switch_to_phase_2 = true
            mxd.optimizer.x0 .= mxd.integrator.u
        end
    end
    converged = false
    # solve the ODE during phase 1
    if !(mxd.switch_to_phase_2)
        step!(mxd.integrator)
        mxd.iter_number += 1
    else
        # minimize!(mxd.optimizer)
        if mxd.hess_current
            converged = one_iteration!(mxd.optimizer, mxd.hess)
        else
            converged = one_iteration!(mxd.optimizer)
        end
        mxd.iter_number += 1
    end
    converged
end

function run!(mxd::Mixed_Descent, max_steps::Int = 10000)
    for i = 1:max_steps
        converged::Bool = one_iteration!(mxd)
        if converged
            mxd.converged = true
            break
        end
    end
    # stat collection from the potential
    mxd.n_e_evals = mxd.potential.neev
    mxd.n_g_evals = mxd.potential.ngev
    mxd.n_h_evals = mxd.potential.nhev
    (mxd.optimizer.x0)
end



# Todo remember to provide a good starting guess
function lsolve_lsmr!(x::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64})
    min_eigval = minimum(eigvals(A))
    @debug "min_eigval:" min_eigval
    if min_eigval < 0
        A[diagind(A)] .-= 2 * min_eigval
    end
    x .= svd!(A) \ b
    # IterativeSolvers.lsqr!(x, A, b, damp=1e-3, conlim=1e-11)00
    # lsmr!(x, A, b)
end

function get_egh_funcs(potential)
    e_func(x) = system_energy_pele(potential, x)
    e_grad(G, x) = system_gradient_pele!(potential, x, G)

    function e_hess(H, g,  x)
        system_grad_hessian_pele!(potential, x, g, H)
    end
    (e_func, e_grad, e_hess)
end


function NewtonLinesearch(pot, coords, conv_tol, ls_max_steps=20)
    ef, gf, hf = get_egh_funcs(pot)
    NewtonLinesearch(lsolve_lsmr!, ef, gf, hf, backtracking_line_search!, coords, conv_tol, ls_max_steps)
end
