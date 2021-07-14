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
using Krylov

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
    # data collection
    n_e_evals  # energy evaluations
    n_g_evals  # gradient evaluations
    n_h_evals  # hessian evaluations
end



function Mixed_Descent(pot::AbstractPotential, ode_solver, optimizer::AbstractOptimizer, coords,  T, ode_tol, lambda_tol, conv_tol)
    odefunc_pele = gradient_problem_function_pele!(pot)
    tspan = (0, 100.)
    prob = ODEProblem{true}(odefunc_pele, coords, tspan)
    ode_solver = CVODE_BDF()
    integrator = init(prob, ode_solver, reltol=ode_tol, abstol=ode_tol)
    converged = false
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
    Mixed_Descent(integrator, optimizer, pot, converged, iter_number, conv_tol, lambda_tol, T, N, switch_to_phase_2, use_phase_1, hessian_calculated, hess, coords_in, 0, 0, 0)
end




function one_iteration!(mxd::Mixed_Descent)
    # Convexity check during cvode phase
    
    if (mxd.iter_number % mxd.T == 0 & (mxd.iter_number>0) & !(mxd.switch_to_phase_2))
        hess = system_hessian_pele(mxd.potential, mxd.integrator.u)
        hess_eigvals = eigvals(hess)
        min_eigval = minimum(hess_eigvals)
        max_eigval = maximum(hess_eigvals)
        if (max_eigval==0)
            max_eigval = 10^(-8)
        end
        convexity_estimate = abs(min_eigval/max_eigval)
        if ((min_eigval< -mxd.lambda_tol) & (convexity_estimate >= mxd.conv_tol))
            mxd.switch_to_phase_2 = false
        else
            mxd.switch_to_phase_2 = true
            mxd.optimizer.x0 .= mxd.integrator.u
            print()
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
        converged = one_iteration!(mxd)
        if converged
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
function lsolve_lsmr!(x, A, b) 
    min_eigval = minimum(eigvals(A))
    @debug "min_eigval:" min_eigval
    if min_eigval<0
        A[diagind(A)] .-=  - 10^-3 + 3*min_eigval
    end
    
    # print(eigvals(A))
    # IterativeSolvers.lsmr!(x, A, b)
    # print("x out of solver:")
    # println(x)
    # x
    # Krylov.minres_qlp!(x, A, b)    
    x .= svd!(A) \ b
end

function get_egh_funcs(potential)
    e_func(x) = system_energy_pele(potential, x)
    e_grad(G, x) = system_gradient_pele!(potential, x, G)

    function e_hess(H, x)
        H .= system_hessian_pele(potential, x)
    end

    (e_func, e_grad, e_hess)
end


function NewtonLinesearch(pot, coords)
    ef, gf, hf = get_egh_funcs(pot)
    NewtonLinesearch(
       lsolve_lsmr!,
        ef,
        gf,
        hf,
        backtracking_line_search!,
        coords
    )
end

