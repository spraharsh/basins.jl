@doc raw"""
Finds the minimum/saddle corresponding to the point of the gradient system
$$
\frac{dx}{dt} = - \grad{V(x)}
$$
"""

using DifferentialEquations
using LinearAlgebra
include("../potentials/inversepower.jl")


struct BasinAssigner
    solver
    reltol
    abstol
    convtol                     # tolerance for convergence to a minimum
end




function gradient_problem_function_cvode!(potential)
    # Helper functions
    negative_grad!(pot_, x_) = -system_gradient!(pot_, x_) #  ODE function
    negative_hess!(pot_, x_) = -system_hessian!(pot_, x_) # jacobian
    function jacobian!(J, u, p, t)
        potential.f_eval +=1
        J = negative_hess!(potential, u)
        nothing
    end
    
    function func!(u, p, t)
        potential.jac_eval+=1
        return negative_grad!(potential, u)
    end

    return ODEFunction(func!, jac = jacobian!)
end


"""
I don't know why the previous version wasn't working with QNDF, and this one doesnt work with CVODE
so I took the quicker route of writing two different but really the same functions instead of
figuring out the root of my problem 
"""
function gradient_problem_function_qndf!(potential)
    # Helper functions
    negative_grad!(pot_, x_) = -system_gradient!(pot_, x_) #  ODE function
    negative_hess!(pot_, x_) = -system_hessian!(pot_, x_) # jacobian
    function jacobian!( u, p, t)
        potential.f_eval +=1
        return negative_hess!(potential, u)
    end
    
    function func!(u, p, t)
        potential.jac_eval+=1
        return negative_grad!(potential, u)
    end

    return ODEFunction(func!, jac = jacobian!)
end



function find_corresponding_minimum(ba::BasinAssigner, func::ODEFunction, initial_point, maxsteps)    
    convergence_check(g_) = norm(g_) < ba.convtol

    tspan = (0, 100000.)
    
    
    prob = ODEProblem(func, initial_point, tspan)
    integrator = init(prob, ba.solver, reltol=ba.reltol, abstol=ba.abstol)
    
    converged = false
    step_number = 0
    while (!converged && step_number<=maxsteps)
	    step!(integrator)
        step_number += 1
        converged = convergence_check(get_du(integrator))
    end
    println(integrator.sol.destats)
    nf = integrator.sol.destats.nf
    nsolve = step_number
    nw = integrator.sol.destats.nw
    success = converged
    return (integrator.u, nw, nf, nsolve, nw)
end




