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




function gradient_problem_function(potential)
    # Helper functions
    negative_grad(pot_, x_) = -system_gradient(pot_, x_) #  ODE function
    negative_hess(pot_, x_) = -system_hessian(pot_, x_) # jacobian
    func(u, p, t) = negative_grad(potential, u)
    jacobian(u, p, t) = negative_hess(potential, u)
    return ODEFunction(func, jac = jacobian)
end


function find_corresponding_minimum(ba::BasinAssigner, func::ODEFunction, initial_point, maxsteps)    
    convergence_check(g_) = norm(g_) < ba.convtol

    tspan = (0, 100000.)
    
    
    prob = ODEProblem(func, initial_point, tspan)
    integrator = init(prob, ba.solver, reltol=ba.reltol, abstol=ba.abstol)
    
    converged = false
    step_number = 1
    while (!converged && step_number<=maxsteps)
	    step!(integrator)
        step_number += 1
        converged = convergence_check(get_du(integrator))
    end
    nf = integrator.destats.nf
    nsolve = integrator.destats.nsolve
    nw = integrator.destats.nw
    success = converged
    return (integrator.u, nw, nf, nsolve, nw)
end


