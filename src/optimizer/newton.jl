

using LinearAlgebra
include("optimizer.jl")

"""
TODO: give types to everything
"""
mutable struct NewtonLinesearch <: AbstractOptimizer
    linear_solver!
    energy_func::Any
    grad_func!::Any
    hess_func!::Any
    line_search!::Any
    nfeval::Int
    nheval::Int
    nsteps::Int
    convtol::Float64
    x0::AbstractVector{Float64}
    xold::AbstractVector{Float64}
    grad::AbstractVector{Float64}
    hess::Matrix{Float64}
    converged::Bool
    energyold::Float64
    ls_maxsteps::Int
end
"""
Newton method with a line search
"""



function NewtonLinesearch(
    linear_solver!,
    energy_func,
    grad_func!,
    hess_func!,
    line_search!,
    x0,
    convtol = 10^-8,
    ls_maxsteps = 30,
)
    N = length(x0)
    println("NewtonLinesearch:")
    println(N)
    nfeval = 0
    nheval = 0
    nsteps = 0
    convtol = convtol
    xold = zeros(N)
    grad = zeros(N)
    print(N)
    x0 = copy(x0)
    hess = (zeros((N, N)))
    converged = false
    energyold = 0.0 # energy at previous iteration
    ls_maxsteps = ls_maxsteps
    NewtonLinesearch(
        linear_solver!,
        energy_func,
        grad_func!,
        hess_func!,
        line_search!,
        nfeval,
        nheval,
        nsteps,
        convtol,
        x0,
        xold,
        grad,
        hess,
        converged,
        energyold,
        ls_maxsteps,
    )
end

"""
Fix function evaluations
"""
@inline function one_iteration!(NLS::NewtonLinesearch)
    # copy the old array into the new one

    
    NLS.xold .= NLS.x0

    # NLS.grad_func!(NLS.grad, NLS.x0)
    NLS.hess_func!(NLS.hess, NLS.grad, NLS.x0)  # hessian

    @debug NLS.grad
    if (norm(NLS.grad)/sqrt(length(NLS.grad)) < NLS.convtol)
        return true
    end
    
    
    # here since the step basically starts
    # after evaluating the gradient at the
    # previous point
    NLS.nsteps += 1

    # NLS.hess_func!(NLS.hess, NLS.x0)
    NLS.nheval += 1
    NLS.linear_solver!(NLS.x0, NLS.hess, NLS.grad)
    NLS.x0 .= -NLS.x0
    grad_dot_prod = - dot(NLS.x0, NLS.grad)
    @debug "grad_dot_prod = ", grad_dot_prod
    if grad_dot_prod < 0
        throw(
            DomainError(0, "Error: step not in direction of negative gradient"),
        )
    end
    # update xnew = newton step + xold with line search and store energy at point
    NLS.energyold =
        NLS.line_search!(NLS.x0, NLS.xold, NLS.energy_func, NLS.energyold, NLS.ls_maxsteps)
    return false
end

"""
Does not calculate the step if the hessian already exists. 
    TODO: fold the calculated hessian version into this one
"""
@inline function one_iteration!(NLS::NewtonLinesearch, hess::Matrix{Float64})
    # copy the old array into the new one
    NLS.xold .= NLS.x0

    NLS.grad_func!(NLS.grad, NLS.x0)

    # @info NLS.grad
    if (norm(NLS.grad)/sqrt(length(NLS.grad)) < NLS.convtol)
        return true
    end
    
    # here since the step basically starts
    # after evaluating the gradient at the
    # previous point
    NLS.nsteps += 1
    NLS.linear_solver!(NLS.x0, hess, NLS.grad)
    NLS.x0 .= -NLS.x0
    grad_dot_prod = - dot(NLS.x0, NLS.grad)
    @debug "grad_dot_prod = ", grad_dot_prod
    if grad_dot_prod < 0
        throw(
            DomainError(0, "Error: step not in direction of negative gradient"),
        )
    end
    # update xnew = newton step + xold with line search and store energy at point
    NLS.energyold =
        NLS.line_search!(NLS.x0, NLS.xold, NLS.energy_func, NLS.energyold, NLS.ls_maxsteps)
    return false
end




function minimize!(NLS::NewtonLinesearch, max_steps::Int = 10000)
    for i = 1:max_steps
        converged = one_iteration!(NLS)
        if converged
            break
        end
    end
    (NLS.x0)
end


"""
simple backtracking line search. just ensures energy decreases (but can implement
sufficient decrease if need be)
"""
@inline function backtracking_line_search!(step::Vector{Float64}, x0::Vector{Float64}, energy_func, eold::Float64, maxsteps::Int = 100)
    if norm(step) == 0
        throw(error("cant have a zero value step"))
    end
    step .+= x0
    enew = 0
    nsteps = 0
    eold = energy_func(x0)
    for i = 1:maxsteps
        # first compute the new position
        nsteps += 1
        enew = energy_func(step)
        # calculate energy
        # halve the step
        if (enew >= eold)
            # decrease step by half
            step .-= (step .- x0) / 2
            enew = energy_func(step)
        else
            return enew
        end
        if (i == maxsteps)
            throw(
                DomainError(0, "line search did not converge, too few steps"),
            )
        end
    end
end
