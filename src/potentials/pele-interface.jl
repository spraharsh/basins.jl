@doc raw"""
Interface for getting potentials from pele
"""

using PyCall
include("base_potential.jl")
using SparseArrays

pot = pyimport("pele.potentials")

"""
calls C++ potentials through the python wrapper and PyCall

I know it's complicated, but it's super easy compared to directly calling C++?

only issue is that the jacobians have to be directly created every time.
goal(write )
"""
mutable struct PythonPotential <: AbstractPotential
    pele_potential::PyObject
    ngev::Int
    nhev::Int
    neev::Int
end


PythonPotential(pot) = PythonPotential(pot, 0, 0, 0)


function system_energy_pele(pot::PythonPotential, x)
    pot.neev += 1
    return pot.pele_potential.getEnergy(x)
end

function system_gradient_pele!(pot::PythonPotential, x, grad)
    pot.ngev += 1
    pot.pele_potential.getEnergyGradientInPlace(x, grad)
end

function system_gradient_pele(pot::PythonPotential, x)
    return pot.pele_potential.getEnergyGradient(x)[2]
end

function system_hessian_pele(pot::PythonPotential, x)
    pot.nhev += 1
    return pot.pele_potential.getEnergyGradientHessian(x)[3]
end


# pass dummy into hessian 
function system_hessian_pele!(
    pot::PythonPotential,
    x::Vector{Float64},
    hess::Matrix{Float64},
)
    pot.nhev += 1
    # since pele evaluates a flat hessian we need to flatten our hessian first
    hess_flat = reshape(hess, (length(x) * length(x), 1))
    hess_flat = dropdims(hess_flat, dims = 2)
    pot.pele_potential.getHessianInPlace(x, hess_flat)
end

# pass dummy into hessian 
function system_grad_hessian_pele!(
    pot::PythonPotential,
    x::Vector{Float64},
    grad::Vector{Float64},
    hess::Matrix{Float64},
)
    pot.nhev += 1
    # since pele evaluates a flat hessian we need to flatten our hessian first
    hess_flat = reshape(hess, (length(grad) * length(grad), 1))
    hess_flat = dropdims(hess_flat, dims = 2)
    pot.pele_potential.getEnergyGradientHessianInPlace(x, grad, hess_flat)
end



function gradient_problem_function_pele!(potential; kwargs...)
    function func!(du, u, p, t)
        system_gradient_pele!(potential, u, du)
        du .= -du
        nothing
    end
    return ODEFunction(func!; kwargs...)
end



function gradient_problem_function_with_hessian_pele!(potential; kwargs...)
    function func!(du, u, p, t)
        system_gradient_pele!(potential, u, du)
        du .= -du
        nothing
    end
    function jac!(J, u, p, t)
        system_hessian_pele!(potential, u, J)
        J .= -J
    end
    return ODEFunction(func!, jac = jac!; kwargs...)
end
