@doc raw"""
Interface for getting potentials from pele
"""

using PyCall
include("base_potential.jl")
using SparseArrays

pot = pyimport("pele.potentials")

th = pot.InversePower(2.5, 1.0, [1, 1, 1], ndim = 2)

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
    pot.ngev +=1
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
function system_grad_hessian_pele!(pot::PythonPotential, x, grad, hess)
	pot.nhev += 1
    return pot.pele_potential.getEnergyGradientHessianInPlace(x, grad, hess)
end



function gradient_problem_function_pele!(potential)
    function func!(du, u, p, t)
        system_gradient_pele!(potential, u, du)
        du .= -du
        nothing
    end
    return ODEFunction(func!)
end



# ippot = pot.InversePower(2.5, 1.0, [1.0, 1.0], ndim = 2)
# wrapped_pot = PythonPotential(ippot)
# x = [1.0, 0.0, 2.0, 0]
# system_energy_pele(wrapped_pot, x)
# system_gradient_pele(wrapped_pot, x)
# println(sparse(system_hessian_pele(wrapped_pot, x)))

# grad = ones(length(x))
# hess = ones(length(x)^2)

# system_grad_hessian_pele!(wrapped_pot, x, grad, hess)
# sparse(reshape(hess, (4,4)))
# println(hess)

