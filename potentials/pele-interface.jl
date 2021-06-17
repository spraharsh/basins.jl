@doc raw"""
Interface for getting potentials from pele
"""

using PyCall
include("base_potential.jl")
using SparseArrays

pot = pyimport("pele.potentials")

th = pot.InversePower(2.5, 1.0, [1,1,1], ndim=2)

"""
calls C++ potentials through the python wrapper and PyCall

I know it's complicated, but it's super easy compared to directly calling C++?

only issue is that the jacobians have to be directly created every time.
goal(write )
"""
struct PythonPotential <: AbstractPotential
    pele_potential::PyObject
    PythonPotential(pot) = new(pot)
end



function system_energy_pele(pot::PythonPotential, x)
    return pot.pele_potential.getEnergy(x)
end

function system_gradient_pele!(pot::PythonPotential, x, grad)
    return pot.pele_potential.getEnergyGradientInPlace(x, grad)
end

function system_gradient_pele(pot::PythonPotential, x)
    return pot.pele_potential.getEnergyGradient(x)[2]
end

function system_hessian_pele(pot::PythonPotential, x)
    return pot.pele_potential.getEnergyGradientHessian(x)[3]
end


function gradient_problem_function_pele!(potential)
    function func!(du, u, p, t)
        -system_gradient_pele!(potential, u, du)
        du .= -du
        nothing
    end
    return ODEFunction(func!)
end

if abspath(PROGRAM_FILE) == @__FILE__
    ippot = pot.InversePower(2.5, 1.0, [1., 1.], ndim =2)
    wrapped_pot = PythonPotential(ippot)
    x = [1.,0., 2., 0]
    system_energy_pele(wrapped_pot,x )
    system_gradient_pele(wrapped_pot, x)
    sparse(system_hessian_pele(wrapped_pot, x))
end

