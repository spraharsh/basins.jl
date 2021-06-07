@doc raw"""
  P.S. This probably can be done with the MOLLY package
    
  Pairwise interaction for Inverse power potential eps/pow * (1 - r/r0)^pow
  the radii array allows for polydispersity
  The most common exponents are:
  pow=2   -> Hookean interaction
  pow=2.5 -> Hertzian interaction

  Comments about this implementation:
         (documentation as copied from pele)
  This implementation is using STL pow for all exponents. Performance wise this may not be
  the fastest possible implementation, for instance using exp(pow*log) could be faster
  (twice as fast according to some blogs).

  Maybe we should consider a function that calls exp(pow*log) though, this should be carefully benchmarked
  though as my guess is that the improvement is going to be marginal and will depend on the
  architecture (how well pow, exp and log can be optimized on a given architecture).

  See below for meta pow implementations for integer and half-integer exponents.

  If you have any experience with pow please suggest any better solution and/or provide a
  faster implementation.
"""

using Distances
using ForwardDiff

abstract type AbstractPotential end


mutable struct InversePowerPeriodic <: AbstractPotential
    dim::Integer
    power::Real
    epsilon::Real
    # TODO: auto calculate from dim and box length to prevent DimensionMismatch
    box_vec::Any
    radii::Any
    # wrote these to ensure that the function evaluations are counted accurately
    # ideally they shouldn't be in this struct
    f_eval::Any
    jac_eval::Any
    InversePowerPeriodic(dim, power, epsilon, box_vec, radii, f_eval, jac_eval) =
        new(dim, power, epsilon, box_vec, radii, f_eval, jac_eval)
end

InversePowerPeriodic(dim, power, epsilon, box_vec, radii) =
    InversePowerPeriodic(dim, power, epsilon, box_vec, radii, 0, 0)


function pairwise_energy(potential::InversePowerPeriodic, r2::Real, radius_sum::Real)
    if (r2 >= radius_sum^2)
        return 0
    else
        r::Real = sqrt(r2)
        return (potential.epsilon / potential.power) *
               ((1 - r / radius_sum)^potential.power)
    end
end


@inline function unravel_index(i, j, ndim)
    return i * 2 - 1 + j * 2 - 1
end

function system_energy(potential, x)
    dim = potential.dim
    natoms::Integer = size(x, 1) / dim

    if natoms * dim != size(x, 1)
        throw(DimensionMismatch(x, "coordinates have the wrong dimensions"))
    end

    if natoms != size(potential.radii, 1)
        throw(
            DimensionMismatch(
                x,
                "mismatch between dimensions of radii and number of atoms",
            ),
        )
    end

    x_ = reshape(x, (potential.dim, natoms))

    energy::Real = 0
    for i = 1:natoms
        for j = (i+1):natoms
            @inbounds r2 = peuclidean(x_[:, i], x_[:, j], potential.box_vec)^2
            @inbounds energy +=
                pairwise_energy(potential, r2, (potential.radii[i] + potential.radii[j]))
        end
    end
    return energy
end


function system_gradient!(g, potential, x)
    f(x_) = system_energy(potential, x_)
    ForwardDiff.gradient!(g, f, x)
    nothing
end

function system_negative_gradient!(g, potential, x)
    f(x_) = system_energy(potential, x_)
    ForwardDiff.gradient!(g, f, x)
    g .= -g
    nothing
end


function system_hessian!(potential, x)
    f(x_) = system_energy(potential, x_)
    return ForwardDiff.hessian(f, x)
end



if abspath(PROGRAM_FILE) == @__FILE__
    # Periodicity check
    radii = [1, 1]
    pot = InversePowerPeriodic(2.0, 2.5, 1.0, 3, radii)
    x = [1.0, 2.0, 3.0, 4.0]
    println(system_energy(pot, x))
    println(system_gradient(pot, x))
    println(system_hessian(pot, x))
end
