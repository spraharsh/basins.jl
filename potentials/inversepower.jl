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


struct InversePowerPeriodic <: AbstractPotential
    dim::Integer
    power::Real
    epsilon::Real
    # TODO: auto calculate from dim and box length to prevent DimensionMismatch
    box_vec
    radii
    InversePowerPeriodic(dim, power, epsilon,box_vec, radii) = new(dim, power, epsilon,box_vec, radii)
end




function pairwise_energy(potential::InversePowerPeriodic, r2::Real, radius_sum::Real)
    if (r2 >= radius_sum^2)
        return 0
    else
        r::Real = sqrt(r2)
        return (potential.epsilon/potential.power)*((1-r/radius_sum)^potential.power)
    end
end


function system_energy(potential, x)
    natoms::Integer = size(x, 1)/potential.dim
    if natoms*potential.dim != size(x, 1)
        throw(DimensionMismatch(x, "coordinates have the wrong dimensions"))
    end

    if natoms != size(potential.radii, 1)
        throw(DimensionMismatch(x, "mismatch between dimensions of radii and number of atoms"))
    end

    x_ = reshape(x, (potential.dim, natoms))
    energy::Real = 0
    for atomi in 1:natoms
        for atomj in (atomi+1):natoms
            r2 = peuclidean(x_[:, atomi], x_[:, atomj], potential.box_vec)^2
            energy += pairwise_energy(potential,r2,  potential.radii[atomi]+ potential.radii[atomj])
        end
    end
    return energy
end

function system_gradient(potential, x)
    f(x_) = system_energy(potential, x_)
    return ForwardDiff.gradient(f, x)
end

function system_hessian(potential, x)
    f(x_) = system_energy(potential, x_)
    return ForwardDiff.hessian(f, x)
end



if abspath(PROGRAM_FILE) == @__FILE__
    # Periodicity check
    radii = [1, 1]
    pot = InversePowerPeriodic(2., 2.5, 1.,3, radii)
    x = [1., 2., 3., 4.]
    println(system_energy(pot, x))
    println(system_gradient(pot, x))
    println(system_hessian(pot, x))
end