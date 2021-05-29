"""
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


abstract type AbstractPotential{dim} end


mutable struct InversePower{dim} <: AbstractPotential{dim}
    power::Real
    ϵ::Real
    dim::Real
    InversePower(dim, power, ϵ) = new(dim, power, ϵ)
end




function pairwise_energy!(potential::InversePower, r2::Real, radius_sum::Real)
    if (r2 >= radius_sum^2)
        return 0
    else
        r::Real = sqrt(r2)
        return (potential.\epsilon/potential.power)*((1-r/radius_sum)^potential.power)
    end
end


