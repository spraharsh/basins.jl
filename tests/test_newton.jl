@doc raw"""
    Tests for the newton solver in julia
"""

include("../optimizer/newton.jl")

using Test
using LinearAlgebra
using IterativeSolvers





harmonic(x) = sum(x .^ 2)

# r_g!(G, x) = 2*x

function harmonic_g!(G, x)
    G .= 2 * x
end

function harmonic_h!!(H, x)
    H .= diagm(2 * ones(length(x)))
end


"""
svd maybe slower but apparently gives the minimum norm solution
"""
function lsolve_svd!(x, A, b)
    # cholesky!(A)
    svd!(A)
    # calculate nullspace of A
    x .= A \ b
end



# Todo remember to provide a good starting guess
function lsolve_lsmr!(x, A, b)
    IterativeSolvers.lsmr!(x, A, b)
end


# Line search check over backtracking
energy_func(x) = sum(x .^ 2)




step = [2.0, 0.0]
backtracking_line_search!(step, [-1.0, 0.0], energy_func, 1, 10)

@test all(step .== [0.0 0.0])
@test all(step .== [0.0 0.0])


nls = NewtonLinesearch(
    lsolve_svd!,
    harmonic,
    harmonic_g!,
    harmonic_h!!,
    backtracking_line_search!,
    [1.0, 0.0],
)

# harmonic minimize fast
minimize!(nls,1)
@test nls.nsteps == 1




# utils

rosenbrock(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function r_g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end



function r_h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

nls = NewtonLinesearch(
    lsolve_svd!,
    rosenbrock,
    r_g!,
    r_h!,
    backtracking_line_search!,
    [0.0, 0.0],
)
minimize!(nls)
println(nls.x0)
println(nls.nsteps)
