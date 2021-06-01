@doc raw"""
Creates a system of particles in a box with radii centered around 1, and 1.4, generates a
bunch of random initial points, and finds their corresponding minima
"""

using Random, Distributions
using Sundials
include("potentials/inversepower.jl")
include("minimumassign/minimumassign.jl")
include("utils/utils.jl")




# Generate radii




natoms = 8
radii_arr = generate_radii(0, 8, 1.,1.4, 0.05, 0.05*1.4)
dim = 2.0
phi = 0.9
power = 2.5
eps = 1

length = get_box_length(radii_arr, dim, phi)

potential = InversePowerPeriodic(2, power, eps, [length, length],  radii_arr)



println(length)

coords = generate_random_coordinates(length, natoms, dim)



ba_qndf = BasinAssigner(QNDF(), 1e-4, 1e-4, 1e-6) # The convergence tolerance is error in gradient
ba_cvode = BasinAssigner(CVODE_BDF(), 1e-4, 1e-4, 1e-6)

# final_qndf= find_corresponding_minimum(ba_qndf, gradient_problem_function(potential), coords, 1000)
final_cvode= find_corresponding_minimum(ba_cvode, gradient_problem_function(potential), coords, 1000)

println(final)
println(final_cvode)




















