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
radii_arr = generate_radii(0, natoms, 1.,1.4, 0.05, 0.05*1.4)
dim = 2.0
phi = 0.9
power = 2.5
eps = 1

length = get_box_length(radii_arr, dim, phi)

potential_qndf = InversePowerPeriodic(2, power, eps, [length, length],  radii_arr)
potential_cvode = InversePowerPeriodic(2, power, eps, [length, length],  radii_arr)



println(length)

coords = generate_random_coordinates(length, natoms, dim)


tol = 1e-9
ba_qndf = BasinAssigner(QNDF(), tol, tol, 1e-6) # The convergence tolerance is error in gradient
ba_cvode = BasinAssigner(CVODE_Adams(), tol, tol, 1e-6)
ba_auto_switch = BasinAssigner(VCABM(), tol, tol, 1e-6)


final_qndf = find_corresponding_minimum(ba_qndf, gradient_problem_function_qndf!(potential_qndf), coords, 10000)

final_cvode = find_corresponding_minimum(ba_cvode, gradient_problem_function_cvode!(potential_cvode), coords, 10000)

final_auto_switch = find_corresponding_minimum(ba_auto_switch, gradient_problem_function_qndf!(potential_qndf), coords, 10000)


println(potential_cvode.f_eval)
println(potential_cvode.jac_eval)
println(final_qndf)
println(final_cvode)


println("Hessian CVODE")
println(eigvals(system_hessian!(potential_cvode, final_qndf[1])))
println("Final Hessian QNDF")
println(eigvals(system_hessian!(potential_qndf, final_cvode[1])))






















