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
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
dim = 2.0
phi = 0.9
power = 2.5
eps = 1


length = get_box_length(radii_arr, phi, dim)



volume = length^3
println(volume)
println


potential_qndf = InversePowerPeriodic(2, power, eps, [length, length], radii_arr)
potential_cvode = InversePowerPeriodic(2, power, eps, [length, length], radii_arr)

coords = generate_random_coordinates(length, natoms, dim)

tol = 1e-6
ba_qndf = BasinAssigner(QNDF(), tol, tol, 1e-4) # The convergence tolerance is error in gradient
ba_cvode_bdf = BasinAssigner(CVODE_BDF(), tol, tol, 1e-4)
ba_auto_switch = BasinAssigner(AutoTsit5(Rosenbrock23()), tol, tol, 1e-4)





final_qndf = find_corresponding_minimum(
    ba_qndf,
    gradient_problem_function_qndf!(potential_qndf),
    coords,
    500,
    potential_qndf,
)

final_cvode = find_corresponding_minimum(
    ba_cvode_bdf,
    gradient_problem_function_cvode!(potential_cvode),
    coords,
    1000,
    potential_cvode,
)

# final_auto_switch = find_corresponding_minimum(ba_auto_switch, gradient_problem_function_qndf!(potential_qndf), coords, 500, potential_qndf)


println(potential_cvode.f_eval)
println(potential_cvode.jac_eval)
# println(final_qndf)
println(final_cvode)
# println(final_auto_switch)




println("Hessian CVODE")
# println(eigvals(system_hessian!(potential_cvode, final_qndf[1])))
println("Final Hessian QNDF")
# println(eigvals(system_hessian!(potential_qndf, final_cvode[1])))
