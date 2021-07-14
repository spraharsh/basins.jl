@doc raw"""
    Uses the Newton's method to find the minimum of the problem
    
"""

using Optim
include("../potentials/pele-interface.jl")
include("../utils/utils.jl")
using Krylov
using IterativeSolvers

natoms = 1024
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
dim = 2
phi = 0.9
power = 2.5
eps = 1

length_arr = get_box_length(radii_arr, phi, dim)

coords = generate_random_coordinates(length_arr, natoms, dim)
boxvec = [length_arr, length_arr]

utils = pyimport("pele.utils.cell_scale")
cell_scale = utils.get_ncellsx_scale(radii_arr, boxvec)

pele_pot = pot.InversePower(
    2.5,
    1.0,
    radii_arr,
    ndim = 2,
    boxvec = boxvec,
    use_cell_lists = false,
    ncellx_scale = cell_scale,
)
ippot = PythonPotential(pele_pot)



energyfun(x) = system_energy_pele(ippot, x)

function grad_func!(G, x)
    system_gradient_pele!(ippot, x, G)
end

function hess_func!(H, x)
    H .= system_hessian_pele(ippot, x)
end



rosenbrock(x) =  (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function g!(G, x)
G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
G[2] = 200.0 * (x[2] - x[1]^2)
end



function h!(H, x)
    H[1, 1] = 2.0 - 400.0 * x[2] + 1200.0 * x[1]^2
    H[1, 2] = -400.0 * x[1]
    H[2, 1] = -400.0 * x[1]
    H[2, 2] = 200.0
end

# result = optimize(rosenbrock, zeros(2), NewtonTrustRegion(); autodiff=:forward)

# result = optimize(rosenbrock, g!, h!, zeros(2), NewtonTrustRegion)

# optimize(energyfun, grad_func!, hess_func!, coords , NewtonTrustRegion())

# single step newton's method line search
# optimize(energyfun, grad_func!, hess_func!, coords, NewtonTrustRegion())

step = zeros(length(coords))

hessian = zeros((length(coords), length(coords)))




grad_func!(step, coords)
hess_func!(hessian, coords)

# println(step)
# println(hessian)
# Solve H x = - g

# (x, stats) = minres_qlp(hessian, step)

# (x, stats) = minres(hessian, step)

# hessian = sparse(hessian)


# x, stats = minres_qlp(hessian, step, history=true, itmax=10000, atol=10^-3, rtol=10^-3)

























