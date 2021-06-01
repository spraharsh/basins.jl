"""
Tests for the potential to ensure it's the same as what we're using in C++
"""

include("../potentials/inversepower.jl")
using Test
using DelimitedFiles




test_radii = [1.08820262, 1.02000786, 1.0489369,  1.11204466, 1.53072906, 1.33159055,  1.46650619, 1.389405]

test_coords = [6.43109269, 2.55893249,
               5.28365038,3.52962924,
               3.79089799, 6.1770549,
               0.47406571, 0.58146545,
               0.13492935, 5.55656566,
               5.19310115, 5.80610666,
               6.53090015, 5.3332587,
               3.07972527, 5.20893375]

test_box_length = 6.673592625078725
box_vec = [test_box_length, test_box_length]

pot = InversePowerPeriodic(2, 2.5, 1, box_vec, test_radii)

cpp_hessian = readdlm("hessian.csv",',')
cpp_gradient = readdlm("gradient.csv",',')
cpp_energy = readdlm("energy.csv",',')

julia_energy = system_energy(pot, test_coords)
julia_gradient = system_gradient(pot, test_coords)
julia_hessian = system_hessian(pot, test_coords)

@test all(cpp_energy .≈ julia_energy)
@test all(cpp_hessian .≈ julia_hessian)
@test all(cpp_gradient .≈ julia_gradient)








