@doc raw"""
testing utilities to ensure they work
"""

using Test
include("../utils/utils.jl")

natoms = 200000
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
natoms_by_2 = convert(Integer, natoms / 2)

@test isapprox(mean(radii_arr[1:natoms_by_2]), 1.0, rtol = 1e-2, atol = 1e-2)
@test isapprox(mean(radii_arr[natoms_by_2+1:natoms]), 1.4, rtol = 1e-2, atol = 1e-2)
@test isapprox(mean(radii_arr[1:natoms_by_2]), 1.0)
