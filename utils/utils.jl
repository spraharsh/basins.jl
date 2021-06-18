using Random, Distributions
include("../potentials/inversepower.jl")


function generate_radii(seed, natoms, radius_1, radius_2, rstd_1, rstd_2)
    Random.seed!(seed)
    normal_1 = Normal(radius_1, rstd_1)
    normal_2 = Normal(radius_2, rstd_2)
    natoms_by_2 = convert(Int64, natoms / 2)
    radii_arr_1 = rand(normal_1, natoms_by_2)
    radii_arr_2 = rand(normal_2, natoms_by_2)
    return vcat(radii_arr_1, radii_arr_2)
end


function get_box_length(radii, phi, dim)
    if dim == 3
        vol_spheres = sum(4 / 3 * pi * (radii) .^ 3)
        box_length = (vol_spheres / phi)^(1 / 3)
    elseif dim == 2
        vol_discs = sum(pi * (radii) .^ 2)
        box_length = (vol_discs / phi)^(1 / 2)
    else
        throw("not implemented")
    end
    return box_length
end

function generate_random_coordinates(box_length, natoms, dim)
    dist = Uniform(0, box_length)
    println(natoms * dim)
    s = convert(Int64, natoms * dim)
    return rand(dist, s)
end


function generate_random_coordinates(seed::Int, box_length, natoms, dim)
    Random.seed!(seed)
    dist = Uniform(0, box_length)
    println(natoms * dim)
    s = convert(Int64, natoms * dim)
    return rand(dist, s)
end
