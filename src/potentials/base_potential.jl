abstract type AbstractPotential end

function system_energy(potential, x)
    dim = potential.dim
    natoms::Integer = size(x, 1) / dim

    if natoms * dim != size(x, 1)
        throw(DimensionMismatch(x, "coordinates have the wrong dimensions"))
    end

    if natoms != size(potential.radii, 1)
        throw(
            DimensionMismatch(
                x,
                "mismatch between dimensions of radii and number of atoms",
            ),
        )
    end

    x_ = reshape(x, (potential.dim, natoms))

    energy::Real = 0
    for i = 1:natoms
        for j = (i+1):natoms
            @inbounds r2 = peuclidean(x_[:, i], x_[:, j], potential.box_vec)^2
            @inbounds energy +=
                pairwise_energy(potential, r2, (potential.radii[i] + potential.radii[j]))
        end
    end
    return energy
end


@inline function unravel_index(i, j, ndim)
    return i * 2 - 1 + j * 2 - 1
end

function system_energy(potential, x)
    dim = potential.dim
    natoms::Integer = size(x, 1) / dim

    if natoms * dim != size(x, 1)
        throw(DimensionMismatch(x, "coordinates have the wrong dimensions"))
    end
    println(natoms)
    println(dim)
    if natoms != size(potential.radii, 1)
        throw(
            DimensionMismatch(
                x,
                "mismatch between dimensions of radii and number of atoms",
            ),
        )
    end

    x_ = reshape(x, (potential.dim, natoms))

    energy::Real = 0
    for i = 1:natoms
        for j = (i+1):natoms
            @inbounds r2 = peuclidean(x_[:, i], x_[:, j], potential.box_vec)^2
            @inbounds energy +=
                pairwise_energy(potential, r2, (potential.radii[i] + potential.radii[j]))
        end
    end
    return energy
end


function system_gradient!(g, potential, x)
    f(x_) = system_energy(potential, x_)
    ForwardDiff.gradient!(g, f, x)
    nothing
end

function system_negative_gradient!(g, potential, x)
    f(x_) = system_energy(potential, x_)
    ForwardDiff.gradient!(g, f, x)
    g .= -g
    nothing
end


function system_hessian!(potential, x)
    f(x_) = system_energy(potential, x_)
    return ForwardDiff.hessian(f, x)
end
