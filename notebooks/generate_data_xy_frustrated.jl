#!/usr/bin/env julia
# generate_data_kuramoto.jl
using DiffEqBase, OrdinaryDiffEq, Sundials, Plots, ODEInterface, ODEInterfaceDiffEq, LSODA, ModelingToolkit, DiffEqDevTools
using Interpolations, JLD2
using DataStructures
using DataFrames, CSV
using LinearAlgebra
using Random
using SparseConnectivityTracer, ADTypes
using LinearSolve






function get_triangular_lattice_on_square_periodic_list(n::Int)

    # current version only works for even n
    # because there is a question whether the boundary is odd 
    # or even in the case of odd n
    if n % 2 != 0
        throw(ArgumentError("n must be even"))
    end


    # For n == 1, there is exactly one site with no neighbors.
    # Return a single-element array of an empty vector.
    if n == 1
        return [Int[]]
    end

    # Utility to wrap around 1..n periodically
    wrap_index(i) = i == 0 ? n : (i == n + 1 ? 1 : i)

    # Convert (row, col) => 1-based index in [1, n^2]
    convert1d(i, j) = (i - 1) * n + j

    # Initialize adjacency list: for each site 1..n^2, store a Vector{Int} of neighbors
    adjacency_list = [Int[] for _ = 1:(n^2)]

    # Loop over all sites in the n x n grid
    for i = 1:n
        for j = 1:n
            site = convert1d(i, j)

            # Basic 4 neighbors
            up = convert1d(wrap_index(i - 1), j)
            down = convert1d(wrap_index(i + 1), j)
            left = convert1d(i, wrap_index(j - 1))
            right = convert1d(i, wrap_index(j + 1))

            push!(adjacency_list[site], up)
            push!(adjacency_list[site], down)
            push!(adjacency_list[site], left)
            push!(adjacency_list[site], right)


            # Diagonal neighbors
            if isodd(i)
                # up-right, down-right if row i is odd
                up_right = convert1d(wrap_index(i - 1), wrap_index(j + 1))
                down_right = convert1d(wrap_index(i + 1), wrap_index(j + 1))
                push!(adjacency_list[site], up_right)
                push!(adjacency_list[site], down_right)
            else
                # up-left, down-left if row i is even
                up_left = convert1d(wrap_index(i - 1), wrap_index(j - 1))
                down_left = convert1d(wrap_index(i + 1), wrap_index(j - 1))
                push!(adjacency_list[site], up_left)
                push!(adjacency_list[site], down_left)
            end
        end
    end

    # Optionally remove duplicates (if any) from each neighbor list
    # TODO: really don't need this, (and adjacency should be a set anyway)
    for k in eachindex(adjacency_list)
        unique!(adjacency_list[k])  # ensures each neighbor appears only once
    end

    return adjacency_list
end

function get_times_for_dist_fraction(sol, uinit; space = range(0.0, 1.0, length = 100))
    us = sol.u
    ts = sol.t
    dists = [0.0]
    dist_curr = 0.0
    uprev = uinit
    for u in us[2:end]
        dist = norm(u - uprev)
        dist_curr += dist
        push!(dists, dist_curr)
    end

    # insert 0.0 at the beginning of ts
    tot_dist = dist_curr
    # space should only be from
    # dists vs ts is my function I want to invert over space
    space = space * tot_dist
    interpolation = linear_interpolation(dists, ts)
    return interpolation.(space)
end

# -------------------------
# Define tolerances and list of algorithms (alg_list_8)
# -------------------------
function get_times_for_dist_fraction(sol, uinit; space = range(0.0, 1.0, length = 100))
    us = sol.u
    ts = sol.t
    dists = [0.0]
    dist_curr = 0.0
    uprev = uinit
    for u in us[2:end]
        dist = norm(u - uprev)
        dist_curr += dist
        push!(dists, dist_curr)
    end

    # insert 0.0 at the beginning of ts
    tot_dist = dist_curr
    # space should only be from
    # dists vs ts is my function I want to invert over space
    space = space * tot_dist
    interpolation = linear_interpolation(dists, ts)
    return interpolation.(space)
end


calculate_max_distance(x, y) =
    maximum(sqrt((x[i] - y[i])^2 + (x[i+1] - y[i+1])^2) for i = 1:2:length(x))


# do sparse banded iterative



# Your generate_data_for_seed function
function generate_data_for_seed(seed, tols, side_length, tspan, output_dir)

    
    # Create output directory if it doesn't exist

    function ivp_rhs_kuramoto_with_adjacency!(
        dthetadt,     # Output: derivative of thetas
        thetas,       # Current state: theta_i
        adjacency_list::Vector{Vector{Int}}  # Adjacency list
    )
        @inbounds for i in eachindex(thetas)
            s = 0.0
            for nb in adjacency_list[i]
                s += -sin(thetas[nb] - thetas[i])
            end
            dthetadt[i] = s
        end
    end
    

    
    number_of_oscillators = side_length^2
    du_arr = zeros(number_of_oscillators)
    adj_list = get_triangular_lattice_on_square_periodic_list(side_length)
    # Define your Kuramoto ODE using the adjacency list
    function kuramoto_triangle_list!(dθ, θ, p, t)
        ivp_rhs_kuramoto_with_adjacency!(dθ, θ, adj_list)
    end


    thetas0 = rand(number_of_oscillators) * (2π)
    tspan = (0.0, 1000.0)
    # sparsity detection
    detector = TracerSparsityDetector()
    jac_sparsity = ADTypes.jacobian_sparsity(
        (du, u) -> kuramoto_triangle_list!(du, u, 0.0, 0.0), du_arr, thetas0, detector)

    ode_func = ODEFunction(kuramoto_triangle_list!, jac_prototype = float.(jac_sparsity))


    prob = ODEProblem{true}(kuramoto_triangle_list!, thetas0, tspan)
    prob_sparse = ODEProblem(ode_func, thetas0, tspan)

    

    # Solve and save true solution
    truesol = solve(prob, CVODE_BDF(), abstol = 1e-12, reltol = 1e-12)

    tinterps = get_times_for_dist_fraction(truesol, thetas0)
    trueuinterps = truesol.(tinterps)
    println("tinterps: ", tinterps)
    tsend = truesol.u[end]
    mkpath(output_dir)
    jldsave(joinpath(output_dir, "truesol_seed$(seed).jld2"); 
        t = truesol.t, 
        u = truesol.u
    )

    # also save interpolated true solution
    jldsave(joinpath(output_dir, "trueuinterps_seed$(seed).jld2"); 
        t = tinterps, 
        u = trueuinterps
    )

    plot_data = DataFrame(
        Seed = Int[],
        Tolerance = Float64[],
        Algorithm = String[],
        Time = Float64[],
        L2Error = Float64[],
        MaxDist = Float64[],
        MaxTrajDist = Float64[],
    )

    for tol in tols
        alg_list = [CVODE_BDF(linear_solver=:GMRES), QNDF(autodiff = false, linsolve = KrylovJL_GMRES()), FBDF(autodiff = false, linsolve = KrylovJL_GMRES()), Tsit5()]
        for alg in alg_list
            println("Seed: $seed, Tolerance: $tol, Algorithm: $(nameof(typeof(alg)))")
            alg_name = string(nameof(typeof(alg)))
            #    #elapsed_time = @elapsed sol = solve(prob, alg, abstol = tol, reltol = tol, callback = other_sol_callback)
            #    # unfortunately we don't have a callback for lsoda
            #    # so we just solve it for a little longer than tsol
            #    # note that the times are heavily affected by rtol and atol
            #    # so there is no clean comparison method
            elapsed_time = @elapsed sol = solve(prob, alg, abstol = tol, reltol = tol)


            # divide tinterps into stuff that is less than sol.t[end] and greater than sol.t[end]
            tinterps_less = tinterps[tinterps .<= sol.t[end]]
            tinterps_more = tinterps[tinterps .> sol.t[end]]

            

            uinterps = sol.(tinterps_less)
            # use the last value of sol as the solution for the rest of the time
            uinterps = vcat(uinterps, fill(sol[end], length(tinterps_more)))
            

            # Save the solution
            filename = "sol_$(alg_name)_tol$(tol)_seed$(seed).jld2"
            jldsave(joinpath(output_dir, filename); t= sol.t, u = sol.u)


            max_dist = calculate_max_distance(sol[end], tsend)
            # save the interpolated solution
            filename = "uinterps_$(alg_name)_tol$(tol)_seed$(seed).jld2"
            jldsave(joinpath(output_dir, filename); t= tinterps, u = uinterps)

            max_trajectory_dist = maximum(calculate_max_distance.(uinterps, trueuinterps))
            disp_norm = norm(sol[end] - tsend)
            push!(
                plot_data,
                (
                    seed,
                    tol,
                    alg_name,
                    elapsed_time,
                    disp_norm,
                    max_dist,
                    max_trajectory_dist,
                ),
            )
            println("Completed Seed: $seed, Tolerance: $tol, Algorithm: $(nameof(typeof(alg)))")
        end
    end
    return plot_data
end



# -------------------------
# Parse command-line arguments
# -------------------------
if length(ARGS) < 2
    println("Usage: julia generate_data_kuramoto.jl <seed> <side_length>")
    exit(1)
end

seed = parse(Int, ARGS[1])
side_length = parse(Int, ARGS[2])  # number of oscillators
Random.seed!(seed)

# -------------------------
tspan = (0, 10000.0)
tols = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

output_dir = "frustrated_xy_side_length_$(side_length)_trajectory_data_iterative_cleaned_up"
# Generate data for the given seed and natoms
data = generate_data_for_seed(seed, tols, side_length, tspan, output_dir)

# Refactor csv save path to use output_dir
csv_save_path = "$(output_dir)/data_frustrated_kuramoto_seed_$(seed)_side_length_$(side_length)_traj_tols.csv"

# Save data to a CSV file
CSV.write(csv_save_path, data)

println("Data for seed $seed and number of oscillators $(side_length) has been generated and saved in $output_dir.")