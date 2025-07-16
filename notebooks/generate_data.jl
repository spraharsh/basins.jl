using DiffEqBase, OrdinaryDiffEq, Sundials, Plots, ODEInterface, ODEInterfaceDiffEq, LSODA, ModelingToolkit, DiffEqDevTools
using Interpolations, JLD2
using DataStructures
using DataFrames, CSV

include("../src/potentials/inversepower.jl")
include("../src/minimumassign/minimumassign.jl")
include("../src/utils/utils.jl")

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


# Your generate_data_for_seed function
function generate_data_for_seed(seed, tols, natoms, phi, power, eps, dim, tspan, output_dir)

    du_arr = zeros(2 * natoms)
    function gradient_stop_condition(u, t, integrator, gtol)
        integrator.f(du_arr, u, integrator.p, t)
        gnorm_normalized = norm(du_arr) / sqrt(length(du_arr))
        return gnorm_normalized - gtol
    end
    true_sol_stop_condition(u, t, integrator) = gradient_stop_condition(u, t, integrator, 1e-12)
    other_sol_stop_condition(u, t, integrator) = gradient_stop_condition(u, t, integrator, 1e-10)
    affect!(integrator) = terminate!(integrator)
    true_sol_callback = ContinuousCallback(true_sol_stop_condition, affect!)
    other_sol_callback = ContinuousCallback(other_sol_stop_condition, affect!)
    # Create output directory if it doesn't exist
    mkpath(output_dir)

    radii_arr = generate_radii(seed, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
    box_length = get_box_length(radii_arr, phi, dim)
    potential = InversePowerPeriodic(2, power, eps, [box_length, box_length], radii_arr)
    coords = generate_random_coordinates(box_length, natoms, dim)
    odefunc = gradient_problem_function_all!(potential)
    prob = ODEProblem{true}(odefunc, coords, tspan)

    # Solve and save true solution
    truesol = solve(prob, CVODE_BDF(), abstol = 1e-12, reltol = 1e-12)

    tinterps = get_times_for_dist_fraction(truesol, coords)
    trueuinterps = truesol.(tinterps)
    println("tinterps: ", tinterps)
    tsend = truesol.u[end]
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
        alg_list_8 = [CVODE_BDF(), QNDF(autodiff = false), FBDF(autodiff = false), RK4(), ROCK2(), ImplicitEuler(autodiff=false), Rosenbrock23(autodiff=false), lsoda(), KenCarp47(autodiff=false), Tsit5()]
        alg_list_other = [lsoda(), CVODE_BDF(), QNDF(autodiff=false), FBDF(autodiff=false)]
        alg_list = alg_list_8 
        if natoms > 16
            alg_list = alg_list_other
        end
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



# Parse arguments
seed = parse(Int, ARGS[1])
natoms = parse(Int, ARGS[2])  # Added natoms as a command-line argument

# Parameters
phi = 0.9
power = 2.0
eps = 1
dim = 2.0
tspan = (0, 10000.0)
tols = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]

output_dir = "data_n_$(natoms)_trajectory_data_more_tols_with_lsoda_harmonic"
# Generate data for the given seed and natoms
data = generate_data_for_seed(seed, tols, natoms, phi, power, eps, dim, tspan, output_dir)

# Refactor csv save path to use output_dir
csv_save_path = "$(output_dir)/data_seed_$(seed)_n_$(natoms)_traj_tols.csv"

# Save data to a CSV file
CSV.write(csv_save_path, data)

println("Data for seed $seed and natoms $natoms has been generated and saved in $output_dir.")
