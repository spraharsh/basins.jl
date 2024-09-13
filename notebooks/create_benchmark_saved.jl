using Distributed

# Add worker processes if not already done
if nworkers() == 1
    addprocs(Sys.CPU_THREADS - 1)
end

@everywhere using DiffEqBase,
    OrdinaryDiffEq,
    Sundials,
    ODEInterface,
    ODEInterfaceDiffEq,
    LSODA,
    ModelingToolkit,
    DiffEqDevTools
@everywhere using DataStructures, Statistics
@everywhere using Plots, DataFrames, CSV

@everywhere include("../src/potentials/inversepower.jl")
@everywhere include("../src/minimumassign/minimumassign.jl")
@everywhere include("../src/utils/utils.jl")

@everywhere function process_seed(seed, tols, natoms, phi, power, eps, dim, tspan)
    radii_arr = generate_radii(seed, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
    box_length = get_box_length(radii_arr, phi, dim)
    potential = InversePowerPeriodic(2, power, eps, [box_length, box_length], radii_arr)
    coords = generate_random_coordinates(box_length, natoms, dim)
    odefunc = gradient_problem_function_all!(potential)
    prob = ODEProblem{true}(odefunc, coords, tspan)

    benchsol = solve(prob, CVODE_BDF(), abstol = 1 / 10^10, reltol = 1 / 10^10)

    setups = [Dict(:alg => QNDF(autodiff = false)), Dict(:alg => CVODE_BDF())]
    names = ["QNDF", "CVODE_BDF"]

    wpset = WorkPrecisionSet(
        prob,
        tols,
        tols,
        setups;
        names = names,
        error_estimate = :final,
        appxsol = sol,
        maxiters = Int(1e5),
        numruns = 1,
    )

    for tol in tols
        for alg in [QNDF(autodiff = false), CVODE_BDF()]
            sol = solve(prob, alg, abstol = tol, reltol = tol, maxiters = Int(1e5))
            # get the end points

        end
    end
    work_precision_data = Dict()
    for work_precision in wpset.wps
        data = [work_precision.times, work_precision.errors]
        push!(get!(work_precision_data, work_precision.name, []), data)
    end
    print("don")
    return work_precision_data
end

function benchmark_data(seeds, tols, natoms, phi, power, eps, dim, tspan)
    results = @distributed (merge) for seed = 1:seeds
        process_seed(seed, tols, natoms, phi, power, eps, dim, tspan)
    end
    return results
end

# Main script
natoms = 8
dim = 2.0
phi = 0.9
power = 2.5
eps = 1
tspan = (0, 10000.0)
seeds = 5
tols = [1e-7, 1e-8]

wp_data = benchmark_data(seeds, tols, natoms, phi, power, eps, dim, tspan)

println("wp_data out-------")
println(wp_data)


# Save data and create plots
mean_data = Dict()

for (name, data) in wp_data
    name_df = DataFrame()
    for (i, (times, errors)) in enumerate(data)
        name_df =
            hcat(name_df, DataFrame(times = times, errors = errors), makeunique = true)
    end

    mean_times = mean(Array(name_df[!, r"times"]), dims = 2)
    mean_errors = mean(Array(name_df[!, r"errors"]), dims = 2)

    mean_data[name] = (times = vec(mean_times), errors = vec(mean_errors))

    CSV.write("inversepower_$(natoms)atoms_$(seeds)_average_$name.csv", name_df)
end
println("mean data")
# Plotting
p = plot(
    title = "Mean Errors vs Mean Times",
    xlabel = "Mean Error",
    ylabel = "Mean Time",
    xscale = :log10,
    yscale = :log10,
)

for (name, (times, errors)) in mean_data
    plot!(p, errors, times, label = name, marker = :circle)
end

savefig(p, "mean_errors_vs_times_plot.png")
display(p)