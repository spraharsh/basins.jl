using DiffEqBase,
    OrdinaryDiffEq,
    Sundials,
    Plots,
    ODEInterface,
    ODEInterfaceDiffEq,
    LSODA,
    ModelingToolkit,
    DiffEqDevTools
using DataStructures
using Plots, DataFrames, CSV, Statistics

include("../src/potentials/inversepower.jl")
include("../src/minimumassign/minimumassign.jl")
include("../src/utils/utils.jl")

function benchmark_data(seeds, tols, natoms, phi, power, eps, dim, tspan)
    setups = [Dict(:alg => QNDF(autodiff = false)), Dict(:alg => CVODE_BDF())]
    names = ["QNDF", "CVODE_BDF"]
    all_plot_data = []
    for seed in seeds
        radii_arr = generate_radii(seed, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
        box_length = get_box_length(radii_arr, phi, dim)
        potential = InversePowerPeriodic(2, power, eps, [box_length, box_length], radii_arr)
        coords = generate_random_coordinates(box_length, natoms, dim)
        odefunc = gradient_problem_function_all!(potential)
        prob = ODEProblem{true}(odefunc, coords, tspan)
        truesol = solve(prob, CVODE_BDF(), abstol = 1 / 10^12, reltol = 1 / 10^12)
        tsend = truesol[end]

        plot_data = DataFrame(
            Tolerance = Float64[],
            Algorithm = String[],
            Time = Float64[],
            L2Error = Float64[],
            MaxDist = Float64[],
        )

        for tol in tols
            for alg in [CVODE_BDF(), QNDF(autodiff = false)]
                elapsed_time = @elapsed sol = solve(prob, alg, abstol = tol, reltol = tol, callback = other_sol_callback)
                max_dist = maximum([
                    sqrt((sol[end][i] - tsend[i])^2 + (sol[end][i+1] - tsend[i+1])^2)
                    for i = 1:2:length(sol[end])
                ])
                disp_norm = norm(sol[end] - tsend)
                push!(
                    plot_data,
                    (tol, string(nameof(typeof(alg))), elapsed_time, disp_norm, max_dist),
                )
            end
        end

        push!(all_plot_data, plot_data)
    end

    # Combine data from all seeds
    combined_data = vcat(all_plot_data...)

    # Group by Tolerance and Algorithm, then calculate the average time and maximum errors
    grouped_data = combine(
        groupby(combined_data, [:Tolerance, :Algorithm]),
        :Time => mean => :AvgTime,
        :L2Error => maximum => :MaxL2Error,
        :MaxDist => maximum => :MaxMaxDist,
    )

    # Save combined data
    CSV.write("solver_comparison_data_avg_time_max_error.csv", grouped_data)

    # Create and save plots
    p1 = plot(
        grouped_data.MaxL2Error,
        grouped_data.AvgTime,
        group = grouped_data.Algorithm,
        xscale = :log10,
        yscale = :log10,
        xlabel = "Max L2 Error",
        ylabel = "Average Time (s)",
        title = "Max L2 Error vs Average Time",
    )
    savefig(p1, "max_l2error_vs_avg_time.png")

    p2 = plot(
        grouped_data.MaxMaxDist,
        grouped_data.AvgTime,
        group = grouped_data.Algorithm,
        xscale = :log10,
        yscale = :log10,
        xlabel = "Max Distance",
        ylabel = "Average Time (s)",
        title = "Max Distance vs Average Time",
    )
    savefig(p2, "max_maxdist_vs_avg_time.png")

    return grouped_data
end


# Parameters
natoms = 8
phi = 0.9
power = 2.5
eps = 1
dim = 2.0
tspan = (0, 10000.0)

seeds = 1:10
tols = [1e-5, 1e-6, 1e-7, 1e-8]

wp_data = benchmark_data(seeds, tols, natoms, phi, power, eps, dim, tspan)