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


include("../src/potentials/inversepower.jl")
include("../src/minimumassign/minimumassign.jl")
include("../src/utils/utils.jl")


natoms = 16
radii_arr = generate_radii(0, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
dim = 2.0
phi = 0.9
power = 2.5
eps = 1

length = get_box_length(radii_arr, phi, dim)

potential = InversePowerPeriodic(2, power, eps, [length, length], radii_arr)

coords = generate_random_coordinates(length, natoms, dim)

tol = 1e-4

odefunc = gradient_problem_function_all!(potential)
tspan = (0, 100.0)



function benchmark_data(seeds, tols, natoms, phi, power, eps, dim, tspan)
    setups = [
        Dict(:alg => QNDF(autodiff = false)),
        #Dict(:alg=>rodas()),
        Dict(:alg => CVODE_BDF()),
        #Dict(:alg=>Rodas4(autodiff=false)),
        #Dict(:alg=>Rodas5(autodiff=false)),
        Dict(:alg => RadauIIA5(autodiff = false)),
        #Dict(:alg=>lsoda()),
        #Dict(:alg=>ImplicitEuler(autodiff=false))
    ]
    names = ["QNDF", "CVODE_BDF", "Implicit RK4"]
    work_precision_data = DefaultDict([])
    for seed in seeds
        generate_radii(seed, natoms, 1.0, 1.4, 0.05, 0.05 * 1.4)
        length = get_box_length(radii_arr, phi, dim)
        potential = InversePowerPeriodic(2, power, eps, [length, length], radii_arr)
        coords = generate_random_coordinates(length, natoms, dim)
        odefunc = gradient_problem_function_all!(potential)
        prob = ODEProblem{true}(odefunc, coords, tspan)
        # benchmark solution
        sol = solve(prob, CVODE_BDF(), abstol = 1 / 10^12, reltol = 1 / 10^12)
        println("prob", prob)
        println("tol", tols)
        println("setups", setups)
        println(names, "names")
        println(sol, "sol")
        wpset = WorkPrecisionSet(
            prob,
            tols,
            tols,
            setups;
            names = names,
            error_estimate = :l2,
            appxsol = sol,
            maxiters = Int(1e5),
            numruns = 5,
        )
        for work_precision in wpset.wps
            data = [work_precision.times, work_precision.errors]
            push!(work_precision_data[work_precision.name], data)
        end
    end
    return work_precision_data
end


seeds = 1:100

wp_data =
    benchmark_data(seeds, [1e-5, 1e-6, 1e-7, 1e-8], natoms, phi, power, eps, dim, tspan)

# save the data

# save each element as a separate CSV
using CSV, DataFrames
for (name, data) in wp_data
    name_df = DataFrame()
    for (i, (times, errors)) in enumerate(data)
        name_df =
            hcat(name_df, DataFrame(times = times, errors = errors), makeunique = true)
    end
    CSV.write("inversepower_16atoms_100_average_$name.csv", name_df)
end
