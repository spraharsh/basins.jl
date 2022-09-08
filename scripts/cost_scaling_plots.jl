@doc raw"""
Generates averaged performance scaling plots
"""


include("potentials/quick_define_potentials.jl")
include("minimumassign/minimumassign.jl")
using Sundials, DifferentialEquations, Plots


"""
Performance functions for CVODE at a given tolerance
"""
function perf_cvode(nparticles, seed, linear_solver, tol)
    pot, box_length = make_canonical_2d_IP_potential(nparticles)
    coords = generate_random_coordinates(seed, box_length, nparticles, 2)
    pele_wrapped_python_pot = PythonPotential(pot)
    odefunc_pele = gradient_problem_function_pele!(pele_wrapped_python_pot)
    tspan = (0, 100.0)
    prob = ODEProblem{true}(odefunc_pele, coords, tspan)
    @timed solve(prob, CVODE_BDF(linear_solver = linear_solver), abstol = tol, reltol = tol)
end


"""
Performance function for CVODE at finding the minimum corresponding to the 
"""
function perf_cvode_basin_finding(nparticles, seed, linear_solver, tol, minimum_tol)
    pot, box_length = make_canonical_2d_IP_potential(nparticles)
    coords = generate_random_coordinates(seed, box_length, nparticles, 2)
    pele_wrapped_python_pot = PythonPotential(pot)
    odefunc_pele = gradient_problem_function_pele!(pele_wrapped_python_pot)
    tspan = (0, 100.0)
    ba = BasinAssigner(CVODE_BDF(linear_solver = linear_solver), tol, tol, minimum_tol)
    @timed find_corresponding_minimum(
        ba,
        odefunc_pele,
        coords,
        100000,
        pele_wrapped_python_pot,
    )
end




time_cvode(nparticles, seed, linear_solver, tol) =
    perf_cvode(nparticles, seed, linear_solver, tol)[2]

time_cvode_basin_finding(nparticles, seed, linear_solver, tol, mtol) =
    perf_cvode_basin_finding(nparticles, seed, linear_solver, tol, mtol)[2]

# To compile
println(time_cvode_basin_finding(8, 0, :GMRES, 10^-5, 10^-5))
println(time_cvode_basin_finding(8, 0, :PCG, 10^-5, 10^-5))
println(time_cvode_basin_finding(8, 0, :BCG, 10^-5, 10^-5))
println(time_cvode_basin_finding(8, 0, :TFQMR, 10^-6, 10^-5))
println(time_cvode_basin_finding(8, 0, :Dense, 10^-6, 10^-5))


nparticles = map(n -> 2^n, 3:10)

solvers = [:Dense, :GMRES, :BCG, :PCG, :TFQMR]

seeds = 1:10

mean_run_time_arr = []
tol = 10^-5
mtol = 10^-5
for solver in solvers
    mean_run_time_for_nparticles = []
    for p in nparticles
        run_times = []
        for s in seeds
            run_time = time_cvode_basin_finding(p, s, solver, tol, mtol)
            push!(run_times, run_time)
        end
        mean_run_time = mean(run_times)
        push!(mean_run_time_for_nparticles, mean_run_time)
    end
    push!(mean_run_time_arr, mean_run_time_for_nparticles)
end

println(mean_run_time_arr)
plot(
    nparticles,
    mean_run_time_arr,
    label = ["Dense" "GMRES" "BCG" "PCG" "TFQMR"],
    xaxis = :log,
    yaxis = :log,
    markershape = :circle,
)
