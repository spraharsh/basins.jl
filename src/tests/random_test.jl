using DifferentialEquations
using Sundials

func_eval = 0
function n_grad(u, p, t)
    global func_eval += 1
    return -u
end

ode_func = ODEFunction(n_grad)
tspan = (0, 1.0)
initial_point = [1.0]
prob = ODEProblem(ode_func, initial_point, tspan)
integrator_cvode = init(prob, CVODE_BDF(), reltol = 1e-4, abstol = 1e-4)


step!(integrator_cvode)
step!(integrator_cvode)
print("function calls: ")
println(func_eval)
print(func_eval)
println(integrator_cvode.sol.destats)
println(integrator_cvode.sol.destats)
