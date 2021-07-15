using Sundials: linear_solver
using Base: Float64
"""
Tests for the minimum finder to match previous QNDF code through the python interface that was tested
"""

include("../potentials/inversepower.jl")
include("../minimumassign/minimumassign.jl")
include("../utils/utils.jl")
using Test
using Random, Distributions
using Sundials
test_radii = [
    1.0339553713017888,
    1.0414206741450018,
    0.9823496299849702,
    0.9932573064034739,
    1.0293308537316554,
    1.0148667925424708,
    1.003247377427417,
    0.9945491307459141,
    1.3640052726416674,
    1.510203114149589,
    1.3517765010522012,
    1.3466037328512679,
    1.4278237686452713,
    1.456814073575481,
    1.375755177700484,
    1.3868699122638386,
]

test_coords = [
    5.26547250894492,
    7.938874364818388,
    8.8488166077642,
    7.019171417218076,
    4.290953061383954,
    0.5702411025692091,
    3.228738571851173,
    7.018362820827343,
    0.3944484188752373,
    2.4502458959085827,
    0.6111925985564105,
    1.4321652375868488,
    5.534368996139129,
    1.2411440977435457,
    7.6631057250472505,
    8.363424965543553,
    2.7436551939555747,
    6.60918361882509,
    1.0940112754127833,
    7.01349620997795,
    7.33217401704462,
    0.3231634405273944,
    4.431363975462053,
    8.22158712979583,
    8.701529612544768,
    7.324813082841852,
    1.1367116491304365,
    1.0447858894419346,
    0.7273850491669233,
    7.101313208494044,
    0.9584178814432026,
    7.66271463627935,
]

test_box_length = 9.143232784798267
box_vec = [test_box_length, test_box_length]


pot = InversePowerPeriodic(2, 2.5, 1, box_vec, test_radii)


ode_problem = gradient_problem_function_all!(pot)

tol = 1e-6

ba_qndf = BasinAssigner(QNDF(autodiff = false), tol, tol, 1e-6) # The convergence tolerance is error in gradient
ba_cvode_bdf = BasinAssigner(CVODE_BDF(), tol, tol, 1e-6)
ba_auto_switch = BasinAssigner(AutoTsit5(Rosenbrock23(autodiff = false)), tol, tol, 1e-6)

@time final_cvode =
    find_corresponding_minimum(ba_cvode_bdf, ode_problem, test_coords, 1000, pot)
println(final_cvode)

@time final_qndf = find_corresponding_minimum(ba_qndf, ode_problem, test_coords, 1000, pot)
println(final_qndf)
