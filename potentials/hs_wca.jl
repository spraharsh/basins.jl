@doc raw"""
  This a WCA-like potential, which mostly avoids square roots and
  extrapolates linearly "into the hard core".
  That should be useful to minimize HS-WCA-like systems.
  The pair potential is:
  V_\text{sfHS-WCA}(r^2) = 0                                               \text{ if } r \geq r_S
  V_\text{sfHS-WCA}(r^2) = V_\text{fHS-WCA}(r^2)                           \text{ if } r_\times < r < r_S
  V_\text{sfHS-WCA}(r^2) = E_\times - (\sqrt{r^2} - r_\times)G_\times      \text{ if } r \leq r_\times
  Here:
  E_\times = V_\text{fHS-WCA}(r_\times)
  G_\times = \text{grad}[V_\text{fHS-WCA}](r_\times)
  And:
  radius_sum : sum of hard radii
  r_S = (1 + \alpha) * radius_sum
  r_\times = radius_sum + \delta
  The choice of the delta parameter below is somewhat arbitrary and
  could probably be optimised.
  Computing the gradient GX at the point where we go from fWCA to
  linear is somewhat confusing because the gradient is originally
  computed as grad / (-r).
"""


include("base_potential.jl")
using Distances
using ForwardDiff




mutable struct HS_WCAPeriodic <: AbstractPotential
    dim::Integer
    epsilon::Real
    delta::Real
    prfac::Real
    alpha_12::Real
    exp::Int
    # TODO: auto calculate from dim and box length to prevent DimensionMismatch
    box_vec::Any
    radii::Any
    # wrote these to ensure that the function evaluations are counted accurately
    # ideally they shouldn't be in this struct
    f_eval::Any
    jac_eval::Any
    HS_WCAPeriodic(dim, epsilon, delta, prfac, alpha_12, exp, box_vec, radii, f_eval, jac_eval) =
        new(dim, epsilon, delta, prfac, alpha_12, exp, box_vec, radii, f_eval, jac_eval)
end


function HS_WCAPeriodic(dim, eps, alpha, delta, exp, box_vec, radii)
	alpha_12 = (alpha+1)^2
    prfac = 0.5*(2*alpha + alpha^2)^exp
    return HS_WCAPeriodic(dim, eps, delta, prfac, alpha_12, exp, box_vec, radii, 0, 0)
end


function pairwise_energy(potential::HS_WCAPeriodic, r2::Real, radius_sum::Real)
    # Hard sphere radius
	r_H2 = radius_sum^2
    # Soft sphere radius
    r_S2 = potential.alpha_12*r_H2
    
    if r2 >r_S2
        return 0
    end

    r_X = radius_sum + potential.delta
    r_X2 = r_X^2


    if (r2 > r_X2)
        dr = r2 - r_H2
        ir = 1/dr
        r_H2_ir = r_H2 * ir
        C_ir_m = potential.prfac * r_H2_ir^(potential.exp)
        C_ir_2m = C_ir_m^2
        EX = max(0, 4*potential.epsilon*(C_ir_2m-C_ir_m) + potential.epsilon)
        return EX
    end
    dr = r_X2 - r_H2

    ir = 1/dr
    r_H2_ir = r_H2 * ir
    C_ir_m = potential.prfac * r_H2_ir^(potential.exp)
    C_ir_2m = C_ir_m^2
    grad_prfac = 8*potential.exp
    grad_prfac2 = 16*potential.exp
    EX = max(0, 4*potential.epsilon*(C_ir_2m-C_ir_m) + potential.epsilon)
    GX = (potential.epsilon * (grad_prfac2 * C_ir_2m - grad_prfac * C_ir_m) * ir) * (-r_X)
    return EX + GX * (r2^(0.5) - r_X)
end


if abspath(PROGRAM_FILE) == @__FILE__
    # Periodicity check

    x = [-1.26049482, -0.95521073,  1.26534317, -0.3978736 , -2.03585937,
       -2.12901647,  2.31972795, -2.13775773,  0.38452674,  1.23081331,
        1.69439858, -1.14643869, -0.22386022,  0.22083366, -2.16366322,
        2.1621745 ,  0.27984705,  0.40418538, -0.03676139,  0.32428513,
        0.06877512, -1.2921155 ,  1.62574596,  1.37820726,  0.86374639,
       -1.08976784,  1.48627568, -2.50258403, -2.04333088, -2.47876631,
        0.9743414 ,  1.78943571,  1.63232619, -1.63667732, -0.96949313,
       -0.83257371, -0.11080114, -2.18505503,  0.06158633, -1.56067732,
        1.60050288, -0.88748628,  1.25802853, -1.10362087, -1.0863623 ,
        2.50590404,  0.19892436, -2.37223693]
    radii = [1.92391561, 1.97588283, 1.79582555, 2.05414624, 2.10975251,
             2.0876029 , 1.94621551, 2.08321329, 1.85513953, 1.85981893,
             2.05620311, 1.75396169, 1.8238525 , 2.03632431, 1.97105859,
             2.05234171]

    eps = 1.0
    sca = 0.1186889420813968
    box_vec = [5.0353449208573426, 5.0353449208573426, 5.0353449208573426]
    
    # extra pieces
    dim = 3
    alpha = sca
    delta = 1e-10
    exp =6
    pot = HS_WCAPeriodic(dim, eps, alpha, delta, exp, box_vec, radii)
    println(system_energy(pot, x))
    # println(system_hessian!(pot, x))
end
