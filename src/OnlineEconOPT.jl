module OnlineEconOPT

using JuMP, Ipopt

export ModelParams, dynamic_RT_OPT

mutable struct ModelParams
    β::Real
    Cᵢ::Real
    κ::Real
    S::Real
    A::Real
    V::Real
    αꜝ::Real
    αꜝꜝ::Real
    L::Real
    Ce::Vector{Real}
    Cg::Vector{Real}
    Δt::Real
    T::Real
    Δx::Real
    Tᵢₘᵢₙ::Real # in ᵒC
    Tᵢₘₐₓ::Real # in ᵒC
    Tᵢᵒ::Real # initial condition in ᵒC
    Twᵒ::Vector{Real} # boundary condition in ᵒC
    Tₐ::Vector{Real} # ambient temperature in ᵒC

    function ModelParams(; β = 1e-3,
        Cᵢ = 8325,
        κ = 1.16e-4,
        S = 3500 * (20), # the *50 is just to scale things up (until the units problem is fixed)
        A = 1000,
        V = 10000,
        αꜝ = 4.64e-3,
        αꜝꜝ = 1.16e-2,
        L = 0.2,
        Ce = [0.12 * ones(13); 0.04 * ones(11)], # 9am to 10pm offpeak (first 13 hours, assuming we start from 9am)
        Cg = 0.10 * ones(24),
        Δt = 1,
        T = 24,
        Δx = 0.05,
        Tᵢₘᵢₙ = (69 − 32) * 5/9,
        Tᵢₘₐₓ = (77 - 32) * 5/9,
        Tᵢᵒ = (70 - 32) * 5/9, # initial condition
        Twᵒ = (50 * ones(4) .- 32) .* 5/9,
        Tₐ = (55 * ones(24) .- 32) .* 5/9 + 10 * randn(24))
        new(β, Cᵢ, κ, S, A, V, αꜝ, αꜝꜝ, L, Ce, Cg, Δt, T, Δx, Tᵢₘᵢₙ, Tᵢₘₐₓ, Tᵢᵒ, Twᵒ, Tₐ)
    end
end

function dynamic_RT_OPT(Params::ModelParams)
    model = Model(Ipopt.Optimizer);
    Nₜ = Int(Params.T ÷ Params.Δt); Nₓ = Int(Params.L ÷ Params.Δx) # Nₓ has to be 4 at least

    # Optimization Variables
    @variable(model, ϕ_h_gas[1:Nₜ] .>= 0)
    @variable(model, ϕ_h_elect[1:Nₜ] .>= 0)
    @variable(model, ϕ_c_elect[1:Nₜ] .>= 0)
    @variable(model, Params.Tᵢₘₐₓ .>= Tᵢ[1:Nₜ] .>= Params.Tᵢₘᵢₙ)
    @variable(model, Tw[1:Nₜ, 1:Nₓ])

    # Initial Conditions
    @constraint(model, Tw[1,:] == Params.Twᵒ)
    @constraint(model, Tᵢ[1] == Params.Tᵢᵒ)

    # Dynamic Constraints
    forcingfunc(k) = 1/Params.Cᵢ * (ϕ_h_gas[k]+ϕ_h_elect[k]-ϕ_c_elect[k]- Params.S * Params.αꜝ * (Tᵢ[k]-Tw[k,1]))
        # Implicit (Backward) Euler in time
    @constraint(model, [k in 1:Nₜ-1], Tᵢ[k+1] == Tᵢ[k] + Params.Δt * forcingfunc(k+1))
        # Centered Difference in space
    @constraint(model, [k in 1:Nₜ-1, j in 2:Nₓ-1], Tw[k+1,j] == 
                    Tw[k,j] + Params.Δt * Params.β/(Params.Δx)^2 * (Tw[k+1,j+1]-2*Tw[k+1,j]+Tw[k+1,j-1]))
    # Boundary Conditions (O(Δx²) forward and backward diff. first derivatives at the tips)
    @constraint(model, [k in 2:Nₜ], 0 == Params.αꜝ*(Tᵢ[k]-Tw[k,1]) + Params.κ/(2*Params.Δx)*(-3*Tw[k,1]+4*Tw[k,2]-Tw[k,3]))
    @constraint(model, [k in 2:Nₜ], 0 == Params.αꜝꜝ*(Tw[k,Nₓ]-Params.Tₐ[k]) + Params.κ/(2*Params.Δx)*(3*Tw[k,Nₓ]-4*Tw[k,Nₓ-1]+Tw[k,Nₓ-2]))

    # Cost Function
    @objective(model, Min, Params.Cg' * ϕ_h_gas + Params.Ce' * (ϕ_h_elect .+ ϕ_c_elect));
    set_silent(model)
    optimize!(model);
    if !is_solved_and_feasible(model)
        error("Solver did not find an optimal solution")
    end
    AchievedCost = objective_value(model)
    Tᵢ_t, ϕ_h_gas_t, ϕ_h_elect_t, ϕ_c_elect_t, Tw_t = value.(Tᵢ), value.(ϕ_h_gas), value.(ϕ_h_elect), value.(ϕ_c_elect), value.(Tw);
    return AchievedCost, Tᵢ_t, ϕ_h_gas_t, ϕ_h_elect_t, ϕ_c_elect_t, Tw_t
end



end