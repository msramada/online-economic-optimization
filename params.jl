β = 1e-3
Δt = 1
T = 24

Δx = 0.05
L = 0.2
Nₜ = Int(T ÷ Δt); Nₓ = Int(L ÷ Δx) # Nₓ has to be 4 at least
Tᵢₘᵢₙ = (69 − 32) * 5/9
Tᵢₘₐₓ = (77 - 32) * 5/9
Tᵢᵒ = (70 - 32) * 5/9 # initial condition
Twᵒ = (50 * ones(Nₓ) .- 32) .* 5/9
Tₐ = (75 * ones(Nₜ) .- 32) .* 5/9 + 10 * randn(Nₜ)
κ = 1.16e-4
Cᵢ = 8325

S = 3500
A = 1000
V = 10000
αꜝ = 4.64e-3
αꜝꜝ = 1.16e-2
C_elec_onPeak = 0.12 # 9am to 10pm (first 13 hours, we start from 9am)
C_elec_offPeak = 0.04 # the rest
C_gas = 0.10