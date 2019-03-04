# this version encapsulates
# a learning rule for a spike NN
# that uses fixed thresholds and interconected layers via "neurotransmitter"
# 𝛆 = threhsold
# ϑ = weights
# ψ = neur-transmitter
include("..\\global_utility_functions\\activation_functions.jl")
include("..\\global_utility_functions\\loss_functions.jl")
using Distributions

FloatN = Float32

mutable struct Layer
    S::AbstractArray{FloatN, 1}
    decay::AbstractArray{FloatN, 1}
    ϑ::AbstractArray{FloatN, 2}
    ψ::AbstractArray{FloatN, 1}
    𝛆::AbstractArray{FloatN, 1}
end

function test_activation(l::Layer)
    l.𝛆 .* (l.S .>= l.𝛆)
end
function d_test_activation(l::Layer)
    (l.S .>= l.𝛆)
end

function forward_update!(l::Layer, x)
    l.S .= l.S .- (test_activation(l) .* l.𝛆)
    l.S = (l.S .* l.decay + l.ϑ * x) .* l.ψ
    return test_activation(l), l.S
end

function backward_update!(l::Layer, x, dEds; lr=0.003)
    dsdϑ = (d_test_activation(l) .* l.ψ) * x'
    l.ϑ -= lr .* dEds * dsdϑ
end


init_states = [0, 0, 0]
init_decays = [0.99, 0.99, 0.99]
init_weights = rand(Uniform(0, 0.1), 3, 4)
init_NT = [1, 1.1, 0.9]
init_thresholds = [1, 1, 1]

l1 = Layer(init_states, init_decays, init_weights, init_NT, init_thresholds)


X = rand(4)
y = rand(3)


y_hat, s = forward_update!(l1, X)
println(s, y_hat)
