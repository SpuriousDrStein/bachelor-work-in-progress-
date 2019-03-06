# this version encapsulates
# a learning rule for a spike NN
# that uses fixed thresholds and interconected layers via "neurotransmitter"
# 𝛆 = threhsold
# ϑ = weights
# ψ = neur-transmitter

include("..\\global_utility_functions\\activation_functions.jl")
include("..\\global_utility_functions\\loss_functions.jl")
using Distributions
using MLDatasets

FloatN = Float64

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
    l.S .= abs.(l.S) .- test_activation(l)
    l.S = (l.S .* l.decay .+ l.ϑ' * x) .* l.ψ
    return test_activation(l), l.S
end

function backward_pass(l::Layer, dEda)
    return ((l.ϑ .* l.ψ') .* d_test_activation(l)') * dEda
end

function backward_update!(l::Layer, x, dEda; lr=0.003)
    dadϑ = x * l.ψ'
    l.ϑ -= dadϑ .* (lr .* dEda)'
end


function network_backward!(N::Array{Layer, 1}, X, dEdy; final_act_df=d_softmax)
    dEdXl = backward_pass(N[end], dEdy')
    backward_update!(N[end], test_activation(N[end-1]), dEdy')

    if length(N) > 2
        for l in reverse((1:length(N))[2:end-1])
            dEdXl = backward_pass(N[l], dEdXl)
            backward_update!(N[l], test_activation(N[l-1]), dEdXl')
        end
    end

    dEdXl = backward_pass(N[1], dEdXl)
    backward_update!(N[1], X, dEdXl')
    return dEdXl
end

function network_forward!(N::Array{Layer, 1}, X)
    a, s = forward_update!(N[1], X)
    for l in N[2:end]
        a, s = forward_update!(l, s)
    end
    return a, s
end


function create_layer(input_size, output_size;
    init_decays=[0.99 for _ in 1:output_size],
    init_states=zeros(output_size),
    init_weights=rand(Uniform(-0.05, 0.05),
    input_size, output_size),
    init_NT=[1 for _ in 1:output_size],
    init_thresholds=[1 for _ in 1:output_size])

    Layer(init_states, init_decays, init_weights, init_NT, init_thresholds)
end


lr=0.003
iterations = 500
input_size = 28*28
output_size = 10


l1 = create_layer(input_size, 400)
l2 = create_layer(400, 300)
l3 = create_layer(300, 200)
l4 = create_layer(200, 100)
l5 = create_layer(100, 50)
l6 = create_layer(50, output_size)
NN = [l1,l2,l3,l4,l5,l6]


train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

train_x_f = reshape(train_x, (28*28, 60000))
test_x_f = reshape(test_x, (28*28, 10000))

train_y_onehot = zeros(10, length(train_y))
for (i, a) in enumerate(train_y)
    train_y_onehot[a+1, i] = 1.
end
test_y_onehot = zeros(10, length(test_y))
for (i, a) in enumerate(test_y)
    test_y_onehot[a+1, i] = 1.
end

function main_loop!(iterations, network, X, y)
    for i in 1:iterations
        r = rand(1:size(X)[2])
        x_ = X[:, r]
        y_ = y[:, r]


        out, s = network_forward!(network, x_)
        y_hat = softmax(out)
        loss = crossentropy(y_hat, y_)

        dEdCE = d_crossentropy(y_hat, y_)
        dCEdSoft = d_softmax(out)
        dEdSoft = dEdCE' * dCEdSoft
        network_backward!(network, x_, dEdSoft)

        # println("sum: ", sum(l1.ϑ))
        # println("max: ", maximum(l1.ϑ))
        # println("min: ", minimum(l1.ϑ))
        # println("avg: ", sum(l1.ϑ)/length(l1.ϑ))
        println("|| loss:  \t", loss)
        println("states     : ", NN[end].S)
        println("acivations : ", test_activation(NN[end]))
        println("weights_sum: ", sum(NN[end].ϑ, dims=1))
        println("\n")
    end
end

main_loop!(999999, NN, train_x_f, train_y_onehot)
