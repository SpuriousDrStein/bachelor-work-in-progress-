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

println("size Float16 = ", sizeof(Float16(1.)))
println("size Float32 = ", sizeof(Float32(1.)))
println("size Float64 = ", sizeof(Float64(1.)))
println("size BigFloat = ", sizeof(BigFloat(1.)))

FloatN = Float64

function Base.repeat(a::AbstractArray{T, 2} where T <: Number, n::Integer)
    aa = zeros(n, length(axes(a,1)), length(axes(a,2)))
    for i in axes(aa, 1)
        aa[i, :, :] .= a
    end
    aa
end

mutable struct Layer
    S::AbstractArray{FloatN, 1}
    H::AbstractArray{FloatN, 2}
    ψ::AbstractArray{FloatN, 1}
    𝛆::AbstractArray{FloatN, 1}
    ϑ::AbstractArray{FloatN, 2}
    S_buffer::AbstractArray{FloatN, 2} # buffers have [buffer_length, S_size]   ## enteries are also saved from top to bottom
    H_buffer::AbstractArray{FloatN, 3}
    ψ_buffer::AbstractArray{FloatN, 2}
end

function test_activation(l::Layer; ɣ=0.8)
     l.S ./ (1 .+ (l.S - l.𝛆).^2 .* ɣ) .* (l.S .>= l.𝛆)
end
function d_test_activation(l::Layer; ɣ=0.8)
    ((-2 .* (l.S .- l.𝛆) .* ɣ) ./ (1 .+ (l.S .- l.𝛆).^2) .* ɣ) .* (l.S .>= l.𝛆)
end

function forward_update!(l::Layer, x)
    push_buffer!(l)
    l.S = (l.H * l.S .+ l.ϑ' * x) .* l.ψ
    act = test_activation(l)
    # println("layer state: ", l.S)
    l.S .*= (l.S .< l.𝛆)
    return act
end

function network_forward!(N::Array{Layer, 1}, X)
    a = forward_update!(N[1], X)
    # println("avg activ: ", sum(a)/length(N[1].S))
    for l in N[2:end]
        a = forward_update!(l, a)
        # println("avg activ: ", sum(a)/length(l.S))
    end
    return a
end


function spatial_backward_pass(l::Layer, dEdA)
    dEdS = d_test_activation(l) .* dEdA
    return (l.ϑ .* l.ψ') * dEdS
end
function spatial_backward_update!(l::Layer, x, dEda; lr=0.003)
    dadϑ = x * l.ψ'
    l.ϑ .-= dadϑ .* (lr .* dEda)
end

function temporal_backward_update!(l::Layer, dEdA; lr=0.001)
    dEdSt = d_test_activation(l) .* dEdA
    dStdHt = (l.ψ * l.S') # S(t-1) .* ψ(t-1)
    dEdHt = dEdSt .* dStdHt
    del_H = dEdHt

    dStdSlt = (l.H .* l.ψ)
    dEdSlt = dEdSt' * dStdSlt
    for t in axes(l.S_buffer, 1)
        dSltdHlt = (l.ψ_buffer[t, :] * l.S_buffer[t, :]') # S(t-1) .* ψ(t-1)
        dEdHlt = dEdSlt .* dSltdHlt

        dSltdSllt = (l.H_buffer[t, :, :] .* l.ψ_buffer[t, :])
        dEdSlt = dEdSlt * dSltdSllt

        del_H += dEdHlt
    end
    l.H .-= (del_H .* lr)
end

function network_backward!(N::Array{Layer, 1}, dEdy; final_act_df=d_softmax)
    dEdS = d_test_activation(N[end]) .* dEdy
    spatial_backward_update!(N[end], test_activation(N[end-1]), dEdy)
    temporal_backward_update!(N[end], dEdS)

    dEdXl = spatial_backward_pass(N[end], dEdy')
    if length(N) > 2
        for l in reverse((1:length(N))[2:end-1])

            spatial_backward_update!(N[l], test_activation(N[l-1]), dEdXl')
            temporal_backward_update!(N[l], dEdXl)
            dEdXl = spatial_backward_pass(N[l], dEdXl)
        end
    end

    temporal_backward_update!(N[1], dEdXl)
    dEdXl = spatial_backward_pass(N[1], dEdXl)
    return dEdXl
end

function push_buffer!(l::Layer)
    for bi in reverse(axes(l.ψ_buffer, 1))[1:end-1]
        l.S_buffer[bi,:] .= l.S_buffer[bi-1,:]
        l.H_buffer[bi,:,:] .= l.H_buffer[bi-1,:,:]
        l.ψ_buffer[bi,:] .= l.ψ_buffer[bi-1,:]
    end
    l.S_buffer[1,:] .= l.S
    l.H_buffer[1,:,:] .= l.H
    l.ψ_buffer[1,:] .= l.ψ
end


# NETWORK INIT
function create_layer(input_size, output_size, buffer_length;
    init_states=zeros(output_size),
    init_hidden=rand(Uniform(-0.28, 0.28), output_size, output_size),
    init_weights=rand(Uniform(0.7, 1), input_size, output_size),
    init_NT=[1 for _ in 1:output_size],
    init_thresholds=[1 for _ in 1:output_size])
    s_buf = reshape(repeat(init_states', buffer_length), (buffer_length, length(init_states)))
    NT_buf= reshape(repeat(init_NT', buffer_length), (buffer_length, length(init_NT)))
    h_buf = repeat(init_hidden, buffer_length)
    Layer(init_states,
        init_hidden,
        init_NT,
        init_thresholds,
        init_weights,
        s_buf,
        h_buf,
        NT_buf)
end


#
# # DATA
# train_x, train_y = MNIST.traindata()
# test_x,  test_y  = MNIST.testdata()
#
# train_x_f = reshape(train_x, (28*28, 60000))
# test_x_f = reshape(test_x, (28*28, 10000))
#
# train_y_onehot = zeros(10, length(train_y))
# for (i, a) in enumerate(train_y)
#     train_y_onehot[a+1, i] = 1.
# end
# test_y_onehot = zeros(10, length(test_y))
# for (i, a) in enumerate(test_y)
#     test_y_onehot[a+1, i] = 1.
# end
#
#
# # MAIN
# function main_loop!(iterations, network, X, y)
#     # total_activations_final_layer = [0. for _ in 1:length(network[end].S)]
#     loss_metric = [99999. for _ in 1:iterations]
#     for i in 1:iterations
#         r = rand(1:length(X)-input_size-1)
#         x_ = X[r:r+input_size-1]
#         y_ = X[r+input_size]
#         println("---- itration: $i")
#
#         out = network_forward!(network, x_)
#         y_hat = sigmoid.(out)
#         loss = SD(y_hat, y_)
#
#         # println("end state:     ", s)
#         # println("end acts:      ", out)
#         # println("labels:        ", y_)
#         # println("H sum:         ", sum(NN[end].H, dims=2))
#         # println("W sum:         ", sum(NN[end].ϑ, dims=1))
#
#         dEdCE = d_SD(y_hat, y_)
#         dCEdSoft = d_sigmoid.(out)
#         dEdSoft = dEdCE' * dCEdSoft
#         network_backward!(network, x_, dEdSoft)
#
#         # println("sum: ", sum(l1.ϑ))
#         # println("max: ", maximum(l1.ϑ))
#         # println("min: ", minimum(l1.ϑ))
#         # println("avg: ", sum(l1.ϑ)/length(l1.ϑ))
#         println("|| loss:       ", loss)
#         loss_metric[i] = loss
#         # total_activations_final_layer .+= abs.(s)
#         println("\n")
#     end
#     loss_metric
#     # println("total activations (last layer): ", total_activations_final_layer)
# end
#
# # loss_over_time = main_loop!(1000, NN, [sin(i) for i in 1:10000000], [])

using Plots
using OpenAIGym
using StatsBase
using Random

# describing:
# - model free
# - policy based
# - on policy
lr=0.003
batch_size = 30
top_selection = 5
simulation_time = 60
input_size = 4
output_size = 2
buffer_length = 1
training_iterations = 300

l1 = create_layer(input_size, 10, buffer_length)
l2 = create_layer(10, 7, buffer_length)
l3 = create_layer(7, output_size, buffer_length)
NN = [l1, l2, l3]


env = GymEnv(:CartPole, :v0)


# cross entropy methode

reward_metric = [0. for _ in 1:training_iterations]
action_index = [i for i in 1:length(env.actions)]
action_space = one(action_index*action_index')
for k in 1:training_iterations
    batch = []
    for i in 1:batch_size
        s = reset!(env)
        episode_steps = []

        for j in 1:simulation_time
            out = network_forward!(NN, Array(s))
            a_prob = Weights(softmax(out))
            a = sample(action_index, a_prob)

            append!(episode_steps, [Pair(Array(s), action_space[a, :])])
            r, s = step!(env, env.actions[a])

            # println("out:  ", out)

            # render(env)

            if env.done
                append!(batch, [Pair(env.total_reward, episode_steps)])
                break
            end
        end
    end
    println("batch $k finished")
    println("avg_reward = $(mean([b[1] for b in batch]))")
    reward_metric[k] += mean([b[1] for b in batch])
    RL_train_net!(NN, batch)
end

function RL_train_net!(NN, batch)
    rew, best = [b[1] for b in sort(batch)][end-top_selection:end], [b[2] for b in sort(batch)][end-top_selection:end]
    println("best selection: $rew")
    for b in best
        for (s, a) in b
            out = network_forward!(NN, Array(s))
            pa = softmax(out)
            loss = crossentropy(pa, a)
            dE = d_crossentropy(pa, a)
            dEdSoft = dE' * d_softmax(out)
            network_backward!(NN, dEdSoft)
        end
    end
end
