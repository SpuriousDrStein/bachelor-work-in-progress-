# version 0-14 : akeen to version 0-12
# - uses a frequency prediction mechanism to determin action prob. distribution

include("..\\global_utility_functions\\activation_functions.jl")
include("..\\global_utility_functions\\loss_functions.jl")
using Distributions
using MLDatasets
using Plots
using OpenAIGym
using StatsBase
using Random

FloatN = Float64

function Base.repeat(a::AbstractArray{T, 2} where T <: Number, n::Integer)
    aa = zeros(n, length(axes(a,1)), length(axes(a,2)))
    for i in axes(aa, 1)
        aa[i, :, :] .= a
    end
    aa
end


# -----------------------------------


mutable struct Layer
    S::AbstractArray{FloatN, 1}
    W::AbstractArray{FloatN, 2}
    H::AbstractArray{FloatN, 1}
    THR::AbstractArray{FloatN, 1}
end

mutable struct Buffer # buffer dimentions = [Temporal, data]
    X_b::AbstractArray{AbstractArray, 1}
    S_b::AbstractArray{AbstractArray, 1} # this will be the input state (S@t-1) since the function get_derivs() computes the forward function seperatly
    W_b::AbstractArray{AbstractArray, 1}
    H_b::AbstractArray{AbstractArray, 1}
end # Buffer.X_b[timestep] = input into layer at timestep

function append_buffer!(B::Buffer, t_input, t_state, t_weight, t_hidden)
    # buffer appends on possition 1
    for i in reverse(eachindex(B.X_b, B.S_b, B.H_b, B.W_b))[1:end-1]
        B.X_b[i], B.S_b[i], B.H_b[i], B.W_b[i] = B.X_b[i-1], B.S_b[i-1], B.H_b[i-1], B.W_b[i-1]
    end

    B.X_b[1], B.S_b[1], B.H_b[1], B.W_b[1] = t_input, t_state, t_hidden, t_weight
end
# SHOWCASE: THE BUFFER WORKS
# b = Buffer([ones(3,3) for _ in 1:5],[ones(3,3) for _ in 1:5],[ones(3,3) for _ in 1:5],[ones(3,3) for _ in 1:5])
# println(b.S_b)
# append_buffer!(b, zeros(3,3), zeros(3,3), zeros(3,3), zeros(3,3))
# append_buffer!(b, rand(3,3), rand(3,3), rand(3,3), rand(3,3))


function activate(x, threshold; base=ℯ)
    (ℯ + threshold * log(base, 1+(x - threshold)^2)) * (abs(x) >= threshold)
end
function d_activate(x, threshold; base=ℯ)
    (threshold * d_log(1+(x - threshold)^2, base=base) * 2*(x - threshold)) * (abs(x) >= threshold)
end


temporal_forward(S, H) = H .* S
temporal_backward(S, H) = H, S
spatial_forward(X, W) = W * X
spatial_backward(X, W) = W, X

function forward(X::AbstractArray,
        S::AbstractArray,
        THRs::AbstractArray,
        H::AbstractArray,
        W::AbstractArray)

    s = temporal_forward(S, H) .+ spatial_forward(X, W')
    # println("S: $S\t->\t$(temporal_forward(S, H'))")
    a = activate.(S, THRs)
    return s, a
end

function forward!(X::AbstractArray, NN::AbstractArray{Layer, 1}, B::AbstractArray{Buffer, 1})
    states = []

    state, act = forward(X, copy(NN[1].S), copy(NN[1].THR), copy(NN[1].H), copy(NN[1].W))
    append_buffer!(B[1], X, copy(NN[1].S), copy(NN[1].W), copy(NN[1].H))
    append!(states, [state])

    # println("layer: 1")
    # println("input      : ", X)
    # println("old states : ", NN[1].S)
    # println("new states : ", state)
    # println("activ      : ", act, "\n")

    if length(NN) > 1
        for i in eachindex(NN, B)[2:end]
            append_buffer!(B[i], act, copy(NN[i].S), copy(NN[i].W), copy(NN[i].H))
            state, act = forward(act, copy(NN[i].S), copy(NN[i].THR), copy(NN[i].H), copy(NN[i].W))
            append!(states, [state])

            # println("layer  : $i")
            # println("input      : ", B[i].X_b[1])
            # println("old states : ", NN[i].S)
            # println("new states : ", state)
            # println("activ      : ", act, "\n")
        end
    end

    update_states!(NN, states) # also does the reset for states above threshold
    return act, states
end

function update_states!(NN::AbstractArray{Layer, 1}, new_states::AbstractArray)
    for l in eachindex(NN, new_states)
        for si in eachindex(NN[l].S)
            if abs(NN[l].S[si]) >= NN[l].THR[si]
                # println(NN[l].S[si], " reset")
                NN[l].S[si] = 0.
            else
                # println(NN[l].S[si], " no reset")
                NN[l].S[si] = new_states[l][si]
            end
        end
    end
end

function reset_states!(NN)
    for l in eachindex(NN)
        NN[l].S .= 0.
    end
end

function reset_buffer!(BN, buffer_length)
    for l in BN
        for i in 1:buffer_length
            l.X_b[i] .= 0.
            l.S_b[i] .= 0.
            l.W_b[i] .= 0.
            l.H_b[i] .= 0.
        end
    end
end


function get_derivs(X::AbstractArray, S::AbstractArray, THRs::AbstractArray, H::AbstractArray, W::AbstractArray)
    dyd_cs = d_activate.(temporal_forward(S, H) .+ spatial_forward(X, W'), THRs)

    dyd_x = spatial_backward(X, W')[1] .* dyd_cs # clamp.(spatial_backward(X, W')[1] .* dyd_cs, d_trunc...)
    dyd_w = spatial_backward(X, W')[2] * dyd_cs' # clamp.(spatial_backward(X, W')[2] * dyd_cs', d_trunc...)

    dyd_ls = temporal_backward(S, H)[1] .* dyd_cs # clamp.(temporal_backward(S, H')[1] .* dyd_cs, d_trunc...)
    dyd_h = temporal_backward(S, H)[2] .* dyd_cs # clamp.(temporal_backward(S, H')[2] * dyd_cs', d_trunc...)

    return dyd_x, dyd_ls, dyd_w, dyd_h
end

function get_temporal_derivs(l::Layer, B::Buffer, buffer_length, dy) # for one layer
    bybhs = []

    _, dyd_ls, _, dyd_h = get_derivs(B.X_b[1], B.S_b[1], l.THR, B.H_b[1], B.W_b[1])
    append!(bybhs, [dy .* dyd_h])

    # println("dy:    $(size(dy))\ndyd_h:  $(size(dyd_h))\ndyd_ls: $(size(dyd_ls))")

    if buffer_length > 1
        for i in 2:buffer_length
            dy = dyd_ls .* dy
            _, dyd_ls, _, dyd_h = get_derivs(B.X_b[i], B.S_b[i], l.THR, B.H_b[i], B.W_b[i])
            append!(bybhs, [dy .* dyd_h])

            # println("dy:    $(size(dy))\ndyd_h:  $(size(dyd_h))\ndyd_ls: $(size(dyd_ls))")
        end
    end
    return bybhs # goes from current_timestep -> buffer_length
end

function get_derivs(NN::AbstractArray{Layer, 1}, B::AbstractArray{Buffer, 1}, buffer_length, dy)
    dydws = [] # a set of derivatives for each weight in NN
    dydhs = []

    dyd_x, _, dyd_w, _ = get_derivs(B[end].X_b[1], B[end].S_b[1], NN[end].THR, B[end].H_b[1], B[end].W_b[1])
    append!(dydws, [dy' .* dyd_w])
    append!(dydhs, [get_temporal_derivs(NN[end], B[end], buffer_length, dy)])

    if any(isnan.(dyd_x))
        throw("derivative is NaN")
    end

    if length(NN) > 1
        for l in reverse(1:length(B)-1)
            dy = dyd_x' * dy
            dyd_x, _, dyd_w, _ = get_derivs(B[l].X_b[1], B[l].S_b[1], NN[l].THR, B[l].H_b[1], B[l].W_b[1])

            if any(isnan.(dyd_x))
                throw("derivative is NaN")
            end

            append!(dydws, [dy' .* dyd_w])
            append!(dydhs, [get_temporal_derivs(NN[l], B[l], buffer_length, dy)])
        end
    end

    return reverse(dydws), reverse(dydhs)
end

function backward!(NN, derivs::Tuple{Array, Array}; lr=0.001, h_accumulation_f=(x)->mean(x), H_trunc=(0.2, 0.5)) # dervis should be [dEdWs, dEdHs]
    for l in eachindex(NN, derivs...)
        NN[l].W .= NN[l].W .- (lr .* derivs[1][l])
        NN[l].H .= clamp.(NN[l].H .- lr .* h_accumulation_f(derivs[2][l]), H_trunc...) # clamp.(NN[l].H - lr .* h_accumulation_f(derivs[2][l]), H_trunc...)
    end
end

function create_layer(input_size, output_size, buffer_length, weight_range, hidden_range;
    init_states=zeros(output_size),
    init_hidden=rand(Uniform(hidden_range...), output_size),
    init_weights=rand(Uniform(weight_range...), input_size, output_size),
    init_thresholds=[1 for _ in 1:output_size])

    x_buf = [zeros(input_size) for _ in 1:buffer_length]
    s_buf = [zeros(output_size) for _ in 1:buffer_length]
    w_buf = [zeros(input_size, output_size) for _ in 1:buffer_length]
    h_buf = [zeros(output_size) for _ in 1:buffer_length]

    Layer(init_states, init_weights, init_hidden, init_thresholds), Buffer(x_buf, s_buf, w_buf, h_buf)
end

function cross_entropy_learning!(NN, batch, buffer_length; top_percentile=0.5, hidden_range=(0.5,0.99))
    rew = [b[1] for b in batch]
    best = [b[2] for b in batch if b[1] > (maximum(rew)*(1-top_percentile))]
    best_rewards = rew[rew .> (maximum(rew)*(1-top_percentile))]

    println("best $(length(best_rewards)) = $best_rewards")
    for b in shuffle(best)
        for (i, (s, a)) in enumerate(b)
            reset_states!(NN)
            reset_buffer!(BN, buffer_length)
            final_out = [0., 0.]
            for rep in 1:buffer_length
                out, _ = forward!(Array(s), NN, BN)
                final_out .+= out
            end

            println(final_out)
            sleep(0.7)

            # if i % buffer_length == 0 # from version 0-13
            pa = softmax(final_out)
            loss = crossentropy(pa, a)
            dE = d_crossentropy(pa, a)
            dEdAct = d_softmax(final_out) * dE
            derivs = get_derivs(NN, BN, buffer_length, dEdAct)

            # println("pred: $pa\t:::\tdE: $dE\t:::\tout: $out")
            # println(derivs[1])

            backward!(NN, derivs, lr=lr, H_trunc=hidden_range)

            # sleep(0.3)
            # end
        end
    end
    return length(rew)
end


# cross entropy methode
# describing:
# - model free
# - policy based
# - on policy

env = GymEnv(:CartPole, :v0)
# env = GymEnv(:Acrobot, :v1)

action_index = [i for i in 1:length(env.actions)]
action_space = one(action_index*action_index')


input_size = length(env.state)^2
output_size = length(env.actions)
buffer_length = 7

# h_truncate = (0.85, 0.999)
weight_init_range = (-0.5, 0.5)
hidden_init_range = (0.99, 0.999)
threshold = 0.5
hidden_sizes = (3,3,3,3,3)

lr=0.1
early_reward_accentuation = 1.1
percentile = 0.004
training_iterations = 300
batch_size = 40
max_simulation_time = 100



# NETWORK
hs = (hidden_sizes..., output_size)
l, b = create_layer(input_size, hs[1], buffer_length, weight_init_range, hidden_init_range, init_thresholds=[threshold for _ in 1:hs[1]])
NN = [l]; BN = [b]; if hidden_sizes != ()
    for i in eachindex(hs[1:end-1])
        l, b = create_layer(hs[i], hs[i+1], buffer_length, weight_init_range, hidden_init_range, init_thresholds=[threshold for _ in 1:hs[i+1]])
        append!(NN, [l]); append!(BN, [b])
    end
end

delta_w, delta_h = [copy(l.W) for l in NN], [copy(l.H) for l in NN]

num_iterations_over_min = [0 for _ in 1:training_iterations]; reward_metric = [0. for _ in 1:training_iterations]; for k in 1:training_iterations
    batch = []

    for i in 1:batch_size
        s = Array(reset!(env))
        s = reshape(((s * s') .+ 1.), length(s)^2)

        episode_steps = []
        reward_counter = 0

        for j in 1:max_simulation_time
            reset_states!(NN)
            reset_buffer!(BN, buffer_length)
            final_out = [0. for _ in 1:output_size]
            for rep in 1:buffer_length
                out, _ = forward!(Array(s), NN, BN)
                final_out .+= out
            end

            a_prob = Weights(softmax(final_out))
            a = sample(action_index, a_prob)

            # println("out: $out\nstates:\n$(NN[end].S)\n$(NN[end-1].S)\n$(NN[1].S)\nweights:\n$(NN[end].W)\n$(NN[end-1].W)\nhidden:\n$(NN[end].H)")
            # println(["$sb\n" for sb in b2.S_b]...)
            # println(l2.W)
            # sleep(0.3)

            # if NN[end].S > NN[end].THR
            #     println("out = $out\naction = $a\nstates = $(NN[end].S)\n$(NN[end-1].S)\n$(NN[1].S)\nhidden = $(NN[end].H)\n$(NN[end-1].H)\n$(NN[1].H)")
            #     sleep(1)
            # end

            # println("input      : ", BN[end].X_b[1])
            # println("old states : ", NN[end].S)
            # println("activ      : ", out, "\n")

            append!(episode_steps, [Pair(Array(s), action_space[a, :])])
            r, s = step!(env, env.actions[a])
            s = reshape(((Array(s) * Array(s')) .+ 1.), length(s)^2)

            reward_counter = reward_counter * early_reward_accentuation + r

            # render(env)

            if env.done
            append!(batch, [Pair(reward_counter, episode_steps)])
                break
            end
        end
    end
    println("batch $k finished")
    println("avg_reward = $(mean([b[1] for b in batch]))")
    reward_metric[k] = mean([b[1] for b in batch])
    num_iterations_over_min[k] = cross_entropy_learning!(NN, batch, buffer_length; top_percentile=percentile, hidden_range=hidden_init_range)
end

delta_w, delta_h = [delta_w[i] .- NN[i].W for i in eachindex(NN, delta_w)], [delta_h[i] .- NN[i].H for i in eachindex(NN, delta_h)]
println(delta_w)
println(delta_h)


plot(reward_metric./maximum(reward_metric), ylabel="maximum reward", xlabel="batch")
# plot(num_iterations_over_min, ylabel="reward over $minimum_acceptable_reward", xlabel="batch")
