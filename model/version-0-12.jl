# version akeen to 0-11
# more general function strucutre
# replaces v0-11

include("..\\global_utility_functions\\activation_functions.jl")
include("..\\global_utility_functions\\loss_functions.jl")
using Distributions
using MLDatasets

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
    H::AbstractArray{FloatN, 2}
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
        B.X_b[i], B.S_b[i], B.H_b[i], B.W_b[i] = B.X_b[i+1], B.S_b[i+1], B.H_b[i+1], B.W_b[i+1]
    end

    B.X_b[1], B.S_b[1], B.H_b[1], B.W_b[1] = t_input, t_state, t_weight, t_hidden
end


function activate(x, threshold; ɣ=0.8)
    (threshold + x * ((x - threshold)^2) * ɣ) * (x >= threshold)
end
function d_activate(x, threshold; ɣ=0.8)
    (((x - threshold)^2 * ɣ) + (2 * (x - threshold) * x * ɣ)) * (x >= threshold)
end

temporal_forward(S, H) = H * S
temporal_backward(S, H) = H, S
spatial_forward(X, W) = W * X
spatial_backward(X, W) = W, X

function forward(X::AbstractArray,
        S::AbstractArray,
        THRs::AbstractArray,
        H::AbstractArray,
        W::AbstractArray)

    s = temporal_forward(S, H) .+ spatial_forward(X, W)
    a = activate.(s, THRs)
    return s, a
end

function forward!(X::AbstractArray, NN::AbstractArray{Layer, 1}, B::AbstractArray{Buffer, 1})
    states = []

    state, act = forward(X, NN[1].S, NN[1].THR, NN[1].H, NN[1].W)
    append_buffer!(B[1], X, NN[1].S, NN[1].W, NN[1].H)
    append!(states, [state])

    if length(NN) > 1
        for i in eachindex(NN[2:end], B[2:end])
            append_buffer(B[i], act, NN[i].S, NN[i].W, NN[i].H)
            state, act = forward(act, NN[l].S, NN[l].THR, NN[l].H, NN[l].W)
            append!(states, [state])
        end
    end

    return act, states
end

function update_states!(NN::AbstractArray{Layer, 1}, new_states::AbstractArray{AbstractArray})
    for l in eachindex(NN, new_states)
        NN[l].S = new_states[l]
    end
end

function get_derivs(X::AbstractArray,
        S::AbstractArray,
        THRs::AbstractArray,
        H::AbstractArray,
        W::AbstractArray)

    dyd_cs = d_activate.(temporal_forward(S, H') .+ spatial_forward(X, W'), THRs)

    dyd_x = spatial_backward(X, W)[1] .* dyd_cs'
    dyd_w = spatial_backward(X, W)[2] * dyd_cs'

    dyd_ls = temporal_backward(S, H)[1] .* dyd_cs'
    dyd_h = temporal_backward(S, H)[2] * dyd_cs'

    return dyd_x, dyd_ls, dyd_w, dyd_h
end

function get_temporal_derivs!(l::Layer, B::Buffer, buffer_length, dy) # this is for one layer
    bybhs = []

    _, dyd_ls, _, dyd_h = get_derivs(B.X_b[1], B.S_b[1], l.THR, B.H_b[1], B.W_b[1])
    append!(bybhs, [dy' .* dyd_h])

    if buffer_length > 1
        for i in 2:buffer_length
            dy = dyd_ls * dy
            _, dyd_ls, _, dyd_h, _ = get_derivs(B.X_b[i], B.S_b[i], l.THR, B.H_b[i], B.W_b[i])
            append!(bybhs, [dy' .* dyd_h])
        end
    end
    return bybhs # goes from current_timestep -> buffer_length
end

function get_derivs!(NN::AbstractArray{Layer, 1}, B::AbstractArray{Buffer, 1}, buffer_length, dy)
    dydws = [] # a set of derivatives for each weight in NN
    dydhs = []

    dyd_x, _, dyd_w, _ = get_derivs(B[end].X_b[1], B[end].S_b[1], NN[end].THR, B[end].H_b[1], B[end].W_b[1])
    append!(dydws, [dy' .* dyd_w])
    append!(dydhs, [get_temporal_derivs!(NN[end], B[end], buffer_length, dy)])

    if length(NN) > 1
        for l in reverse(1:buffer_length-1)
            dy = dyd_x * dy
            dyd_x, _, dyd_w, _ = get_derivs(B[l].X_b[1], B[l].S_b[1], NN[l].THR, B[l].H_b[1], B[l].W_b[1])
            append!(dydws, [dy' .* dyd_w])
            append!(dydhs, [get_temporal_derivs!(NN[l], B[l], buffer_length, dy)])
        end
    end

    return reverse(dydws), reverse(dydhs)
end


function create_layer(input_size, output_size, buffer_length;
    init_states=zeros(output_size),
    init_hidden=rand(Uniform(0.95, 1), output_size, output_size),
    init_weights=rand(Uniform(-0.05, 0.05), input_size, output_size),
    init_NT=[1 for _ in 1:output_size],
    init_thresholds=[1 for _ in 1:output_size])
    x_buf = [zeros(input_size) for _ in 1:buffer_length]
    s_buf = [init_states for _ in 1:buffer_length]
    w_buf = [zeros(input_size, output_size) for _ in 1:buffer_length]
    h_buf = [zeros(output_size, output_size) for _ in 1:buffer_length]

    Layer(init_states, init_weights, init_hidden, init_thresholds), Buffer(x_buf, s_buf, w_buf, h_buf)
end


# HPs
buffer_size = 5



l1, b1 = create_layer(2, 10, buffer_size)
l2, b2 = create_layer(10, 15, buffer_size)
l3, b3 = create_layer(15, 10, buffer_size)
l4, b4 = create_layer(10, 2, buffer_size)

NN = [l1,l2,l3,l4]
BN = [b1,b2,b3,b4]

dydws, dydhs = get_derivs(NN, BN, buffer_size, ones(2))
