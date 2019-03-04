using Distributions

mutable struct Layer
    weights::AbstractArray
    biases::Vector
    thresholds::Vector
    act_frequencies::Vector
    # membranes::Vector
end


# function frq_truncate_z(z::Vector, thresholds::Vector, r_max::Integer)
#     min.(z ./ thresholds .* (z .> 0), r_max)
# end
#
# function frq_check_thr(input::Vector, threshold::Vector)
#     (input .- threshold) .>= 0
# end


# function frq_update_mem(z::Vector, last_mem::Vector, thresholds::Vector)
#     last_mem .+ z .- (thresholds .* (last_mem .- thresholds .>= 0))
# end

function forward(l::Layer, input::Vector, T::Integer) # !t denotes temprary mutation
    s = l.thresholds .* (l.weights' * (input .* T) .+ l.biases)
    s = floor.(s ./ l.thresholds)
    s = s .* (s .>= 0)
    min.(s, T)
    s
end

# DERIVATIVES
# function d_frq_update_z(input::Vector, l::Layer)
#     dydw = repeat(l.thresholds', size(input)[1]) .* input
#     dydb = l.thresholds
#     dydx = l.thresholds' .* l.weights # NxM matrix
#     dydx, dydw, dydb
# end

function d_forward(l::Layer, x::Vector, T::Integer)
    z = frq_update_z(x .* T, l)
    a = floor.(z ./ l.thresholds)

    dadw = (repeat(l.thresholds', size(x)[1]) .* x .* T) ./ (l.thresholds.^2)'
    dadb = l.thresholds
    dadx = l.weights ./ l.thresholds'

    dadw = dadw .* (z .>= 0)'
    dadb = dadb .* (z .>= 0)
    dadx = dadx .* (z .>= 0)'

    dadx, dadw, dadb
end

function d_frq_update!(layer::Layer, input::Vector, d, T::Integer; lr=0.0003)
    dx, dw, db = d_forward(layer, input, T)

    layer.weights += lr .* (d' .* dw)
    layer.biases += lr .* (d .* db)

    dx
end


function create_dense_layer(input_size, output_size; distribution_w=Uniform(-0.05, 0.05), distribution_b=Uniform(0, 0.01), thresholds=[1 for _ in 1:output_size])
    w = rand(distribution_w, input_size, output_size)
    b = rand(distribution_b, output_size)
    Layer(w, b, thresholds, zeros(output_size))
end
