using Distributions

mutable struct Layer
    weights::AbstractArray
    biases::Vector
    thresholds::Vector
    act_frequencies::Vector
    membranes::Vector
end


# function frq_truncate_z(z::Vector, thresholds::Vector, r_max::Integer)
#     min.(z ./ thresholds .* (z .> 0), r_max)
# end
#
# function frq_check_thr(input::Vector, threshold::Vector)
#     (input .- threshold) .>= 0
# end

function frq_update_z(input::Vector, l::Layer)
    l.thresholds .* dropdims(sum(l.weights .* input .+ l.biases', dims=1), dims=1)
end

function frq_update_mem(z::Vector, last_mem::Vector, thresholds::Vector)
    last_mem .+ z .- (thresholds .* (last_mem .- thresholds .>= 0))
end

function frq_simulate(layer::Layer, input::Vector, T::Integer) # !t denotes temprary mutation
    s = frq_update_z(input .* T, layer)
    s = floor.(s ./ layer.thresholds)
    s = s .* (s .> 0)
    min.(s, T)
end

function frq_update!(layer::Layer, input::Vector, T::Integer)
    # yt = frq_truncate_z(yw, layer.thresholds, T)
    z = frq_update_z(input, layer)
    layer.membranes = frq_update_mem(z, layer.membranes, layer.thresholds)

    layer.act_frequencies = frq_simulate(layer, input, T)
end

# DERIVATIVES
function d_frq_update_z(input::Vector, l::Layer)
    dzdw = repeat(l.thresholds', size(input)[1]) .* input
    dzdb = l.thresholds'
    dzdx = l.thresholds' .* l.weights # NxM matrix
    dzdx, dzdw, dzdb
end

function d_frq_simulate(l::Layer, x::Vector, T::Integer)
end

function d_frq_update_mem(z::Vector, membranes::Vector, thesholds::Vector)
end

function d_frq_update!(layer::Layer, input::Vector, T::Integer; lr=0.003)
    dzdx, dzdw, dzdb = d_frq_update_z(input, layer)
    # println(size(dzdx))
    # println(size(dzdw))
    # println(size(dzdb))
    
end




function create_dense_layer(input_size, output_size; distribution_w=Uniform(-0.05, 0.05), distribution_b=Uniform(0, 0.01), thresholds=[1 for _ in 1:output_size])
    w = rand(distribution_w, input_size, output_size)
    b = rand(distribution_b, output_size)
    Layer(w, b, thresholds, zeros(output_size), zeros(output_size))
end
