using Flux

# STRUCTURE
mutable struct edge
    VAL::AbstractFloat

    thr::AbstractFloat
    res_coef::AbstractFloat # recovery speed
    freq_coef::AbstractFloat # frequency
end
mutable struct node
    VAL::AbstractFloat

    thr::AbstractFloat
    res_coef::AbstractFloat
    freq_coef::AbstractFloat
end

conectum = Array{edge, 2}
mask = Array{Bool, 2}
neuron_layer = Array{node, 1}

mutable struct network
    neuron_layers::Array{neuron_layer, 1}
    conectums::Array{conectum, 1}
    masks::Array{mask, 1}
end

# UTILITY FUNCTIONS
function Base.getindex(n::node, i::Integer); return n[i]; end
raw(e::edge) = [copy(e.thr), copy(e.res_coef), copy(e.freq_coef)]
raw(n::node) = [copy(n.thr), copy(n.res_coef), copy(n.freq_coef)]
function Base.copy(x::neuron_layer); return collect(Iterators.flatten([raw(l) for l in x])); end
function Base.copy(x::conectum); return collect(Iterators.flatten([raw(l) for l in x])); end

# OTHER ///UC (under construction)
function update!(nl::neuron_layer, x::AbstractArray, d_ut)
    out = [0. for _ in 1:length(nl)]
    for ni in eachindex(nl)
        nl[ni].VAL += x[ni]
        nl[ni].VAL += d_ut * nl[ni].res_coef
        nl[ni].VAL += d_ut * cos(nl[ni].freq_coef)

        nl[ni].res_coef += d_ut * (nl[ni].VAL - nl[ni].thr) # negative if not fiering -> making the resistance coeff negative -> increase probabilty of fiering
        nl[ni].freq_coef = (nl[ni].freq_coef + d_ut) * (nl[ni].freq_coef < 2π)

        out[ni] = nl[ni].VAL * (nl[ni].VAL >= nl[ni].thr)
    end
    return out
end
function update!(con::conectum, m::mask, x::AbstractArray, d_ut)
    out = [0. for _ in axes(con, 2)]
    for xi in eachindex(x)
        for (ai, syn) in enumerate(con[xi, :])
            if m[xi, ai] # if is true in mask
                syn.VAL += x[xi]
                syn.VAL += d_ut * syn.res_coef
                syn.VAL += d_ut * cos(syn.freq_coef)

                syn.res_coef += d_ut * (syn.VAL - syn.thr)
                syn.freq_coef = (syn.freq_coef + d_ut) * (syn.freq_coef < 2π)

                out[ai] += copy(syn.VAL) * (syn.VAL >= syn.thr)
            end
        end
    end
    return out
end

function feed_forward!(net::network, x::AbstractArray, d_ut)
    o = update!(net.neuron_layers[1], x, d_ut)
    o = update!(net.conectums[1], net.masks[1], o, d_ut)
    for i in 2:length(net.neuron_layers)-1
        o = update!(net.neuron_layers[i], o, d_ut)
        o = update!(net.conectums[i], net.masks[i], o, d_ut)
    end
    return update!(net.neuron_layers[end], o, d_ut)
end

function construct_network(params::AbstractVector, hidden_sizes::AbstractVector; number_of_edge_params=4, number_of_node_params=4)::network
    ret_net = network([], [], [])
    it = 1

    for i in length(hidden_sizes)-1
        neuron_param_ind   = it + hidden_sizes[i] * number_of_node_params
        conectum_param_ind = neuron_param_ind + hidden_sizes[i] * hidden_sizes[i+1] * number_of_edge_params
        mask_param_ind     = conectum_param_ind + hidden_sizes[i] * hidden_sizes[i+1]

        append!(ret_net.neuron_layers, [neuron_layer([node(params[j:j+number_of_node_params-1]...) for j in it:number_of_node_params:neuron_param_ind-1])])
        append!(ret_net.conectums, [conectum(reshape([edge(params[j:j+number_of_edge_params-1]...) for j in neuron_param_ind:number_of_edge_params:conectum_param_ind-1], (hidden_sizes[i], hidden_sizes[i+1])))])
        append!(ret_net.masks, [mask(reshape(clamp.(round.(params[conectum_param_ind:mask_param_ind-1]), 0, 1) .|> Bool, (hidden_sizes[i], hidden_sizes[i+1])))])

        it += mask_param_ind - neuron_param_ind
    end

    append!(neuron_layers, [neuron_layer([node(params[j:j+number_of_node_params-1]...) for j in mask_param_ind:number_of_node_params:length(params)])])

    # INFO: order - neurons, conectum, mask, neurons, conectum, mask, ..., mask, neurons
    return ret_net
end

function deconstruct_network(net::network)
    ret_par = []
    for l in eachindex(net.conectums, net.masks)
        append!(ret_par, copy(net.neuron_layers[l]))
        append!(ret_par, copy(net.conectums[l]))
        append!(ret_par, copy(net.masks[l]))
    end
    append!(ret_par, copy(net.neuron_layers[end]))

    return collect(Iterators.flatten(ret_par))
end

mutable struct RNN_Runner_agent
    buffer_size::Integer
    input_buffer::AbstractArray{AbstractFloat}
    hidden_buffer::AbstractArray{AbstractFloat}
    xh_w_buffer::AbstractArray{AbstractFloat}
    xh_b_buffer::AbstractArray{AbstractFloat}
    hh_w_buffer::AbstractArray{AbstractFloat}
    hh_b_buffer::AbstractArray{AbstractFloat}
    hy_w_buffer::AbstractArray{AbstractFloat}
    hy_b_buffer::AbstractArray{AbstractFloat}
end

function run_agent!(x::AbstractArray, runner::RNN_Runner_agent)
    h_t = x * xh_w_buffer[end] .+ xh_b_buffer[end]
    h_t1 = h_t * hh_w_buffer[end] .+ hh_b_buffer[end] # x=512 -> 512xh_size -> h_size -> h_sizexh_size -> h_size

    # dxhw_dy = hidden_buffer[end]
    # dhhw_dy = dht_dy * hidden_buffer[end-1]
    # dxhw_dw = dht_dy * dht1_dht * input_buffer[end]

    # PUT IN LOOP

    # append input & hidden
    if length(runner.input_buffer) <= unner.buffer_size
        append!(runner.hidden_buffer, h_t)
        append!(runner.xh_w_buffer, )
    else
        # shift up
    end
end



# TEST
using Main.IMPORTS
using Main.FUNCTIONS

input_size = length(X)
hidden_sizes = [15, 10, 5, 2]
no_params_per_lay = hidden_sizes .+ hidden_sizes .^ 2 .* 2
d_UT = 0.01 |> AbstractFloat # update time
d_RT = 0.5 |> AbstractFloat # reaction time
d_GT = 1 |> AbstractFloat # growth time


X = rand(20)

parrs = rand(sum(no_params_per_lay))
net = construct_network(parrs, hidden_sizes)
feed_forward!(net, X, d_UT)

pars = deconstruct_network(net)
pars .|> round
construct_network(pars, hidden_sizes)
