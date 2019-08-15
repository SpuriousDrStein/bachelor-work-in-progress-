
# STRUCTURE
mutable struct edge
    VAL::Float16

    thr::Float16
    res_coef::Float16 # recovery speed
    freq_coef::Float16 # frequency
end
mutable struct node
    VAL::Float16

    thr::Float16
    res_coef::Float16
    phase::Float16
end

conectum = AbstractArray{Union{edge, Missing}, 2}
mask = AbstractArray{Bool, 2}
neuron_layer = AbstractVector{node}
network = AbstractVector{Union{neuron_layer, conectum, mask}}

# UTILITY FUNCTIONS
function Base.getindex(n::node, i::Integer); return n[i]; end
raw(e::edge) = [copy(e.thr), copy(e.res_coef), copy(freq_coef)]
raw(n::node) = [copy(n.thr), copy(n.res_coef), copy(phase)]
function Base.copy(x::neuron_layer); return collect(Iterators.flatten([raw(l) for l in x])); end
function Base.copy(x::conectum); return collect(Iterators.flatten([raw(l) for l in x])); end




# OTHER ///UC (under construction)
function feed_forward!(nl1::neuron_layer, nl2::neuron_layer, em::conectum, x::AbstractVector, d_ut::Float16)
    for n1 in eachindex(nl1, x)
        nl1[n1].VAL += x[n1]
        nl1[n1].VAL += d_ut * cos(nl1[n1].phase)
        nl1[n1].res_coef += d_ut * (nl1[n1].VAL - nl1[n1].thr) # negative if not fiering -> making the resistance coeff negative -> increase probabilty of fiering

        if nl1[n1].VAL >= nl1[n1].thr
            for axon in em[n1, :]
                axon.VAL += nl1[n1].thr
                axon.VAL += d_ut * axon.res_coef
                axon.VAL += d_ut * cos(axon.freq_coef)
                axon.res_coef += d_ut * (axon.VAL - axon.thr)
            end
            nl1[n1].VAL = 0
        end
    end

    out = []
    for n2 in eachindex(nl2)
        for den in em[:, n2]
            if den.VAL >= den.thr
                nl2[n2].VAL += den.thr
                den.VAL = 0
            end
        end
        append!(out, [nl2[n2].thr * (nl2[n2].VAL >= nl2[n2].thr)])
    end

    return out
end

function get_param_count(hidden_sizes::AbstractVector)
    s = sum(hidden_sizes)
    for h in eachindex(hidden_sizes)[1:end-1]
        s += hidden_sizes[h] * hidden_sizes[h+1]
    end
    return s
end

function construct_network(params::AbstractVector, hidden_sizes::AbstractVector; number_of_edge_params=3, number_of_node_params=3)::network
    ret_net = network([])
    it = 1
    for i in length(hidden_sizes)-1
        neuron_param_ind   = it + hidden_sizes[i] * number_of_node_params
        conectum_param_ind = neuron_param_ind + hidden_sizes[i] * hidden_sizes[i+1] * number_of_edge_params
        mask_param_ind     = conectum_param_ind + hidden_sizes[i] * hidden_sizes[i+1]

        append!(ret_net, [neuron_layer(params[it:neuron_param_ind])])
        append!(ret_net, [conectum(reshape(params[neuron_param_ind+1:conectum_param_ind]), (hidden_sizes[i], hidden_sizes[i+1]))])
        append!(ret_net, [mask(reshape(params[conectum_param_ind+1:mask_param_ind], (hidden_sizes[i], hidden_sizes[i+1])))])

        it += neuron_params + conectum_params + mask_params
    end

    return ret_net
end

function deconstruct_network(net::network)
    ret_par = []
    for layer in net
        if typeof(layer) == neuron_layer
            append!(ret_par, [copy(layer)])
        elseif typeof(layer) == conectum
            append!(ret_par, [copy(layer)])
        elseif typeof(layer) == mask
            append!(ret_par, [copy(layer)])
        end
    end
    return ret_par
end

function apply_mask!(cons::conectum, m::mask)
    for i in eachindex(cons, m)
        if !m[i]
            cons[i] = missing
        end
    end
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

X = rand(20)
input_size = length(X)
hidden_sizes = [15, 10, 5, 2]
d_UT = 0.01 |> Float16 # update time
d_RT = 0.5 |> Float16 # reaction time
d_GT = 1 |> Float16 # growth time

nl1 = neuron_layer([node(rand(4)...) for _ in 1:input_size])
nl2 = neuron_layer([node(rand(4)...) for _ in 1:input_size])
con = conectum(reshape([edge(rand(4)...) for _ in 1:input_size^2], (input_size, input_size)))
net = network([nl2, nl1, con])
feed_forward!(nl1, nl2, con, X, d_UT) |> println

deconstruct_network(net)
construct_network()
