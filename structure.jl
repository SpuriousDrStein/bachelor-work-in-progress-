d_UT = 0.01 # update time
d_RT = 0.5 # reaction time
d_GT = 1 # growth time

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
neuron_layer = AbstrVector{node}

function feed_forward!(nl1::neuron_layer, nl2::neuron_layer, em::edge_matrix, x::AbstractVector, d_ut::Float16)
    for n1 in eachindex(nl1)
        nl1[n1].VAL += x[n1]
        nl1[n1].VAL += d_ut * cos(nl1[n1].phase)
        nl1[n1].res_coef += d_ut * (nl1[n1].VAL - nl1[n1].thr) # negative if not fiering -> making the resistance coeff negative -> increase probabilty of fiering

        if nl1[n1].VAL >= nl1[n1].thr
            for axon in em[n1, :]
                edg.VAL += nl1[n1].thr
                edg.VAL += d_ut * edg.res_coef
                edg.VAL += d_ut * cos(edg.freq_coef)
                edg.res_coef += d_ut * (edg.VAL - edg.thr)
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
        append(out, [nl2[n2].thr * (nl2[n2].VAL >= nl2[n2].thr)])
    end

    return out
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

raw(e::edge) = [copy(e.thr), copy(e.res_coef)]
raw(n::node) = [copy(n.thr), copy(n.phase)]

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

function train_agent()
end
# Q-table -> sigmoid -> board-mask
#  board-mask


function get_mask(con::conectum)
    # axon = 'a'
    # dendrite = 'd'
    # nucleus = 'n'
    # missing = 'x'

    out = []
    for i in eachindex(conectum)
        if typeof(conectum[i]) == node
            append!(out[i], 'n')
        elseif typeof(conectum[i]) == edge
            append!(out[i], 'x')
        else
            append!(out[i], 'x')
        end
    end
    return reshape(out, ())
end

function get_param_count(hidden_sizes::AbstractVector)
    s = 0
    for h in eachindex(hidden_sizes)[1:end-1]
        s += hidden_sizes[h] * hidden_sizes[h+1]
    end
    return s
end
