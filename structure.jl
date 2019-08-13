mutable struct edge
    VAL::Float16

    thr::Float16
    res_coef::Float16 # recovery speed
end

mutable struct node
    VAL::Float16

    thr::Float16
    phase::Float16
end

mutable struct neuron
    nucleus::node
    dendrites::AbstractArray{edge}
    axons::AbstractArray{edge}
end

[neuron(node(0,1,0), [edge(0,1)], [edge(0,1)]) for _ in 1:10]

function feed_forward!(dt, neurons::neuron...)
    for n in neurons
        for e in n.dendrites
            e.VAL += dt * e.res_coef
            e.VAL += dt * cos(e.freq_coef)

            if e.VAL >= e.thr
                n.nucleus.VAL += e.VAL
                e.res_coef += e.VAL - e.thr
                e.VAL = 0
            end
        end

        if n.nucleus.VAL >= n.nucleus.thr
            for e in n.axons
                e.VAL += dt * n.nucleus.VAL
            end
            n.nucleus.VAL = 0
        end
    end
end

# conectum
'''
     A  B  C
A   |x||_||_|
B   |_||x||_|
C   |_||_||x|
'''

mutable struct conectum
    connections::AbstractArray{edge}
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
