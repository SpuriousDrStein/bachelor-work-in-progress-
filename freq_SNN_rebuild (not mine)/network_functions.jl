using Zygote


# STRUCTURE
mutable struct neuron
    threshold::Float32
    membrane::Float32
    spike_freq::Int16
    weights::Vector{Float32}
    bias::Float32
end

mutable struct layer
    nodes::Vector{neuron}
end
mutable struct data_layer
    nodes::Vector{Number}
end
mutable struct network
    layers::Vector{Union{layer, data_layer}}
end

println("structs imported")

# OVERRIDE
overrides = ["Base.length(::data_layer)", "Base.length(::layer)"]

function Base.length(dl::data_layer)
    length(dl.nodes)
end
function Base.length(l::layer)
    length(l.nodes)
end

println("overrides imported for: ", overrides)



# FORWARD PASS
function get_z(N::neuron, p_l::layer, r_max::Integer)
    sum = 0
    for i in eachindex(p_l.nodes, N.weights)
        sum += p_l.nodes[i].spike_freq * N.weights[i] + N.bias
    end
    min(sum/N.threshold * (sum > 0), r_max)
end
function get_z(N::neuron, p_l::data_layer, r_max::Integer)
    sum = 0
    for i in eachindex(p_l.nodes, N.weights)
        sum += p_l.nodes[i] * N.weights[i] + N.bias
    end
    min(sum/N.threshold * (sum > 0), r_max)
end

function update_membrane(N::neuron, z)
    return N.membrane + z - N.threshold * update_spike(N)
end

function update_spike(N::neuron)
    N.membrane - N.threshold >= 0
end

function simulate!t(N::neuron, input::layer, T::Integer) # !t denotes a temporary mutation - manly for multi threading
    freq = 0
    save_value = N.membrane
    for _ in 1:T
        z = get_z(N, input, T)
        N.membrane = update_membrane(N, z)
        freq += update_spike(N)
    end
    N.membrane = save_value
    return freq
end
function simulate!t(N::neuron, input::data_layer, T::Integer) # !t denotes a temporary mutation - manly for multi threading
    freq = 0
    save_value = N.membrane
    for _ in 1:T
        z = get_z(N, input, T)
        N.membrane = update_membrane(N, z)
        freq += update_spike(N)
    end
    N.membrane = save_value
    return freq
end

function update_neuron!(N::neuron, previous_layer::layer, T::Integer)
    freq = simulate!t(N, previous_layer, T)

    z = get_z(N, previous_layer, T)     # z(t, l)
    N.membrane = update_membrane(N, z)  # V(t, l)
    N.spike_freq = freq                 # a(t, l)
end
function update_neuron!(N::neuron, previous_layer::data_layer, T::Integer)
    N.spike_freq = simulate!t(N, previous_layer, T)

    z = get_z(N, previous_layer, T)     # z(t, l)
    N.membrane = update_membrane(N, z)  # V(t, l)
end

println("--- forwarding functions imported")

# BACKWARD PASS


println("--- backprop functions imported (not implemented)")

# STRUCTURE GENERATION & MAINTAINANCE
function initiate_dense_network_layers!(N::network, input_size::Integer, output_size::Integer, hidden_sizes::Tuple; rand_distribution=Uniform(-0.05, 0.05), threshold=1, init_bias=0)::network
    clear!(N)
    append!(N.layers, [data_layer([0 for _ in 1:input_size])])

    append!(N.layers, [layer([neuron(threshold, 0, 0, rand(rand_distribution, input_size), init_bias) for i in 1:hidden_sizes[1]])])

    for hsi in eachindex(hidden_sizes)[1:end-1]
        append!(N.layers, [layer([neuron(threshold, 0, 0, rand(rand_distribution, hidden_sizes[hsi]), init_bias) for _ in 1:hidden_sizes[hsi+1]])])
    end
    append!(N.layers, [layer([neuron(threshold, 0, 0, rand(rand_distribution, hidden_sizes[end]), init_bias) for _ in 1:output_size])])
    return N
end

function forward!(N::network, input, T::Integer)
    N.layers[1].nodes = input
    for i in eachindex(N.layers)[2:end]
        for neu in N.layers[i].nodes
            update_neuron!(neu, N.layers[i-1], T)
        end
    end
end

function clear!(N::network)
    N.layers = []
end

println("--- initialization functions imported")

function get_membrane_values(L::layer)
    [n.membrane for n in L.nodes]
end

function get_spike_freqs(L::layer)
    [n.spike_freq for n in L.nodes]
end

println("--- access functions imported")
