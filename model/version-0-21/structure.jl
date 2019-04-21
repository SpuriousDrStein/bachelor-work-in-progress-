# GENERAL STRUCTURES
FloatN = Float32
mutable struct Position
    x::FloatN
    y::FloatN
end
mutable struct Surge
    position::Position
    strength::FloatN
end
mutable struct Sink
    position::Position
    strength::FloatN
end



# DNA STRUCTURES
mutable struct NeuroTransmitterDNA
    init_strength::FloatN # mean should be 1 for most "accurate" effect
end
mutable struct SynapsDNA
    THR::FloatN
    r_rec::FloatN
    maxR::FloatN
end
mutable struct NeuronDNA
    THR::FloatN
end
mutable struct DNAStack
    nt_dna_samples::Array{NeuroTransmitterDNA}
    syn_dna_samples::Array{SynapsDNA}
    n_dna_samples::Array{NeuronDNA}
end


# NETWORK STRUCTURES
mutable struct InputNode
    position::Position
    value::FloatN
    referenced::Bool
end
mutable struct OutputNode
    position::Position
    value::FloatN
    referenced::Bool
end
mutable struct NeuroTransmitter # small possitive or small negative
    strength::FloatN
end
mutable struct Dendrite
    # constants
    lifeTime::FloatN

    # change at t
    position::Position
end
mutable struct AxonPoint
    # constants
    lifeTime::FloatN

    # change at t
    position::Position
end
mutable struct Synaps
    # constants
    lifeTime::FloatN

    # change at t
    position::Position
    NT::NeuroTransmitter
    Q::FloatN
    THR::FloatN
    R::FloatN
    RRecovery::FloatN # how quickly the value comes back up again
    maxR::FloatN

    total_fitness::FloatN
    d_total_fitness::FloatN
    destruction_threshold::FloatN
end
mutable struct AllCell
    cell::Union{AxonPoint, Dendrite, Synaps, InputNode, OutputNode}
end
mutable struct Neuron
    # constants
    den_init_interval::Integer
    ap_init_interval::Integer

    # change at t
    position::Position
    lifeTime::FloatN
    Q::FloatN
    THR::FloatN

    priors::Array{Union{Missing, AllCell}, 1}
    posteriors::Array{Union{Missing, AllCell}, 1}

    total_fitness::FloatN
    d_total_fitness::FloatN
    destruction_threshold::FloatN
end
mutable struct Subnet # may be used for more update by predetermined references or for neurotransmitter dispersion
    position::Position
    range::FloatN
end
mutable struct Network
    # constants
    size::FloatN
    global_stdv::FloatN
    neuron_lifetime::FloatN
    synaps_lifetime::FloatN
    dendrite_lifetime::FloatN
    axon_point_lifetime::FloatN
    min_fuse_distance::FloatN
    ap_sink_attractive_force::FloatN # force: AxonPoint's -> ap_sinks
    ap_surge_repulsive_force::FloatN
    den_surge_repulsive_force::FloatN
    input_attraction_force::FloatN
    output_attraction_force::FloatN
    neuron_repel_force::FloatN
    max_nt_strength::FloatN
    max_n_threshold::FloatN
    max_s_threshold::FloatN
    light_life_decay::FloatN
    heavy_life_decay::FloatN
    nt_retain_percentage::FloatN
    den_and_ap_init_range::FloatN
    neuron_init_interval::Integer
    ap_den_init_interval::Integer
    max_num_priors::Integer
    max_num_posteriors::Integer
    dna_stack::DNAStack

    # change at t
    components::Array{Union{Missing, AllCell, Neuron}, 1}
    IO_components::Array{Union{AllCell}, 1}
    total_fitness::FloatN
    n_counter::Integer
    syn_counter::Integer
    den_counter::Integer
    ap_counter::Integer
    n_destruction_threshold::FloatN
    s_destruction_threshold::FloatN
end
