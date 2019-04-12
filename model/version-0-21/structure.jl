# GENERAL STRUCTURES
FloatN = Float32
mutable struct Position
    x::FloatN
    y::FloatN
    z::FloatN
end
# mutable struct Surge
#     pos::Position
#     strength::FloatN
# end
mutable struct Sink
    position::Position
    strength::FloatN
end



# DNA STRUCTURES
mutable struct DendriteDNA
    max_length::FloatN
    lifeTime::FloatN
end
mutable struct AxonPointDNA
    max_length::FloatN
    lifeTime::FloatN
end
mutable struct NeuroTransmitterDNA
    init_strength::FloatN # mean should be 1 for most "accurate" effect
    retain_percentage::FloatN
end
mutable struct SynapsDNA
    THR::FloatN
    QDecay::FloatN
    lifeTime::FloatN
end
mutable struct NeuronDNA
    max_num_priors::FloatN
    max_num_posteriors::FloatN
    lifeTime::FloatN
    den_and_ap_init_range::FloatN
    den_init_interval::FloatN
    ap_init_interval::FloatN
end
mutable struct NetworkDNA
    ap_sink_force::FloatN
    neuron_repel_force::FloatN
end
mutable struct DNAStack
    nt_dna_samples::Array{NeuroTransmitterDNA}
    ap_dna_samples::Array{AxonPointDNA}
    den_dna_samples::Array{DendriteDNA}
    syn_dna_samples::Array{SynapsDNA}
    n_dna_samples::Array{NeuronDNA}
end


# NETWORK STRUCTURES
mutable struct InputNode
    position::Position
    value::FloatN
end
mutable struct OutputNode
    position::Position
    value::FloatN
end
mutable struct NeuroTransmitter # small possitive or small negative
    # change at t
    strength::FloatN
    retain_percentage::FloatN # how much of the old value is retained when influenced by new neuro transmitters
end
mutable struct Dendrite
    # constants
    max_length::FloatN
    lifeTime::FloatN

    # change at t
    position::Position
end
mutable struct AxonPoint
    # constants
    max_length::FloatN
    lifeTime::FloatN

    # change at t
    position::Position
end
mutable struct Synaps
    # constants
    id::Integer
    THR::FloatN
    QDecay::FloatN
    lifeTime::FloatN

    # change at t
    Q::FloatN
    position::Position
    NT::NeuroTransmitter # NTs::Array{NeuroTransmitter}
    total_fitness::FloatN
end
mutable struct AllCell
    cell::Union{AxonPoint, Dendrite, Synaps, InputNode, OutputNode}
end
mutable struct Neuron
    # constants
    id::Integer
    den_init_interval::Integer
    ap_init_interval::Integer
    den_and_ap_init_range::FloatN

    # change at t
    position::Position
    Q::FloatN
    lifeTime::FloatN
    priors::Array{Union{Missing, AllCell}, 1}
    posteriors::Array{Union{Missing, AllCell}, 1}
    fitness::FloatN
    total_fitness::FloatN
end
mutable struct Subnet # may be used for more update by predetermined references or for neurotransmitter dispersion
    position::Position
    range::FloatN
end
mutable struct Network
    # constants
    size::FloatN
    global_stdv::FloatN
    maxNeuronLifeTime::FloatN
    maxSynapsLifeTime::FloatN
    maxDendriteLifeTime::FloatN
    maxAxonPointLifeTime::FloatN
    minFuseDistance::FloatN
    ap_sink_attractive_force::FloatN # force: AxonPoint's -> ap_sinks
    neuron_repel_force::FloatN
    max_nt_strength::FloatN
    max_threshold::FloatN
    fitness_decay::FloatN
    random_fluctuation_scale::FloatN
    neuron_init_interval::Integer
    min_ap_den_init_interval::Integer
    dna_stack::DNAStack

    # change at t
    components::Array{Union{Missing, AllCell, Neuron}, 1}
    IO_components::Array{Union{AllCell}, 1}
    life_decay::FloatN
    total_fitness::FloatN
    n_id_counter::Integer
    s_id_counter::Integer
    n_counter::Integer
    den_counter::Integer
    ap_counter::Integer
end
