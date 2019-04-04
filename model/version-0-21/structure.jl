# GENERAL STRUCTURES
FloatN = Float32
mutable struct Possition
    x::FloatN
    y::FloatN
    z::FloatN
end
# mutable struct Surge
#     pos::Possition
#     strength::FloatN
# end
mutable struct Sink
    possition::Possition
    strength::FloatN
end



# DNA STRUCTURES
mutable struct min_max_pair
    min::Number
    max::Number
end
mutable struct InitializationPossition
    x::min_max_pair
    y::min_max_pair
    z::min_max_pair
end
mutable struct DendriteDNA
    max_length::min_max_pair
    lifeTime::min_max_pair
end
mutable struct AxonPointDNA
    max_length::min_max_pair
    lifeTime::min_max_pair
end
mutable struct NeuroTransmitterDNA
    init_strength::min_max_pair # mean should be 1 for most "accurate" effect
    dispersion_region::InitializationPossition
    dispersion_strength_scale::min_max_pair
    retain_percentage::min_max_pair
end
mutable struct SynapsDNA
    THR::min_max_pair
    QDecay::min_max_pair
    lifeTime::min_max_pair
end
mutable struct NeuronDNA
    max_num_priors::min_max_pair
    max_num_posteriors::min_max_pair
    lifeTime::min_max_pair
    den_and_ap_init_range::FloatN
    den_init_interval::min_max_pair
    ap_init_interval::min_max_pair
end
mutable struct DNAStack
    nt_dna_samples::Array{NeuroTransmitterDNA}
    ap_dna_samples::Array{AxonPointDNA}
    den_dna_samples::Array{DendriteDNA}
    syn_dna_samples::Array{SynapsDNA}
    n_dna_samples::Array{NeuronDNA}
end
mutable struct NetworkDNA
    networkSize::min_max_pair

    ap_sink_force::min_max_pair
    neuron_repel_force::min_max_pair
end


# NETWORK STRUCTURES
mutable struct InputNode
    possition::Possition
    value::FloatN
end
mutable struct OutputNode
    possition::Possition
    value::FloatN
end
mutable struct NeuroTransmitter # small possitive or small negative
    # change at t
    strength::FloatN
    dispersion_region::Possition # range given by strength of activation leftover
    range_scale::FloatN # how much the input is scaled to calculate the range
    retain_percentage::FloatN # how much of the old value is retained when influenced by new neuro transmitters
end
mutable struct Dendrite
    # constants
    max_length::FloatN
    lifeTime::FloatN

    # change at t
    possition::Possition
end
mutable struct AxonPoint
    # constants
    max_length::FloatN
    lifeTime::FloatN

    # change at t
    possition::Possition
end
mutable struct Synaps
    # constants
    id::Integer
    THR::FloatN
    QDecay::FloatN
    lifeTime::FloatN

    # change at t
    Q::FloatN
    possition::Possition
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
    possition::Possition
    Q::FloatN
    lifeTime::FloatN
    priors::Array{Union{Missing, AllCell}, 1}
    posteriors::Array{Union{Missing, AllCell}, 1}
    fitness::FloatN
    total_fitness::FloatN
end
mutable struct Subnet # may be used for more update by predetermined references or for neurotransmitter dispersion
    possition::Possition
    range::FloatN
end
mutable struct Network
    # constants
    size::FloatN
    maxNeuronLifeTime::FloatN
    maxSynapsLifeTime::FloatN
    maxDendriteLifeTime::FloatN
    maxAxonPointLifeTime::FloatN
    minFuseDistance::FloatN
    ap_sink_attractive_force::FloatN # force: AxonPoint's -> ap_sinks
    neuron_repel_force::FloatN
    max_nt_dispersion_strength_scale::FloatN
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
end
