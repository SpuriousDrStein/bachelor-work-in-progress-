# GENERAL STRUCTURES
FloatN = Float32

mutable struct Possition
    x::FloatN
    y::FloatN
    z::FloatN
end

mutable struct Force # identity vector with a force
    x::FloatN
    y::FloatN
    z::FloatN
    strength::FloatN
end

mutable struct InitializationPossition
    x::m_v_pair
    y::m_v_pair
    z::m_v_pair
end

mutable struct m_v_pair
    mean::FloatN
    variance::FloatN
end

mutable struct min_max_pair
    min::Number
    max::Number
end





# NETWORK STRUCTURES
mutable struct NeuroTransmitter # small possitive or small negative
    fingerprint::String

    # change at t
    strength::FloatN
    dispersion_region::Subnet
end

mutable struct Dendrite
    # constants
    max_length::FloatN
    lifeTime::Integer

    # change at t
    possition::Possition
    force::Force
    lifeDecay::Integer
end

mutable struct AxonPoint
    # constants
    max_length::FloatN
    lifeTime::Integer

    # change at t
    possition::Possition

    # # deltas
    # lifeDecay::Integer
    # force::Force
end

mutable struct Synaps
    # constants
    id::Integer
    THR::FloatN
    QDecay::FloatN
    lifeTime::Integer

    # change at t
    Q::FloatN
    possition::Possition
    # NTs::Array{NeuroTransmitter}
    NT::NeuroTransmitter
    updated::Bool

    # # deltas
    # lifeDecay::Integer
end

mutable struct AllCell
    cell::Union{AxonPoint, Dendrite, Synaps}
end

mutable struct Neuron
    # constants
    id::Integer

    # change at t
    possition::Possition
    Q::FloatN
    lifeTime::Integer
    priors::Array{Union{Missing, AllCell}, 1}
    posteriors::Array{Union{Missing, AllCell}, 1}
    fitness::FloatN
    total_fitness::FloatN

    # # deltas
    # force::Force
    # lifeDecay::Integer
end
mutable struct InputNode
    # constants
    possition::Possition

    # changes at t
    value::FloatN
end

mutable struct OutputNode
    # constants
    possition::Possition

    # changes at t
    value::FloatN
end

mutable struct Subnet # may be used for more update by predetermined references or for neurotransmitter dispersion
    possition::Possition
    range::FloatN
end

mutable struct Network
    # constants
    size::FloatN
    maxNeuronLifeTime::min_max_pair
    maxSynapsLifeTime::min_max_pair
    maxDendriteLifeTime::min_max_pair
    maxAxonPointLifeTime::min_max_pair
    NeuronAccessDropout::FloatN

    # change at t
    enteries::Array{Union{AllCell, Neuron, InputNode, OutputNode}, 1}
end
end



# DNA STRUCTURES
mutable struct DendriteDNA
    max_length::m_v_pair
    lifeTime::min_max_pair
    init_pos::InitializationPossition
end
mutable struct AxonPointDNA
    max_length::m_v_pair
    lifeTime::min_max_pair
    init_pos::InitializationPossition
end
mutable struct NeuroTransmitterDNA
    fingerprint::String
    strength::m_v_pair
    dispersion_region::Tuple{InitializationPossition, m_v_pair} # possition range
end
mutable struct SynapsDNA
    THR::m_v_pair
    QDecay::m_v_pair
    lifeTime::min_max_pair
    init_pos::InitializationPossition
    NTs::NeuroTransmitterDNA
end
mutable struct NeuronDNA
    init_pos::InitializationPossition
    lifeTime::min_max_pair
    NT::NeuroTransmitter # NT for different functionalities
    fitness::FloatN

    # deltas
    force::Force
    lifeDecay::Integer
end

mutable struct NetworkDNA
    networkSize::m_v_pair

    maxNeuronLifeTime::min_max_pair
    maxSynapsLifeTime::min_max_pair
    maxDendriteLifeTime::min_max_pair
    maxAxonPointLifeTime::min_max_pair

    NeuronAccessDropout::FloatN # dropout probability for unspecific neuron selections (1 for early tests)
end
