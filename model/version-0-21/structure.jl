using Distributions


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
mutable struct m_v_pair
    mean::FloatN
    variance::FloatN
end
mutable struct min_max_pair
    min::Number
    max::Number
end
mutable struct InitializationPossition
    x::m_v_pair
    y::m_v_pair
    z::m_v_pair
end



# NETWORK STRUCTURES
mutable struct NeuroTransmitter # small possitive or small negative
    # addition for future versions --> have a function that gets the specific Synaps -> for (propergate, disperse, etc...)
    strength::FloatN
end
mutable struct Dendrite
    possition::Possition
    max_length::FloatN
    force::Force # influence by, for example, neurotransmitters
    liefTime::Integer
end
mutable struct AxonPoint
    possition::Possition
    max_length::FloatN
    force::Force
    liefTime::Integer
end
mutable struct Synaps
    possition::Possition
    Q::FloatN # charge at t
    QDecay::FloatN
    THR::FloatN # threshold
    NT::NeuroTransmitter # NT for different functionalities

    # values to manage synaps life cycles
    lifeTime::FloatN
    lifeDecay::FloatN

    numActivation::Integer
end
mutable struct AllCell
    cell::Union{AxonPoint, Dendrite, Synaps}
end
mutable struct Neuron
    possition::Possition
    force::Force
    priors::Array{Union{Missing, AllCell}, 1}
    posterior::Array{Union{Missing, AllCell}, 1}
    NT::NeuroTransmitter # NT for different functionalities

    lifeTime::FloatN
    lifeDecay::FloatN
    fitness::FloatN
end
mutable struct InputNode
    possition::Possition
    value::FloatN
end
mutable struct OutputNode
    possition::Possition
    value::FloatN
end
mutable struct Subnet # may be used for more update by predetermined references or for neurotransmitter dispersion
    possition::Possition
    range::FloatN
end
mutable struct Network
    enteries::Array{Union{AllCell, Neuron, InputNode, OutputNode}, 1}
    size::FloatN
end



# DNA STRUCTURES
mutable struct NetworkDNA
    networkSize::FloatN # i.e. range; centered at 0

    maxNeuronLifeTime::min_max_pair
    maxSynapsLifeTime::min_max_pair
    NeuronLifeTimeDecay::FloatN # >0; <1
    SynapsLifeTimeDecay::FloatN # >0; <1

    NeuronAccessDropout::FloatN # dropout probability for unspecific neuron selections (1 for early tests)

end
mutable struct NeuronDNA
    possition::InitializationPossition
    lifeTime::min_max_pair
    num_priors::min_max_pair
    num_posteriors::min_max_pair
end
mutable struct SynapsDNA
    QDecay::m_v_pair
    THR::m_v_pair
    LifeTime::min_max_pair
end
mutable struct DendriteDNA
    possition::InitializationPossition
    max_length::m_v_pair
    lifeTime::min_max_pair
end
mutable struct AxonPointDNA
    possition::InitializationPossition
    max_length::m_v_pair
    lifeTime::min_max_pair
end
