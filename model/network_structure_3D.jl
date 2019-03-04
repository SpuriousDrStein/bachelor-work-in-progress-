# a collection of function to describe spatial movement of components in a network


# GENERAL
IntN = Int16
FloatN = Float16
Possition = Tuple{IntN, IntN, IntN}
NeuroTransmitter = FloatN

# NODES
EnteryPoints = Array{IPossition3D}
ExitPoints   = Array{IPossition3D}

mutable struct AxonPoint
    possition::Possition
    synaps::Union{Nothing, Synaps}
end

mutable struct Axon
    axonPoints::Array{Union{missing, AxonPoint}, 1}
    decay::FloatN
end

mutable struct Dendrite
    possition::Possition
    value::FloatN
    sensitivity::FloatN
    #synaps::Union{missing, Synaps}
    free::Bool
end

mutable struct Neuron
    possition::Possition
    thrsh::FloatN
    decay::FloatN
    dens::Array{Union{Missing, Dendrite}, 1}
    axon::Axon
end

mutable struct Synaps
    preSyn::AxonPoint
    postSyn::Dendrite
    weight::FloatN
    neuro_transmitter::NeuroTransmitter
end


# TRANSMISSION
DirectionPoint = Possition


# NETWORK
Network = Array{Union{Missing, Neuron, Dendrite, AxonPoint}, 3}
NeuronNetwork = Array{Neuron, 1}
DirectionMap = Array{Union{Direction}, 3}



# NEURO TRANSMITTER
inertNT = NeuroTransmitter(1.)
inhibitoryNT = NeuroTransmitter(-1.)
exitatoryNT(str) = NeuroTransmitter(1+str)
