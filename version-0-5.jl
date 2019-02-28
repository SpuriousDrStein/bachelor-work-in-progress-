FloatN = Float16

mutable struct FPossition3D
    x::FloatN
    y::FloatN
    z::FloatN
end

mutable struct IPossition3D
    x::IntN
    y::IntN
    z::IntN
end

DirectionPoint = Tuple{FloatN, FloatN, FloatN}
NeuroTransmitter = Function

mutable struct AxonPoint
    poss::FPossition3D
    synaps::Union{Nothing, Synaps}
end

mutable struct Axon
    axonPoints::Array{Union{missing, AxonPoint}, 1}
    decay::FloatN
end

mutable struct Dendrite
    poss::FPossition3D
    value::FloatN
    sensitivity::FloatN
    #synaps::Union{missing, Synaps}
    free::Bool
end

mutable struct Neuron
    poss::IPossition3D
    value::FloatN
    thrsh::FloatN
    decay::FloatN
    NT!::NeuroTransmitter
    dens::Array{Union{missing, Dendrite}, 1}
    axon::Axon
end

# mutable struct Synaps{preSynT, postSynT} where preSynT <: Neuron, postSynT <: Neuron
#     preSyn::preSynT
#     postSyn::postSynT
# end
mutable struct Synaps
    preSyn::AxonPoint
    postSyn::Dendrite
end


normalNT = NeuroTransmitter((reciever, value) -> reciever = value)


Network = Array{Union{DirectionPoint, Neuron, Dendrite, AxonPoint}, 3}
NeuronNetwork = Array{Neuron, 1}
DirectionMap = Array{Union{DirectionPoint}}
EnteryPoints = Array{IPossition3D}
ExitPoints   = Array{IPossition3D}

function updateDirectionMap!(dmap::DirectionMap, net::Network) end
function updateDirectionMap!(dmap::DirectionMap, net::Network, env::AbstractArray) end
function extractNeurons(N::Network)::NeuronNetwork end
function feedForward!(net::Network, data::AbstractArray) end


function MainLoop(iterations, N::Network) # the main update taking O(number_neurons * max(max_#_dendrites))
    for i in 1:iterations
        NN_t = NeuronNetowk(extractNeurons(N))
        activatable_neurons = getActivatable(NN_t)
        [updateQ!(n) for n in activatable_neurons]
        [reset!(n) for n in activatable_neurons] # where reset = resets the dendrites values
        
        # from start on activate! all neurons -> giving synaptic (dendraic) states for t+1

    end
end


# CORE
function activate!(n::Neuron) # propergate signal through the axon and activate axonPoint's
    activate!(n.axon, n.Q, n.NT!)
    n.Q = 0
end

function activate!(a::Axon, value::FloatN, nt!::NeuroTransmitter) # update a dendrite given an axons value and neuroTransmitter
    for (i, ap) in enumerate(skipmissing(a.axonPoints))
        if isSynaps(ap)
            nt!(ap.synaps.postSyn.value, value * FloatN(a.decay^i))
        else
            # open connection, add this signal (value * FloatN(a.decay^i))
            # to DirectionMap
        end
    end
end


function updateQ!(NN::NeuronNetwork)
    for n in getActivatable(NN)
        updateQ!(n)
    end
end

function updateQ!(n::Nueorn)
    n.Q = sum(collect())
    if testAct(activatable_n)
        activate!(n)
    else
        n.Q *= n.decay
    end
end



# UTIL
function testThreshold(n::Neuron) # tests threshold for activation
    return n.Q >= n.threshold
end

function getActivatable(NN::NeuronNetwork) # gets all activatable neurons in a net
    return [n for n in NN if testThreshold(n)]
end

function collect(n::Neuron) # collects all values from Dendrites
    return [syn_den.value for syn_den in getNotFreeDens(n)]
end

function getFreeDens(NN::NeuronNetwork) # returns free dendrites i.e. which are no synaps
    return [getFreeDens(n) for n in NN]
end

function getFreeDens(n::Neuron) # get all free dens in one neuron
    return [d for d in skipmissing(n.dens) if d.free]
end

function getNotFreeDens(n::Neuron) # get all free dens in one neuron
    return [d for d in skipmissing(n.dens) if !d.free]
end

function isSynaps(ap::AxonPoint)
    return ismissing(ap.synaps)
end
