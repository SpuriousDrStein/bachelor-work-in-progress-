
function move(network, point::, new_point::)




function updateDirectionMap!(dmap::DirectionMap, net::Network) end
function updateDirectionMap!(dmap::DirectionMap, net::Network, env::AbstractArray) end
function extractNeurons(N::Network)::NeuronNetwork end
function feedForward!(net::Network, data::AbstractArray) end



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
