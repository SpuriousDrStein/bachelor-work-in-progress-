include("network_structure_3D.jl")
include("network_update_functions.jl")

# NETWORK HYPER PARAMETERS
MaxNumberNeurons = 1
NumberInitialNeurons = 1
MaxNearFieldDistance = 1

# NEURON HYPER PARAMETERS
MaxNumberDendrites = 1
MaxAxonLength = 1

# SYNAPS HYPER PARAMETERS
Threshold = 1
InitTheta = 1 # weights; <= 1; > 0
Decay = 0.1 # < 1; > 0
NeuroTransmitter = -1 # ~ range(-2, 2)




function MainLoop(iterations, N::Network) # the main update taking O(number_neurons * max(max_#_dendrites))
    for i in 1:iterations
        NN_t = NeuronNetowk(extractNeurons(N))
        activatable_neurons = getActivatable(NN_t)
        [updateQ!(n) for n in activatable_neurons]
        [reset!(n) for n in activatable_neurons] # where reset = resets the dendrites values

        # from start on activate! all neurons -> giving synaptic (dendraic) states for t+1

    end
end
