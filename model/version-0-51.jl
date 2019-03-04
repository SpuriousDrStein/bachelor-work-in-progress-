include("network_structure_3D.jl")
include("network_update_functions.jl")


function MainLoop(iterations, N::Network) # the main update taking O(number_neurons * max(max_#_dendrites))
    for i in 1:iterations
        NN_t = NeuronNetowk(extractNeurons(N))
        activatable_neurons = getActivatable(NN_t)
        [updateQ!(n) for n in activatable_neurons]
        [reset!(n) for n in activatable_neurons] # where reset = resets the dendrites values

        # from start on activate! all neurons -> giving synaptic (dendraic) states for t+1

    end
end
