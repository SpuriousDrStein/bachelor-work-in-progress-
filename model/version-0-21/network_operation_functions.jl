function decay_charge!(syn::Synaps)
    syn.Q *= syn.QDecay
end
function updateQ!(syn::Synaps)
    decay_charge!(syn)
    syn.Q *= syn.NT.strength
end

function accm!(N::Neuron, all_synapses::Array, dispersion_collection::Dict)
    N.Q = 0.

    prior_all = get_prior_all_cells(N)
    input_syns = get_synapses(prior_all)

    if input_syns != []
        input_valid_syns = get_activatable_synapses(input_syns)
        input_nodes = get_input_nodes(prior_all)

        input_v = [[s.THR for s in input_valid_syns]..., input_nodes...]

        if input_v != []
            N.Q = sum(input_v)

            # calculate new fitness values
            N.fitness += length(input_syns)
            N.fitness += sum(input_v)
            N.fitness *= NN.fitness_decay
            N.total_fitness += N.fitness
        else
            N.Q = 0.

            N.fitness *= NN.fitness_decay
        end

        for is in input_valid_syns
            for s in get_synapses(Subnet(is.NT.dispersion_region, is.NT.range_scale * (is.Q - is.THR)), all_synapses)
                if s in keys(dispersion_collection)
                    dispersion_collection[s] = dispersion_collection[s] .+ (s.NT.strength, 1)
                else
                    dispersion_collection[s] = (s.NT.strength, 1)
                end
            end
        end   # dict of: synaps => neurotransmitter change
    end
end

function propergate!(N::Neuron, sink_list::Array)
    post_all = get_posterior_all_cells(N)
    post_syns = get_synapses(post_all)
    post_aps =  get_axon_points(post_all)

    for s in post_syns
        s.Q += N.Q/length(post_all)
    end
    for ap in post_aps
        append!(sink_list, [Sink(ap.possition, N.Q/length(post_all))])
    end
end

function value_step!(NN::Network, input::Array)
    network_all_cells = get_all_all_cells(NN)
    n_ind = get_all_neuron_indecies(NN)
    in_nodes = get_input_nodes_in_all(NN)
    out_nodes = get_output_nodes_in_all(NN)

    dens = get_dendrites_in_all(network_all_cells)
    # APs = get_axon_points_in_all(network_all_cells)

    dispersion_collection = Dict()
    den_sinks = []
    ap_sinks = [Sink(o_n.cell.possition, NN.ap_sink_attractive_force) for o_n in out_nodes] # array of ap sinks

    # 1
    for i in eachindex(in_nodes, input)
        in_nodes[i].cell.value = input[i]
        append!(den_sinks, [Sink(in_nodes[i].cell.possition, in_nodes[i].cell.value)])
    end

    if get_all_neurons(NN) == []
        println("exit because no neurons exist")
        return [], []
    end

    # 2
    for n_i in n_ind
        # accumulate!(n, dropout(synapses, NN.synapsesAccessDropout), dispersion_collection)
        accm!(NN.components[n_i], get_synapses(get_synapses_in_all(network_all_cells)), dispersion_collection) #all_synapses::Array{Synaps}, dispersion_collection::Dict{Synaps, Pair{FloatN, Integer}}
    end

    # 3
    # this puts the NT(t-1) and then calculates NT(t)
    # this can be reversed
    for s in get_synapses_in_all(network_all_cells)
        if s.cell.Q >= s.cell.THR
            s.cell.Q = 0.
            s.cell.total_fitness += 1
            s.cell.total_fitness *= NN.fitness_decay
        else
            updateQ!(s.cell)
            s.cell.total_fitness *= NN.fitness_decay
        end

        dispersion_value, n = get(dispersion_collection, s.cell, (1, 1))
        avg_NT_change = dispersion_value/n

        # change nt and update syn-fitness
        s.cell.total_fitness += ((avg_NT_change+s.cell.NT.strength)/2)/(abs(avg_NT_change-s.cell.NT.strength)+0.00001)
        s.cell.NT.strength = s.cell.NT.retain_percentage * s.cell.NT.strength + (1-s.cell.NT.retain_percentage) * avg_NT_change

        s.cell.lifeTime -= NN.life_decay
    end

    # 4
    for n_i in n_ind
        propergate!(NN.components[n_i], den_sinks)
        NN.components[n_i].lifeTime -= NN.life_decay
    end

    for d in get_dendrites_in_all(network_all_cells)
        append!(ap_sinks, [Sink(d.cell.possition, NN.ap_sink_attractive_force)])
    end

    return den_sinks, ap_sinks
end


function state_step!(NN::Network, den_sinks, ap_sinks)
    # update spatial relation
    network_all_cells = get_all_all_cells(NN)
    neurons = get_all_neurons(NN)
    in_nodes = get_input_nodes_in_all(NN)
    out_nodes = get_output_nodes_in_all(NN)


    for den in get_dendrites_in_all(network_all_cells)
        total_v = [0.,0.,0.]
        for d_sink in den_sinks
            dir = direction(den.cell.possition, d_sink.possition)
            mag = NN.minFuseDistance / vector_length(dir)
            total_v .+= normalize(dir) .* mag .* (1 + d_sink.strength)
        end
        total_v ./= length(den_sinks)

        den.cell.possition += Possition(total_v...)


        for o_n in out_nodes
            if distance(den.cell.possition, o_n.cell.possition) <= NN.minFuseDistance
                fuse!(den, o_n)
            end
        end

        den.cell.lifeTime -= NN.life_decay
    end

    for ap in get_axon_points_in_all(network_all_cells)
        total_v = [0.,0.,0.]
        for ap_sink in ap_sinks
            dir = direction(ap.cell.possition, ap_sink.possition)
            mag = NN.minFuseDistance / vector_length(dir)
            total_v .+= normalize(dir) .* mag .* (1 + ap_sink.strength)
        end
        total_v ./= length(ap_sinks)

        ap.cell.possition += Possition(total_v...)

        # fuse with dendrite if near
        for d in get_dendrites_in_all(network_all_cells)
            if distance(ap.cell.possition, d.cell.possition) <= NN.minFuseDistance
                # random possition for dispersion
                nt = unfold(rand(NN.dna_stack.nt_dna_samples))

                # basicly rectify_possition() only for dispersion_region
                if distance(nt.dispersion_region, Possition(0,0,0)) > NN.size
                    nt.dispersion_region = normalize(nt.dispersion_region) * NN.size
                end

                half_dist = direction(ap.cell.possition, d.cell.possition) ./ 2
                pos = ap.cell.possition + Possition(half_dist...)

                fuse!(d, ap, unfold(rand(NN.dna_stack.syn_dna_samples), copy(NN.s_id_counter), pos , nt, NN.life_decay))
                NN.s_id_counter += 1
            end
        end
        for i_n in in_nodes
            if distance(ap.cell.possition, i_n.cell.possition) <= NN.minFuseDistance
                fuse!(ap, i_n)
            end
        end

        ap.cell.lifeTime -= NN.life_decay
    end

    # repel neurons away each other
    if length(neurons) > 1
        total_v = [[0.,0.,0.] for _ in 1:length(neurons)]
        for i in eachindex(neurons)
            local_v = [0.,0.,0.]

            for n in neurons
                if neurons[i] !== n
                     local_v += direction(n.possition, neurons[i].possition)
                end
            end

            total_v[i] = normalize(local_v) .* NN.neuron_repel_force #./ (length(neurons) .- 1)
        end
        for i in eachindex(total_v, neurons)
            neurons[i].possition += Possition(total_v[i]...)
        end
    end
end

# STRUCTURE GENERATION FUNCTIONS
function fuse!(den::AllCell, ap::AllCell, to::Synaps)
    if typeof(den.cell) != Dendrite || typeof(ap.cell) != AxonPoint # throw("incorect fuse!($(typeof(den.cell)), $(typeof(ap.cell)))"); end
        # println("warning: attempted prohibited fusion")
    else
        den.cell = to
        ap.cell = to
    end
end
function fuse!(network_node::AllCell, ap::AllCell)
    if typeof(ap.cell) != AxonPoint || typeof(network_node.cell) != InputNode || typeof(network_node.cell) != OutputNode; throw("incorect fuse!($(typeof(ap.cell)), $(typeof(network_node.cell)))"); end
    ap.cell = network_node.cell
end
function fuse!(network_node::AllCell, den::AllCell)
    if typeof(ap.cell) != AxonPoint || typeof(network_node.cell) != InputNode || typeof(network_node.cell) != OutputNode; throw("incorect fuse!($(typeof(ap.cell)), $(typeof(network_node.cell)))"); end
    den.cell = network_node.cell
end
function add_neuron!(NN::Network, n_dna::NeuronDNA)
    n = unfold(n_dna, copy(NN.n_id_counter))
    NN.n_id_counter += 1

    rectify_possition!(n, NN.size)
    append!(NN.components, [n])
    return n
end
function add_dendrite!(NN::Network, N::Neuron)
    if has_empty_prior(N)
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                d = AllCell(unfold(rand(NN.dna_stack.den_dna_samples)))
                rectify_possition!(d.cell, NN.size)

                N.priors[i] = d
                append!(NN.components, [d])
                return nothing
            end
        end
    end
end
function add_axon_point!(NN::Network, N::Neuron)
    if has_empty_post(N)
        for i in eachindex(N.posteriors)
            if ismissing(N.posteriors[i])
                ap = AllCell(unfold(rand(NN.dna_stack.ap_dna_samples)))
                rectify_possition!(ap.cell, NN.size)

                N.posteriors[i] = ap
                append!(NN.components, [ap])
                return nothing
            end
        end
    end
end
function populate_network!(NN::Network, num_neurons::Integer, max_num_priors::Integer, max_num_post::Integer)
    for _ in 1:num_neurons
        # add_neuron! adds it to component list as well
        n = add_neuron!(NN, rand(NN.dna_stack.n_dna_samples))

        for _ in 1:rand(1:max_num_priors)
            add_dendrite!(NN, n)
        end
        for _ in 1:rand(1:max_num_post)
            add_axon_point!(NN, n)
        end
    end
end



# VERIFICATION FUNCTIONS
function rectify_possition!(c, nn_size::FloatN)
    if distance(c.possition, Possition(0,0,0)) > nn_size
        c.possition = Possition((normalize(c.possition) * nn_size)...)
    end
end
function clean_network_components!(NN)
    NN.components = collect(skipmissing(NN.components))
    NN.components

    for n1 in eachindex(NN.components)
        if typeof(NN.components[n1]) == AllCell && typeof(NN.components[n1].cell) != InputNode && typeof(NN.components[n1].cell) != OutputNode
            if NN.components[n1].cell.lifeTime <= 0
                if typeof(NN.components[n1]) == Synaps
                    NN.total_fitness += NN.components[n1].cell.total_fitness
                end
                NN.components[n1] = missing
            end

            if !ismissing(NN.components[n1])
                if typeof(NN.components[n1].cell) == Synaps
                    for n2 in eachindex(NN.components)
                        if typeof(NN.components[n2]) == AllCell
                            if typeof(NN.components[n2].cell) == Synaps
                                if n1 != n2
                                    if NN.components[n1].cell === NN.components[n2].cell
                                        NN.components[n2] = missing
                                    end
                                end
                            end
                        end
                    end
                end
            end
        elseif typeof(NN.components[n1]) == Neuron
            if NN.components[n1].lifeTime <= 0
                NN.total_fitness += NN.components[n1].total_fitness
                NN.components[n1] = missing
            end
        elseif typeof(NN.components[n1]) == InputNode || typeof(NN.components[n1]) == OutputNode
            println("input or output registered in components")
        end
    end
    NN.components = collect(skipmissing(NN.components))
end
function rectifyDNA!(dna::DendriteDNA, NN::Network)
    # accuracy_penalty = 0 # how many values had to be changed as baseline negative fitness

    dna.lifeTime.min = clamp(dna.lifeTime.min, 1., NN.maxDendriteLifeTime-1.)
    dna.lifeTime.max = clamp(dna.lifeTime.max, dna.lifeTime.min, NN.maxDendriteLifeTime)

    dna.max_length.min = clamp(dna.max_length.min, 1., (NN.size/2)-1)
    dna.max_length.max = clamp(dna.max_length.max, dna.max_length.min+0.01, NN.size/2) # 0.01 for max > min

    dna.init_pos.x.min = clamp(dna.init_pos.x.min, -NN.size, NN.size-1.)
    dna.init_pos.x.max = clamp(dna.init_pos.x.max, dna.init_pos.x.min+0.1, NN.size)
    dna.init_pos.y.min = clamp(dna.init_pos.y.min, -NN.size, NN.size-1.)
    dna.init_pos.y.max = clamp(dna.init_pos.y.max, dna.init_pos.y.min+0.01, NN.size)
    dna.init_pos.z.min = clamp(dna.init_pos.z.min, -NN.size, NN.size-1.)
    dna.init_pos.z.max = clamp(dna.init_pos.z.max, dna.init_pos.z.min+0.01, NN.size)

    # return accuracy_penalty
end
function rectifyDNA!(dna::AxonPointDNA, NN::Network);
    # accuracy_penalty = 0 # how many values had to be changed as baseline negative fitness

    dna.lifeTime.min = clamp(dna.lifeTime.min, 1., NN.maxDendriteLifeTime-1.)
    dna.lifeTime.max = clamp(dna.lifeTime.max, dna.lifeTime.min, NN.maxDendriteLifeTime)

    dna.max_length.min = clamp(dna.max_length.min, 1., (NN.size/2)-1)
    dna.max_length.max = clamp(dna.max_length.max, dna.max_length.min+0.01, NN.size/2) # 0.01 for max > min

    dna.init_pos.x.min = clamp(dna.init_pos.x.min, -NN.size, NN.size-1.)
    dna.init_pos.x.max = clamp(dna.init_pos.x.max, dna.init_pos.x.min+0.1, NN.size)
    dna.init_pos.y.min = clamp(dna.init_pos.y.min, -NN.size, NN.size-1.)
    dna.init_pos.y.max = clamp(dna.init_pos.y.max, dna.init_pos.y.min+0.01, NN.size)
    dna.init_pos.z.min = clamp(dna.init_pos.z.min, -NN.size, NN.size-1.)
    dna.init_pos.z.max = clamp(dna.init_pos.z.max, dna.init_pos.z.min+0.01, NN.size)

    # return accuracy_penalty
end
function rectifyDNA!(dna::NeuroTransmitterDNA, NN::Network)
    # accuracy_penalty = 0 # how many values had to be changed as baseline negative fitness

    dna.init_strength.min = max(dna.init_strength.min, 1.)
    dna.init_strength.max = max(dna.init_strength.max, dna.init_strength.min+0.01)

    dna.dispersion_strength_scale.min = clamp(dna.dispersion_strength_scale.min, 0.1, NN.max_nt_dispersion_strength_scale-0.1)
    dna.dispersion_strength_scale.max = clamp(dna.dispersion_strength_scale.max, dna.dispersion_strength_scale.min+0.01, NN.max_nt_dispersion_strength_scale)

    dna.retain_percentage.min = clamp(dna.retain_percentage.min, 0, 1)
    dna.retain_percentage.max = clamp(dna.retain_percentage.max, dna.retain_percentage.min+0.001, 1)

    dna.dispersion_region.x.min = clamp(dna.dispersion_region.x.min, -NN.size, NN.size-1.)
    dna.dispersion_region.x.max = clamp(dna.dispersion_region.x.max, dna.dispersion_region.x.min+0.1, NN.size)
    dna.dispersion_region.y.min = clamp(dna.dispersion_region.y.min, -NN.size, NN.size-1.)
    dna.dispersion_region.y.max = clamp(dna.dispersion_region.y.max, dna.dispersion_region.y.min+0.01, NN.size)
    dna.dispersion_region.z.min = clamp(dna.dispersion_region.z.min, -NN.size, NN.size-1.)
    dna.dispersion_region.z.max = clamp(dna.dispersion_region.z.max, dna.dispersion_region.z.min+0.01, NN.size)

    # return accuracy_penalty
end
function rectifyDNA!(dna::SynapsDNA, NN::Network; max_q_decay=0.1)
    # accuracy_penalty = 0 # how many values had to be changed as baseline negative fitness

    dna.lifeTime.min = clamp(dna.lifeTime.min, 1., NN.maxSynapsLifeTime-1.)
    dna.lifeTime.max = clamp(dna.lifeTime.max, dna.lifeTime.min+0.01, NN.maxSynapsLifeTime)

    dna.QDecay.min = clamp(dna.QDecay.min, max_q_decay, 0.97)
    dna.QDecay.max = clamp(dna.QDecay.max, dna.QDecay.min+0.01, 0.99)

    dna.THR.min = clamp(dna.THR.min, 0.1, NN.max_threshold-0.1)
    dna.THR.max = clamp(dna.THR.max, dna.THR.min+0.01, NN.max_threshold)

    rectifyDNA!(dna.NT, NN)

    # return accuracy_penalty
end
function rectifyDNA!(dna::NeuronDNA, NN::Network)
    # accuracy_penalty = 0 # how many values had to be changed as baseline negative fitness

    dna.lifeTime.min = clamp(dna.lifeTime.min, 1., NN.maxNeuronLifeTime-1.)
    dna.lifeTime.max = clamp(dna.lifeTime.max, dna.lifeTime.min+0.1, NN.maxNeuronLifeTime)

    dna.max_num_priors.min = max(dna.max_num_priors.min, 1)
    dna.max_num_priors.max = max(dna.max_num_priors.max, dna.max_num_priors.min+1)

    dna.max_num_posteriors.min = max(dna.max_num_posteriors.min, 1)
    dna.max_num_posteriors.max = max(dna.max_num_posteriors.max, dna.max_num_posteriors.min+1)

    dna.init_pos.x.min = clamp(dna.init_pos.x.min, -NN.size, NN.size-1.)
    dna.init_pos.x.max = clamp(dna.init_pos.x.max, dna.init_pos.x.min+0.1, NN.size)
    dna.init_pos.y.min = clamp(dna.init_pos.y.min, -NN.size, NN.size-1.)
    dna.init_pos.y.max = clamp(dna.init_pos.y.max, dna.init_pos.y.min+0.01, NN.size)
    dna.init_pos.z.min = clamp(dna.init_pos.z.min, -NN.size, NN.size-1.)
    dna.init_pos.z.max = clamp(dna.init_pos.z.max, dna.init_pos.z.min+0.01, NN.size)

    # return accuracy_penalty
end
function rectifyDNA!(dna::DNAStack, NN::Network)
    # accuracy_penalty = 0 # how many values had to be changed as baseline negative fitness

    for nt_dna_s in dna.nt_dna_samples
        rectifyDNA!(nt_dna_s, NN)
    end
    for ap_dna_s in dna.ap_dna_samples
        rectifyDNA!(ap_dna_s, NN)
    end
    for den_dna_s in dna.den_dna_samples
        rectifyDNA!(den_dna_s, NN)
    end
    for syn_dna_s in dna.syn_dna_samples
        rectifyDNA!(syn_dna_s, NN)
    end
    for n_dna_s in dna.n_dna_samples
        rectifyDNA!(n_dna_s, NN)
    end

    # return accuracy_penalty
end
function rectifyDNA!(dna::NetworkDNA)
    # test = [copy(dna.networkSize.min),
    #         copy(dna.networkSize.min),
    #         copy(dna.maxNeuronLifeTime.min),
    #         copy(dna.maxNeuronLifeTime.max),
    #         copy(dna.maxSynapsLifeTime.min),
    #         copy(dna.maxSynapsLifeTime.max),
    #         copy(dna.maxDendriteLifeTime.min),
    #         copy(dna.maxDendriteLifeTime.max),
    #         copy(dna.maxAxonPointLifeTime.min),
    #         copy(dna.maxAxonPointLifeTime.max)]
    # accuracy_penalty = 0. # how many values had to be changed as baseline negative fitness

    dna.networkSize.min = max(1., dna.networkSize.min)
    dna.networkSize.max = max(dna.networkSize.min+1., dna.networkSize.max)
    # accuracy_penalty += (dna.networkSize.min != test[1])
    # accuracy_penalty += (dna.networkSize.max != test[2])

    dna.maxNeuronLifeTime.min = max(1., dna.maxNeuronLifeTime.min)
    dna.maxNeuronLifeTime.max = max(dna.maxNeuronLifeTime.min+1., dna.maxNeuronLifeTime.max)
    # accuracy_penalty += (dna.maxNeuronLifeTime.min != test[3])
    # accuracy_penalty += (dna.maxNeuronLifeTime.max != test[4])

    dna.maxSynapsLifeTime.min = max(1., dna.maxSynapsLifeTime.min)
    dna.maxSynapsLifeTime.max = max(dna.maxSynapsLifeTime.min+1., dna.maxSynapsLifeTime.max)
    # accuracy_penalty += (dna.maxSynapsLifeTime.min != test[5])
    # accuracy_penalty += (dna.maxSynapsLifeTime.max != test[6])

    dna.maxDendriteLifeTime.min = max(1., dna.maxDendriteLifeTime.min)
    dna.maxDendriteLifeTime.max = max(dna.maxDendriteLifeTime.min+1., dna.maxDendriteLifeTime.max)
    # accuracy_penalty += (dna.maxDendriteLifeTime.min != test[7])
    # accuracy_penalty += (dna.maxDendriteLifeTime.max != test[8])

    dna.maxAxonPointLifeTime.min = max(1., dna.maxAxonPointLifeTime.min)
    dna.maxAxonPointLifeTime.max = max(dna.maxAxonPointLifeTime.min+1., dna.maxAxonPointLifeTime.max)
    # accuracy_penalty += (dna.maxAxonPointLifeTime.min != test[9])
    # accuracy_penalty += (dna.maxAxonPointLifeTime.max != test[10])

    # return accuracy_penalty
    return nothing
end
