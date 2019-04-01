import Base.accumulate!

# SPATIAL UPDATE FUNCTIONS

# VALUE UPDATE FUNCTIONS
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
        else
            N.Q = 0.
        end

        for is in input_valid_syns
            if (is.Q - is.THR) < 0.
                throw("Q - THR < 0 in synaps: $(is.id)")
            end

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

    in_nodes = get_input_nodes_in_all(network_all_cells)
    out_nodes = get_output_nodes_in_all(network_all_cells)
    dens = get_dendrites_in_all(network_all_cells)
    APs = get_axon_points_in_all(network_all_cells)
    syns = get_synapses_in_all(network_all_cells)

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
    println("testing: $dispersion_collection")
    for n_i in n_ind
        # accumulate!(n, dropout(synapses, NN.synapsesAccessDropout), dispersion_collection)
        accm!(NN.components[n_i], get_synapses(syns), dispersion_collection) #all_synapses::Array{Synaps}, dispersion_collection::Dict{Synaps, Pair{FloatN, Integer}}
    end
    println("testing: $dispersion_collection")

    # 3
    # this puts the NT(t-1) and then calculates NT(t)
    # this can be reversed
    if syns != []
        for s in syns
            if s.cell.Q >= s.cell.THR
                s.cell.Q = 0.
            else
                updateQ!(s.cell)
            end
            s.cell.lifeTime -= NN.life_decay
            if s.cell.lifeTime <= 0.
                s = missing
            end

            if !ismissing(s)
                dispersion_value, n = get(dispersion_collection, s.cell, (1, 1))
                avg_NT_change = dispersion_value/n

                # change nt
                s.cell.NT.strength = s.cell.NT.retain_percentage * s.cell.NT.strength + (1-s.cell.NT.retain_percentage) * avg_NT_change
            end
        end
    end

    # 4
    for n_i in n_ind
        NN.components[n_i].lifeTime -= NN.life_decay
        if NN.components[n_i].lifeTime <= 0.
            NN.components[n_i].priors .= missing
            NN.components[n_i].posteriors .= missing
            NN.components[n_i] = missing
        end

        if !ismissing(NN.components[n_i])
            propergate!(NN.components[n_i], den_sinks)
        end
    end

    if dens != []
        for d in skipmissing(dens)
            append!(ap_sinks, [Sink(d.cell.possition, NN.ap_sink_attractive_force)])
        end
    end

    return den_sinks, ap_sinks
end
function state_step!(NN::Network, den_sinks, ap_sinks)
    # update spatial relation
    # - fuse!
    # - split!
    network_all_cells = get_all_all_cells(NN)
    neurons = get_all_neurons(NN)

    in_nodes = get_input_nodes_in_all(network_all_cells)
    out_nodes = get_output_nodes_in_all(network_all_cells)

    for den in get_dendrites_in_all(network_all_cells)
        den.cell.lifeTime -= NN.life_decay
        if den.cell.lifeTime <= 0.
            den = missing
        end

        if !ismissing(den.cell)
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
        end
    end

    for ap in get_axon_points_in_all(network_all_cells)
        # if typeof(ap.cell) == AxonPoint
        if ap.cell.lifeTime <= 0.
            ap = missing
        end

        if !ismissing(ap.cell)
            total_v = [0.,0.,0.]
            for ap_sink in ap_sinks
                dir = direction(ap.cell.possition, ap_sink.possition)
                mag = NN.minFuseDistance / vector_length(dir)
                total_v .+= normalize(dir) .* mag .* (1 + ap_sink.strength)
            end
            total_v ./= length(ap_sinks)

            ap.cell.possition += Possition(total_v...)

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
        end
    end

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
    if typeof(den.cell) != Dendrite || typeof(ap.cell) != AxonPoint; throw("incorect fuse!($(typeof(den.cell)), $(typeof(ap.cell)))"); end
    println(" -- warning -- fusion creates duplicate references in the component list of the network")
    den.cell = to
    ap.cell = to
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

    append!(NN.components, n)
    return n
end
function add_dendrite!(NN::Network, N::Neuron)
    if has_empty_prior(N)
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                d = AllCell(unfold(rand(NN.dna_stack.den_dna_samples)))
                rectify_possition!(d, NN.size)

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
                rectify_possition!(ap, NN.size)

                N.posteriors[i] = ap
                append!(NN.components, [ap])
                return nothing
            end
        end
    end
end
function populate_network!(NN::Network, num_neurons::Integer, max_num_priors::Integer, max_num_post::Integer)
    for _ in 1:num_neurons
        # add_neuron! adds it to component list
        n = add_neuron!(NN, random(NN.dna_stack.n_dna_samples))

        for _ in 1:rand(1:max_num_priors)
            add_dendrite!(NN, n)
        end
        for _ in 1:rand(1:max_num_post)
            add_axon_point!(NN, n)
        end
    end
end


# FITNESS FUNCTIONS
function update_fitness!(s::Synaps, activated::Bool, fitness_decay::FloatN)
    s.fitness *= fitness_decay
    if activated
        s.total_fitness += 1.
        s.fitness += 1.
    end
end
# implement for neuron

# implement for network

# ...


# VERIFICATION FUNCTIONS
function rectify_possition!(c, nn_size::FloatN)
    if distance(c.possition, Possition(0,0,0)) > nn_size
        c.possition = normalize(c.possition) * nn_size
    end
end
function clean_network_components!(NN)
    NN.components = collect(skipmissing(NN.components))
    all_c = NN.components

    for n1 in eachindex(all_c)
        if typeof(all_c[n1]) == AllCell
            if typeof(all_c[n1].cell) == Synaps
                for n2 in eachindex(all_c)
                    if typeof(all_c[n2]) == AllCell
                        if typeof(all_c[n2].cell) == Synaps
                            if n1 != n2
                                if all_c[n1].cell === all_c[n2].cell
                                    all_c[n2] = missing
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
function rectifyDNA!(dna::DendriteDNA, NN::Network)
    clamp!(dna.LifeTime.min, 1., NN.maxDendriteLifeTime-1.)
    clamp!(dna.LifeTime.max, dna.LifeTime.min, NN.maxDendriteLifeTime)

    clamp!(dna.max_length.min, 1., (NN.size/2)-1)
    clamp!(dna.max_length.max, dna_max_length.min+0.01, NN.size/2) # 0.01 for max > min

    # init possition does not have to be rectified
    # because its clamped at initialization time
end
function rectifyDNA!(dna::AxonPointDNA, NN::Network);
    clamp!(dna.LifeTime.min, 1., NN.maxDendriteLifeTime-1.)
    clamp!(dna.LifeTime.max, dna.LifeTime.min, NN.maxDendriteLifeTime)

    clamp!(dna.max_length.min, 1., (NN.size/2)-1)
    clamp!(dna.max_length.max, dna_max_length.min+0.01, NN.size/2) # 0.01 for max > min

    # init possition does not have to be rectified
    # because its clamped at initialization time
end
function rectifyDNA!(dna::NeuroTransmitterDNA, NN::Network)
    dna.init_strength.min = max(dna.init_strength.min, 1.)
    dna.init_strength.max = max(dna.init_strength.max, dna.init_strength.min+0.01)

    clamp!(dna.dispersion_strength_scale.min, 0.1, NN.max_nt_dispersion_strength_scale-0.1)
    clamp!(dna.dispersion_strength_scale.max, dna.dispersion_strength_scale.min+0.01, NN.max_nt_dispersion_strength_scale)

    clamp!(dna.retain_percentage, 0, 1)

    # dispersion region does not have to be rectified
    # because its clamped at initialization time
end
function rectifyDNA!(dna::SynapsDNA, NN::Network; max_q_decay=0.1)
    clamp!(dna.LifeTime.min, 1., NN.maxSynapsLifeTime-1.)
    clamp!(dna.LifeTime.max, dna.LifeTime.min+0.01, NN.maxSynapsLifeTime)

    clamp!(dna.QDecay.min, max_q_decay, 0.97)
    clamp!(dna.QDecay.max, dna.QDecay.min+0.01, 0.99)

    clamp!(dna.THR.min, 0.1, NN.max_threshold-0.1)
    clamp!(dna.THR.max, dna.THR.min+0.01, NN.max_threshold)

    rectifyDNA!(dna.NT)
end
function rectifyDNA!(dna::NeuronDNA, NN::Network)
    clamp!(dna.lifeTime.min, 1., NN.maxNeuronLifeTime-1.)
    clamp!(dna.lifeTime.max, dna.LifeTime.min+0.1, NN.maxNeuronLifeTime)

    dna.max_num_priors.min = max(dna.max_num_priors.min, 1)
    dna.max_num_priors.max = max(dna.max_num_priors.max, dna.max_num_priors.min+1)
    
    dna.max_num_posteriors.min = max(dna.max_num_posteriors.min, 1)
    dna.max_num_posteriors.max = max(dna.max_num_posteriors.max, dna.max_num_posteriors.min+1)

    # init possition does not have to be rectified
    # because its clamped at initialization time
end
function rectifyDNA!(NDNA::NetworkDNA)

end
function rectifyInitializationPossition!(pos::InitializationPossition, network_max_range::FloatN)

end
