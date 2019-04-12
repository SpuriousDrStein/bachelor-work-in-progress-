function updateQ!(syn::Synaps)
    syn.Q *= syn.QDecay
    syn.Q *= syn.NT.strength
end

function accm!(N::Neuron, all_synapses::Array, dispersion_collection::Dict, fitness_decay)
    N.Q = 0.

    prior_all = get_prior_all_cells(N)
    input_syns = get_synapses(prior_all)

    if input_syns != []
        input_valid_syns = get_activatable_synapses(input_syns)
        input_nodes = get_input_nodes(prior_all)

        input_v = [[s.THR for s in input_valid_syns]..., [i_n.value for i_n in input_nodes]...]

        if input_v != []
            N.Q = sum(input_v)

            # calculate new fitness values
            N.total_fitness += length(input_syns)
            N.total_fitness += sum(input_v)
            N.total_fitness *= fitness_decay
        else
            N.Q = 0.
            N.total_fitness *= fitness_decay
        end

        for is in input_valid_syns
            for s in get_synapses(Subnet(is.position, (is.Q - is.THR)), all_synapses)
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
    post_out = get_output_nodes(post_all)

    for s in post_syns
        s.Q += N.Q/length(post_all)
    end
    for ap in post_aps
        append!(sink_list, [Sink(ap.position, N.Q/length(post_all))])
    end
    for p_o in post_out
        p_o.value = N.Q/length(post_all)
    end
end

function value_step!(NN::Network, input::Array)
    network_all_cells = get_all_all_cells(NN)
    n_ind = get_all_neuron_indecies(NN)
    in_nodes = get_input_nodes_in_all(NN)
    out_nodes = get_output_nodes_in_all(NN)

    dispersion_collection = Dict()
    den_sinks = []
    ap_sinks = [Sink(o_n.cell.position, NN.ap_sink_attractive_force) for o_n in out_nodes] # array of ap sinks

    # 1
    for i in eachindex(in_nodes, input)
        in_nodes[i].cell.value = input[i]
        append!(den_sinks, [Sink(in_nodes[i].cell.position, in_nodes[i].cell.value)])
    end

    if get_all_neurons(NN) == []
        # println("exit value step because no neurons exist")
        NN.total_fitness -= 10
        return [], []
    end

    if network_all_cells != []
        if get_synapses(network_all_cells) != []
            # 2
            for n_i in n_ind
                accm!(NN.components[n_i], get_synapses(network_all_cells), dispersion_collection, NN.fitness_decay) #all_synapses::Array{Synaps}, dispersion_collection::Dict{Synaps, Pair{FloatN, Integer}}
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
        end

        if get_dendrites_in_all(network_all_cells) != []
            for d in get_dendrites_in_all(network_all_cells)
                append!(ap_sinks, [Sink(d.cell.position, NN.ap_sink_attractive_force)])
            end
        end
    end

    for n_i in n_ind
        propergate!(NN.components[n_i], den_sinks)
        NN.components[n_i].lifeTime -= NN.life_decay
    end

    return den_sinks, ap_sinks
end

function state_step!(NN::Network, den_sinks, ap_sinks)
    # update spatial relation
    network_all_cells = get_all_all_cells(NN)
    neurons = get_all_neurons(NN)
    out_nodes = get_output_nodes_in_all(NN)

    if get_all_neurons(NN) == []
        # println("exit state step because no neurons exist")
        return [], []
    end

    if network_all_cells != []
        for den in get_dendrites_in_all(network_all_cells)
            total_v = [0.,0.,0.]
            for d_sink in den_sinks
                dir = direction(den.cell.position, d_sink.position)
                mag = NN.minFuseDistance / vector_length(dir)
                total_v .+= normalize(dir) .* mag .* (1 + d_sink.strength)
            end
            total_v ./= length(den_sinks)
            rand_v  = get_random_position(1) * NN.random_fluctuation_scale
            den.cell.position += Position(total_v...) + rand_v

            den.cell.lifeTime -= NN.life_decay

            for i_n in get_input_nodes_in_all(NN)
                if distance(den.cell.position, i_n.cell.position) <= NN.minFuseDistance
                    fuse!(i_n, den)
                end
            end
        end

        for ap in get_axon_points_in_all(network_all_cells)
            total_v = [0.,0.,0.]
            for ap_sink in ap_sinks
                dir = direction(ap.cell.position, ap_sink.position)
                mag = NN.minFuseDistance / vector_length(dir)
                total_v .+= normalize(dir) .* mag .* (1 + ap_sink.strength)
            end
            total_v ./= length(ap_sinks)
            rand_v  = get_random_position(1) * NN.random_fluctuation_scale
            ap.cell.position += Position(total_v...) + rand_v

            ap.cell.lifeTime -= NN.life_decay

            # fuse with dendrite if near
            for d in get_dendrites_in_all(network_all_cells)
                if distance(ap.cell.position, d.cell.position) <= NN.minFuseDistance

                    half_dist = direction(ap.cell.position, d.cell.position) ./ 2
                    pos = ap.cell.position + Position(half_dist...)

                    nt = unfold(rand(NN.dna_stack.nt_dna_samples), NN)

                    fuse!(d, ap, unfold(rand(NN.dna_stack.syn_dna_samples), pos, nt, NN))
                    NN.s_id_counter += 1
                end
            end
            for o_n in get_output_nodes_in_all(NN)
                if distance(ap.cell.position, o_n.cell.position) <= NN.minFuseDistance
                    fuse!(o_n, ap)
                end
            end

        end
    end

    # repel neurons away each other
    if length(neurons) > 1
        total_v = [[0.,0.,0.] for _ in 1:length(neurons)]
        for i in eachindex(neurons)
            local_v = [0.,0.,0.]

            for n in neurons
                if neurons[i] !== n
                     local_v += direction(n.position, neurons[i].position)
                end
            end

            total_v[i] = normalize(local_v) .* NN.neuron_repel_force #./ (length(neurons) .- 1)
        end
        for i in eachindex(total_v, neurons)
            rand_v  = get_random_position(1) * NN.random_fluctuation_scale
            neurons[i].position += Position(total_v[i]...) + rand_v
        end
    end
end

# STRUCTURE GENERATION FUNCTIONS
function fuse!(den::AllCell, ap::AllCell, to::Synaps)
    if typeof(den.cell) != Dendrite || typeof(ap.cell) != AxonPoint
        # throw("incorect fuse!($(typeof(den.cell)), $(typeof(ap.cell)))")
    else
        den.cell = to
        ap.cell = to
    end
end
function fuse!(network_node::AllCell, ac::AllCell)
    if typeof(network_node.cell) != InputNode && typeof(network_node.cell) != OutputNode
        throw("incorect fuse!($(typeof(ac.cell)), $(typeof(network_node.cell)))")
    else
        if typeof(ac.cell) == AxonPoint
            ac.cell = network_node.cell
        elseif typeof(ac.cell) == Dendrite
            ac.cell = network_node.cell
        end
    end
end
function add_neuron!(NN::Network)
    n = unfold(rand(NN.dna_stack.n_dna_samples), get_random_position(NN.size), NN)
    NN.n_id_counter += 1
    NN.n_counter += 1

    append!(NN.components, [n])
    return n
end
function add_dendrite!(NN::Network, N::Neuron)
    if has_empty_prior(N)
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                d = AllCell(unfold(rand(NN.dna_stack.den_dna_samples), N.position + get_random_position(N.den_and_ap_init_range), NN))
                d.cell.position = rectify_position(d.cell.position, NN.size)

                N.priors[i] = d
                append!(NN.components, [d])
                NN.den_counter += 1
                return nothing
            end
        end
    end
end
function add_axon_point!(NN::Network, N::Neuron)
    if has_empty_post(N)
        for i in eachindex(N.posteriors)
            if ismissing(N.posteriors[i])
                ap = AllCell(unfold(rand(NN.dna_stack.ap_dna_samples), N.position + get_random_position(N.den_and_ap_init_range), NN))
                ap.cell.position = rectify_position(ap.cell.position, NN.size)

                N.posteriors[i] = ap
                append!(NN.components, [ap])
                NN.ap_counter += 1
                return nothing
            end
        end
    end
end
function runtime_instantiate_components!(NN::Network, iteration::Integer) # instantiate neurons, dendrites and ap's depending on iteration
    for n in get_all_neurons(NN)
        if iteration % n.den_init_interval == 0
            add_dendrite!(NN, n)
        end
        if iteration % n.ap_init_interval == 0
            add_axon_point!(NN, n)
        end
    end
    if iteration % NN.neuron_init_interval == 0
        add_neuron!(NN)
    end
end
function populate_network!(NN::Network, num_neurons::Integer, max_num_priors::Integer, max_num_post::Integer)
    for _ in 1:num_neurons
        # add_neuron! adds it to component list as well
        n = add_neuron!(NN)

        for _ in 1:rand(1:max_num_priors)
            add_dendrite!(NN, n)
        end
        for _ in 1:rand(1:max_num_post)
            add_axon_point!(NN, n)
        end
    end
end
function reset_network_components!(NN::Network)
    if get_all_neurons(NN) != []
        for n in get_all_neurons(NN)
            n.Q = 0.
        end
    end
    if get_all_all_cells(NN) != []
        for s in get_synapses(get_all_all_cells(NN))
            s.Q = 0.
        end
    end
end


# VERIFICATION FUNCTIONS
function clean_network_components!(NN::Network)
    NN.components = collect(skipmissing(NN.components))

    # println("implement - keep position in network, behaviour")
    # println("implement - keep position max range, behaviour")

    for n1 in eachindex(NN.components)

        if typeof(NN.components[n1]) == AllCell && typeof(NN.components[n1].cell) != InputNode && typeof(NN.components[n1].cell) != OutputNode
            if NN.components[n1].cell.lifeTime <= 0
                if typeof(NN.components[n1]) == Synaps
                    NN.total_fitness += NN.components[n1].cell.total_fitness
                end
                NN.components[n1] = missing
            end

            if !ismissing(NN.components[n1])
                # update position to be inside network
                if vector_length(NN.components[n1].cell.position) > NN.size
                    NN.components[n1].cell.position = Position((normalize(NN.components[n1].cell.position) .* NN.size)...)
                end
                # update position to be inside max_range
                if typeof(NN.components[n1].cell) != Synaps
                    if vector_length(NN.components[n1].cell.position) > NN.components[n1].cell.max_length
                        NN.components[n1].cell.position = Position((normalize(NN.components[n1].cell.position) .* NN.components[n1].cell.max_length)...)
                    end
                else #if typeof(NN.components[n1].cell) == Synaps
                    # remove duplicates in NN.components
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

function rectify_position(p::Position, nn_size::FloatN)
    if distance(p, Position(0,0,0)) > nn_size
        return Position((normalize(p) * nn_size)...)
    else
        return p
    end
end
function rectifyDNA!(dna::DendriteDNA, NN::Network)
    dna.lifeTime = clamp(dna.lifeTime, 1., NN.maxDendriteLifeTime)
    dna.max_length = max(1., dna.max_length)
end
function rectifyDNA!(dna::AxonPointDNA, NN::Network);
    dna.lifeTime = clamp(dna.lifeTime, 1., NN.maxDendriteLifeTime)
    dna.max_length = max(1., dna.max_length)
end
function rectifyDNA!(dna::NeuroTransmitterDNA, NN::Network)
    dna.init_strength = clamp(dna.init_strength, 0.5, NN.max_nt_strength) # 0.5 because everything below negates more than 50% of input
    dna.retain_percentage = clamp(dna.retain_percentage, 0, 1)
end
function rectifyDNA!(dna::SynapsDNA, NN::Network)
    dna.lifeTime = clamp(dna.lifeTime, 1., NN.maxSynapsLifeTime)
    dna.QDecay = clamp(dna.QDecay, 0.1, 0.99)
    dna.THR = clamp(dna.THR, 0.1, NN.max_threshold)
end
function rectifyDNA!(dna::NeuronDNA, NN::Network)
    dna.lifeTime = clamp(dna.lifeTime, 1., NN.maxNeuronLifeTime)
    dna.max_num_priors = max(1, dna.max_num_priors)
    dna.max_num_posteriors = max(1, dna.max_num_posteriors)
    dna.den_and_ap_init_range = max(1., dna.den_and_ap_init_range)
    dna.den_init_interval = max(1, dna.den_init_interval)
    dna.ap_init_interval = max(1, dna.ap_init_interval)
end
function rectifyDNA!(dna::DNAStack, NN::Network)
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
end
function rectifyDNA!(dna::NetworkDNA)
    dna.ap_sink_force = max(0.1, dna.ap_sink_force)
    dna.neuron_repel_force = max(0.01, dna.neuron_repel_force)
end
