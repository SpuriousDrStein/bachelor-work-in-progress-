function update_neuron!(N::Neuron, NN::Network, dispersion_collection::Dict)
    prior_all = get_prior_all_cells(N)
    input_syns = [s for s in get_synapses(prior_all) if s.Q >= s.THR]
    input_nodes = get_input_nodes(prior_all)

    if input_syns != [] || input_nodes != []

        input = sum([[i_n.value for i_n in input_nodes]..., [s.Q * s.NT.strength for s in input_syns]...])

        N.Q = input

        N.d_total_fitness = (N.d_total_fitness + length(input_syns) + 1) / 2 # 1 for beeing activated

        for is in input_syns
            for s in get_synapses(Subnet(is.position, is.Q/NN.size), get_synapses(get_all_all_cells(NN)))
                if s in keys(dispersion_collection)
                    dispersion_collection[s] = dispersion_collection[s] .+ (s.NT.strength, 1)
                else
                    dispersion_collection[s] = (s.NT.strength, 1)
                end
            end
        end
    else
        N.d_total_fitness /= 2
    end
    N.total_fitness += N.d_total_fitness
end
function update_synaps!(s::AllCell, NN::Network, dispersion_collection)
    if s.cell.Q >= s.cell.THR
        s.cell.Q = 0.
        s.cell.R = 0.01
        s.cell.d_total_fitness += 1
    else
        s.cell.Q *= s.cell.R
        s.cell.R = min(s.cell.R * s.cell.RRecovery, s.cell.maxR)
    end

    # change nt
    dispersion_value, n = get(dispersion_collection, s.cell, (1, 1))
    avg_NT_change = dispersion_value/n
    s.cell.NT.strength = NN.nt_retain_percentage * s.cell.NT.strength + (1-NN.nt_retain_percentage) * avg_NT_change

    # update syn-fitness
    # s.cell.d_total_fitness += ((avg_NT_change+s.cell.NT.strength)/2)/(abs(avg_NT_change-s.cell.NT.strength)+0.00001)
    s.cell.d_total_fitness /= 2
    s.cell.total_fitness += s.cell.d_total_fitness

    if s.cell.d_total_fitness <= s.cell.destruction_threshold
        s.cell.lifeTime -= NN.heavy_life_decay
    else
        s.cell.lifeTime -= NN.light_life_decay
    end
end
function propergate!(N::Neuron, NN::Network, den_sinks::Array, ap_surges::Array)
    post_all = get_posterior_all_cells(N)

    if N.Q >= N.THR
        for a in post_all
            if typeof(a.cell) == Synaps
                a.cell.Q += N.Q/length(post_all)
            elseif typeof(a.cell) == AxonPoint
                append!(den_sinks, [Sink(copy(a.cell.position), N.Q/length(post_all))])
                append!(ap_surges, [Surge(copy(a.cell.position), NN.ap_surge_repulsive_force)])
            elseif typeof(a.cell) == OutputNode
                a.cell.value = N.Q/length(post_all)
                append!(ap_surges, [Surge(copy(a.cell.position), NN.ap_surge_repulsive_force)])
            end
        end
    else
        N.Q = 0.

        # THIS IS A TEST TO SEE IF RESETTING (WHEN FAILING TO ACTIVATE) THE NEURON IS VIABLE
    end
end

function value_step!(NN::Network, input::Array)
    network_all_cells = get_all_all_cells(NN)
    n_ind = get_all_neuron_indecies(NN)
    in_nodes = get_input_nodes_in_all(NN)
    out_nodes = get_output_nodes_in_all(NN)

    dispersion_collection = Dict()
    den_sinks = []
    den_surges = [Surge(copy(o_n.cell.position), NN.den_surge_repulsive_force) for o_n in out_nodes]
    ap_sinks = []
    ap_surges = [Surge(copy(i_n.cell.position), NN.ap_surge_repulsive_force) for i_n in in_nodes]

    # if no neurons in network
    if get_all_neurons(NN) == []
        NN.total_fitness -= 10
        return [], [], [], []
    end

    # 1: assign input nodes, in/out surges and in/out sinks
    if network_all_cells != []
        for i in eachindex(in_nodes, input)
            in_nodes[i].cell.value = input[i]
            if in_nodes[i] in get_input_nodes(get_all_all_cells(NN))
                append!(den_surges, [Surge(copy(in_nodes[i].cell.position), NN.den_surge_repulsive_force)])
            else
                append!(den_sinks, [Sink(copy(in_nodes[i].cell.position), in_nodes[i].cell.value)])
            end
        end
        for i in eachindex(out_nodes)
            if out_nodes[i] in get_output_nodes(get_all_all_cells(NN))
                append!(ap_surges, [Surge(copy(out_nodes[i].cell.position), NN.ap_surge_repulsive_force)])
            else
                append!(ap_sinks, [Sink(copy(out_nodes[i].cell.position), NN.ap_sink_attractive_force)])
            end
        end


        if get_synapses(network_all_cells) != []
            # 2: update neuron Q
            for n_i in n_ind
                update_neuron!(NN.components[n_i], NN, dispersion_collection) #all_synapses::Array{Synaps}, dispersion_collection::Dict{Synaps, Pair{FloatN, Integer}}
            end

            # 3: update synapses based on their activation state (I/O)
            for s in get_synapses_in_all(network_all_cells)
                update_synaps!(s, NN, dispersion_collection)
            end
        end

        # append sinks and surges for dendrite end connections
        if get_dendrites_in_all(network_all_cells) != []
            for d in get_dendrites_in_all(network_all_cells)
                append!(ap_sinks, [Sink(copy(d.cell.position), NN.ap_sink_attractive_force)])
                append!(den_surges, [Surge(copy(d.cell.position), NN.den_surge_repulsive_force)])
            end
        end
    end

    # 4 propergate for each activatable neurons
    for n_i in n_ind
        propergate!(NN.components[n_i], NN, den_sinks, ap_surges)

        if NN.components[n_i].d_total_fitness <= NN.components[n_i].destruction_threshold
            NN.components[n_i].lifeTime -= NN.heavy_life_decay
        else
            NN.components[n_i].lifeTime -= NN.light_life_decay
        end
    end
    return den_sinks, den_surges, ap_sinks, ap_surges
end


function state_step!(NN::Network, den_sinks, den_surges, ap_sinks, ap_surges)
    # update spatial relation
    network_all_cells = get_all_all_cells(NN)
    neurons = get_all_neurons(NN)
    out_nodes = get_output_nodes_in_all(NN)

    if get_all_neurons(NN) == []
        # println("exit state step because no neurons exist")
        return nothing
    end

    if network_all_cells != []
        for den in get_dendrites_in_all(network_all_cells)
            total_v = [0.,0.,0.]
            for d_sink in den_sinks
                dir = direction(den.cell.position, d_sink.position)

                if dir != [0,0,0]
                    mag = NN.minFuseDistance
                    if vector_length(dir) > NN.minFuseDistance # this is to avoid heavy overshooting
                        mag = NN.minFuseDistance / vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + d_sink.strength)
                end
            end
            for d_surge in den_surges
                dir = direction(d_surge.position, den.cell.position)

                if dir != [0,0,0]
                    mag = NN.minFuseDistance
                    if vector_length(dir) > NN.minFuseDistance
                        mag /= vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + d_surge.strength)
                end
            end

            total_v ./= (length(den_sinks) + length(den_surges))
            rand_v  = get_random_position(1) * NN.random_fluctuation_scale
            den.cell.position += rand_v
            den.cell.position += Position(total_v...)

            den.cell.lifeTime -= NN.light_life_decay

            for i_n in [inn for inn in get_input_nodes_in_all(NN) if !(inn in get_input_nodes(get_all_all_cells(NN)))]
                if distance(den.cell.position, i_n.cell.position) <= NN.minFuseDistance
                    fuse!(i_n, den)
                end
            end
        end

        for ap in get_axon_points_in_all(network_all_cells)
            total_v = [0.,0.,0.]
            for ap_sink in ap_sinks
                dir = direction(ap.cell.position, ap_sink.position)

                if dir != [0,0,0]
                    mag = NN.minFuseDistance
                    if vector_length(dir) > NN.minFuseDistance
                        mag /= vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + ap_sink.strength)
                end
            end

            for ap_surge in ap_surges
                dir = direction(ap_surge.position, ap.cell.position)

                if dir != [0,0,0]
                    mag = NN.minFuseDistance
                    if vector_length(dir) > NN.minFuseDistance
                        mag /= vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + ap_surge.strength)
                end
            end

            total_v ./= (length(ap_sinks) + length(ap_surges))
            rand_v  = get_random_position(1) * NN.random_fluctuation_scale
            ap.cell.position += rand_v
            ap.cell.position += Position(total_v...)

            ap.cell.lifeTime -= NN.light_life_decay

            # fuse with dendrite if near
            for d in get_dendrites_in_all(network_all_cells)
                if distance(ap.cell.position, d.cell.position) <= NN.minFuseDistance

                    half_dist = direction(ap.cell.position, d.cell.position) ./ 2
                    pos = ap.cell.position + Position(half_dist...)

                    nt = unfold(rand(NN.dna_stack.nt_dna_samples), NN)

                    fuse!(d, ap, unfold(rand(NN.dna_stack.syn_dna_samples), pos, nt, NN))
                    NN.syn_counter += 1
                end
            end
            for o_n in [onn for onn in get_output_nodes_in_all(NN) if !(onn in get_output_nodes(get_all_all_cells(NN)))]
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
function tally_up_fitness!(NN::Network)
    for n in get_all_neurons(NN)
        p_syns = get_synapses(get_prior_all_cells(n))
        if p_syns != []
            for s in p_syns
                if s.Q >= s.THR
                    s.total_fitness += 1
                end
                n.total_fitness += s.total_fitness
            end
        end
        NN.total_fitness += n.total_fitness
    end
end



# VERIFICATION FUNCTIONS
function clean_network_components!(NN::Network)
    NN.components = collect(skipmissing(NN.components))

    for n1 in eachindex(NN.components)

        # lifetime test
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
