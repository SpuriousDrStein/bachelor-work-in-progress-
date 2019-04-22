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
            end
        end
    else
        N.Q = 0.
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
        # NN.total_fitness -= 10
        return [], [], [], []
    end

    # 1: assign input nodes, in/out surges and in/out sinks
    if network_all_cells != []
        for i in eachindex(in_nodes, input)
            in_nodes[i].cell.value = input[i]
            append!(den_sinks, [Sink(copy(in_nodes[i].cell.position), NN.input_attraction_force)])
        end
        for i in eachindex(out_nodes)
            append!(ap_sinks, [Sink(copy(out_nodes[i].cell.position), NN.output_attraction_force)])
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
    in_nodes = get_input_nodes_in_all(NN)
    out_nodes = get_output_nodes_in_all(NN)

    if get_all_neurons(NN) == []
        # println("exit state step because no neurons exist")
        return nothing
    end

    if network_all_cells != []
        # input node position update
        # for i_nn in in_nodes
        #     if !i_nn.cell.referenced
        #         total_v = [0.,0.]
        #         for ap_sink in ap_sinks
        #             dir = direction(i_nn.cell.position, ap_sink.position)
        #
        #             if dir != [0,0]
        #                 if vector_length(dir) < NN.size/4
        #                     mag = 0
        #                 else
        #                     mag = NN.min_fuse_distance / vector_length(dir)
        #                 end
        #
        #                 total_v .+= normalize(dir) .* mag .* (1 + ap_sink.strength)
        #             end
        #         end
        #         for i_nn2 in in_nodes
        #             if i_nn2 !== i_nn
        #                 dir = direction(i_nn2.cell.position, i_nn.cell.position)
        #                 mag = min(NN.min_fuse_distance/vector_length(dir), NN.min_fuse_distance)
        #                 total_v .+= normalize(dir) .* mag
        #             end
        #         end
        #
        #         total_v ./= (length(ap_sinks) + length(in_nodes) - 1)
        #         i_nn.cell.position += Position(total_v...)
        #     end
        # end

        # output node position update
        # for o_nn in out_nodes
        #     if !o_nn.cell.referenced
        #         total_v = [0.,0.]
        #         for d_sink in den_sinks
        #             dir = direction(o_nn.cell.position, d_sink.position)
        #
        #             if dir != [0,0]
        #                 if vector_length(dir) < NN.size/3
        #                     mag = 0
        #                 else
        #                     mag = NN.min_fuse_distance / vector_length(dir)
        #                 end
        #
        #                 total_v .+= normalize(dir) .* mag .* (1 + d_sink.strength)
        #             end
        #         end
        #         for o_nn2 in get_output_nodes_in_all(NN)
        #             if o_nn2 !== o_nn
        #                 dir = direction(o_nn2.cell.position, o_nn.cell.position)
        #                 mag = min(NN.min_fuse_distance/vector_length(dir), NN.min_fuse_distance)
        #                 total_v .+= normalize(dir) .* mag
        #             end
        #         end
        #
        #         total_v ./= (length(den_sinks) + length(out_nodes) - 1)
        #         o_nn.cell.position += Position(total_v...)
        #     end
        # end

        # dendrite position update
        for den in get_dendrites_in_all(network_all_cells)
            total_v = [0.,0.]
            for d_sink in den_sinks
                dir = direction(den.cell.position, d_sink.position)

                if dir != [0,0]
                    mag = NN.min_fuse_distance
                    if vector_length(dir) > NN.min_fuse_distance # this is to avoid heavy overshooting
                        mag /= vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + d_sink.strength)
                end
            end
            for d_surge in den_surges
                dir = direction(d_surge.position, den.cell.position)

                if dir != [0,0]
                    mag = NN.min_fuse_distance
                    if vector_length(dir) > NN.min_fuse_distance
                        mag /= vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + d_surge.strength)
                end
            end

            total_v ./= (length(den_sinks) + length(den_surges))
            den.cell.position += Position(total_v...)

            den.cell.lifeTime -= NN.light_life_decay

            for i_n in get_input_nodes_in_all(NN)
                if distance(den.cell.position, i_n.cell.position) <= NN.min_fuse_distance
                    fuse!(i_n, den)
                end
            end
        end

        # axon point position update
        for ap in get_axon_points_in_all(network_all_cells)
            total_v = [0.,0.]
            for ap_sink in ap_sinks
                dir = direction(ap.cell.position, ap_sink.position)

                if dir != [0,0]
                    mag = NN.min_fuse_distance
                    if vector_length(dir) > NN.min_fuse_distance
                        mag /= vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + ap_sink.strength)
                end
            end

            for ap_surge in ap_surges
                dir = direction(ap_surge.position, ap.cell.position)

                if dir != [0,0]
                    mag = NN.min_fuse_distance
                    if vector_length(dir) > NN.min_fuse_distance
                        mag /= vector_length(dir)
                    end

                    total_v .+= normalize(dir) .* mag .* (1 + ap_surge.strength)
                end
            end

            total_v ./= (length(ap_sinks) + length(ap_surges))
            ap.cell.position += Position(total_v...)

            ap.cell.lifeTime -= NN.light_life_decay

            # fuse with dendrite if near
            for d in get_dendrites_in_all(network_all_cells)
                if distance(ap.cell.position, d.cell.position) <= NN.min_fuse_distance

                    half_dist = direction(ap.cell.position, d.cell.position) ./ 2
                    pos = ap.cell.position + Position(half_dist...)

                    nt = unfold(rand(NN.dna_stack.nt_dna_samples), NN)

                    fuse!(d, ap, unfold(rand(NN.dna_stack.syn_dna_samples), pos, nt, NN))
                    NN.syn_counter += 1
                end
            end
            for o_n in get_output_nodes_in_all(NN)
                if distance(ap.cell.position, o_n.cell.position) <= NN.min_fuse_distance
                    fuse!(o_n, ap)
                end
            end
        end
    end

    # neuron position update
    # repel neurons away each other
    # if length(neurons) > 1
    #     total_v = [[0.,0.] for _ in 1:length(neurons)]
    #     for i in eachindex(neurons)
    #         local_v = [0.,0.]
    #
    #         for n in neurons
    #             mag = min(NN.min_fuse_distance/distance(n.position, neurons[i].position), NN.min_fuse_distance)
    #             if neurons[i] !== n
    #                  local_v += normalize(direction(n.position, neurons[i].position)) .* mag
    #             end
    #         end
    #
    #         total_v[i] = local_v./ length(neurons) .- 1
    #     end
    #     for i in eachindex(total_v, neurons)
    #         rand_v  = get_random_position(1) * NN.random_fluctuation_scale
    #         neurons[i].position += Position(total_v[i]...) + rand_v
    #     end
    # end
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
    elseif typeof(network_node.cell) == InputNode && typeof(ac.cell) == AxonPoint
        throw("WHAT ARE YOU DOIN")
    elseif typeof(network_node.cell) == OutputNode && typeof(ac.cell) == Dendrite
        throw("WHAT ARE YOU DOIN 2")
    else
        if typeof(ac.cell) == AxonPoint
            ac.cell = network_node.cell
        elseif typeof(ac.cell) == Dendrite
            ac.cell = network_node.cell
        end
        network_node.cell.referenced = true
    end
end
function add_neuron!(NN::Network)
    n = unfold(rand(NN.dna_stack.n_dna_samples), get_random_position(NN.size), NN)
    NN.n_counter += 1

    append!(NN.components, [n])
    return n
end
function add_neuron!(NN::Network, init_pos::Position)
    n = unfold(rand(NN.dna_stack.n_dna_samples), init_pos, NN)
    NN.n_counter += 1

    append!(NN.components, [n])
    return n
end
function add_dendrite!(NN::Network, N::Neuron)
    if has_empty_prior(N)
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                d = AllCell(Dendrite(NN.dendrite_lifetime, N.position + get_random_position(NN.den_and_ap_init_range)))
                d.cell.position = rectify_position(d.cell.position, NN.size)

                N.priors[i] = d
                append!(NN.components, [d])
                NN.den_counter += 1
                return nothing
            end
        end
    end
end
function add_dendrite!(NN::Network, N::Neuron, init_pos::Position)
    if has_empty_prior(N)
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                d = AllCell(Dendrite(NN.dendrite_lifetime, N.position + init_pos))

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
                ap = AllCell(AxonPoint(NN.axon_point_lifetime, N.position + get_random_position(NN.den_and_ap_init_range)))
                ap.cell.position = rectify_position(ap.cell.position, NN.size)

                N.posteriors[i] = ap
                append!(NN.components, [ap])
                NN.ap_counter += 1
                return nothing
            end
        end
    end
end
function add_axon_point!(NN::Network, N::Neuron, init_pos::Position)
    if has_empty_post(N)
        for i in eachindex(N.posteriors)
            if ismissing(N.posteriors[i])
                ap = AllCell(AxonPoint(NN.axon_point_lifetime, N.position + init_pos))

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
function populate_network!(NN::Network, num_neurons::Integer, num_priors::Integer, num_post::Integer, init_pos)
    for n in 1:num_priors+num_post:num_neurons*(num_priors+num_post)
        # add_neuron! adds it to component list as well
        cn = add_neuron!(NN, init_pos[n])

        for i in 1:num_priors
            add_dendrite!(NN, cn, init_pos[n+i])
        end
        for j in 1:num_post
            add_axon_point!(NN, cn, init_pos[n+num_priors+j])
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

                # remove duplcate dens and aps in NN.components
                if typeof(NN.components[n1].cell) != Synaps
                    for n2 in eachindex(NN.components)
                        if typeof(NN.components[n2]) == AllCell && typeof(NN.components[n2].cell) != Synaps && typeof(NN.components[n2].cell) != InputNode && typeof(NN.components[n2].cell) != OutputNode
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
    end
    for nid in get_all_neuron_indecies(NN)
        if NN.components[nid].lifeTime <= 0
            NN.total_fitness += NN.components[nid].total_fitness
            for pri in skipmissing(get_prior_all_cells(NN.components[nid]))
                if typeof(pri.cell) == InputNode
                    pri.cell.referenced = false
                else
                    pri = missing
                end
            end
            for post in skipmissing(get_posterior_all_cells(NN.components[nid]))
                if typeof(post.cell) == OutputNode
                    post.cell.referenced = false
                else
                    post = missing
                end
            end
            NN.components[nid] = missing
        else
            if vector_length(NN.components[nid].position) > NN.size
                NN.components[nid].position = Position((normalize(NN.components[nid].position) .* NN.size)...)
            end
        end
    end
    for io_comp in NN.IO_components
        # update position to be inside network
        if vector_length(io_comp.cell.position) > NN.size
            io_comp.cell.position = Position((normalize(io_comp.cell.position) .* NN.size)...)
        end
    end
    NN.components = collect(skipmissing(NN.components))
end
