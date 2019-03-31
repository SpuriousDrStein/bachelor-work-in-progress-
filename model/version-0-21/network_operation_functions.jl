import Base.accumulate!

# SPATIAL UPDATE FUNCTIONS
function remove!(syn::Synaps);                      syn = missing; end
function remove!(den::Dendrite);                    den = missing; end
function remove!(ao::AxonPoint);                    ap = missing; end
function remove!(n::Neuron)
    n.posteriors .= missing
    n.priors .= missing
    n = missing
end

# VALUE UPDATE FUNCTIONS
function decay_charge!(syn::Synaps)
    syn.Q *= syn.QDecay
end
function decay_life!(syn::Synaps, Δlife::Integer)
    syn.lifeTime -= Δlife
    if syn.lifeTime <= 0
        remove!(syn)
    end
end
function decay_life!(den::Dendrite, Δlife::Integer)
    den.lifeTime -= Δlife
    if den.lifeTime <= 0
        remove!(den)
    end
end
function decay_life!(ap::AxonPoint, Δlife::Integer)
    ap.lifeTime -= Δlife
    if ap.lifeTime <= 0
        remove!(ap)
    end
end
function decay_life!(N::Neuron, Δlife::Integer)
    N.lifeTime -= Δlife
    if N.lifeTime <= 0
        remove!(N)
    end
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
                dispersion_collection[s] .+= (s.NT.strength, 1)
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
    neurons = get_all_neurons(NN)

    in_nodes = get_input_nodes(network_all_cells)
    out_nodes = get_output_nodes(network_all_cells)
    dens = get_dendrites(network_all_cells)
    APs = get_axon_points(network_all_cells)
    syns = get_synapses(network_all_cells)
    dispersion_collection = Dict()
    den_sinks = []
    ap_sinks = [Sink(o_n.possition, NN.ap_sink_attractive_force) for o_n in out_nodes] # array of ap sinks

    # 1
    for i in eachindex(in_nodes, input)
        in_nodes[i].value = input[i]
        append!(den_sinks, [Sink(in_nodes[i].possition, in_nodes[i].value)])
    end

    if neurons == []
        return []
    end

    # 2
    # println("testing: $dispersion_collection")
    for n in neurons
        # accumulate!(n, dropout(synapses, NN.synapsesAccessDropout), dispersion_collection)
        accm!(n, syns, dispersion_collection) #all_synapses::Array{Synaps}, dispersion_collection::Dict{Synaps, Pair{FloatN, Integer}}
    end
    # println("testing: $dispersion_collection")

    # 3
    # this puts the NT(t-1) and then calculates NT(t)
    # this can be reversed
    if syns != []
        for s in syns
            if s.Q >= s.THR
                s.Q = 0.
            else
                updateQ!(s)
            end
            decay_life!(s, NN.life_decay)

            if !ismissing(s)
                dispersion_value, n = get(dispersion_collection, s, (1, 1))
                avg_NT_change = dispersion_value/n

                # change nt
                s.NT.strength = s.NT.retain_percentage * s.NT.strength + (1-s.NT.retain_percentage) * avg_NT_change
            end
        end
    end

    # 4
    for n in neurons
        decay_life!(n, NN.life_decay)

        if !ismissing(n)
            propergate!(n, den_sinks)
        end
    end

    if dens != []
        for d in skipmissing(dens)
            append!(ap_sinks, [Sink(d.possition, NN.ap_sink_attractive_force)])
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
    dens = get_dendrites_in_all(network_all_cells)
    APs = get_axon_points_in_all(network_all_cells)

    for den in dens
        decay_life!(den.cell, NN.life_decay)
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

    for ap in APs
        decay_life!(ap.cell, NN.life_decay)
        if !ismissing(ap.cell)
            total_v = [0.,0.,0.]
            for ap_sink in ap_sinks
                dir = direction(ap.cell.possition, ap_sink.possition)
                mag = NN.minFuseDistance / vector_length(dir)
                total_v .+= normalize(dir) .* mag .* (1 + ap_sink.strength)
            end
            total_v ./= length(ap_sinks)

            ap.cell.possition += Possition(total_v...)

            for d in dens
                if distance(ap.cell.possition, d.cell.possition) <= NN.minFuseDistance
                    # random possition for dispersion
                    nt = unfold(rand(NN.dna_stack.nt_dna_samples))

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
function populate_network!(NN::Network, num_neurons, max_num_priors, max_num_post)
    for n in 1:num_neurons
        n = unfold(rand(NN.dna_stack.n_dna_samples), copy(NN.n_id_counter))
        NN.n_id_counter += 1
        append!(NN.components, [n])

        for _ in 1:rand(1:max_num_priors)
            add_dendrite!(n, rand(NN.dna_stack.den_dna_samples), NN.components)
        end
        for _ in 1:rand(1:max_num_post)
            add_axon_point!(n, rand(NN.dna_stack.ap_dna_samples), NN.components)
        end
    end
end
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


# NETWORK ACCESS FUNCTIONS
function add_dendrite!(N::Neuron, denDNA::DendriteDNA, NN_collection)
    if has_empty_prior(N)
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                d = AllCell(unfold(denDNA))
                N.priors[i] = d
                append!(NN_collection, [d])
                return nothing
            end
        end
    end
end
function add_axon_point!(N::Neuron, apDNA::AxonPointDNA, NN_collection)
    if has_empty_post(N)
        for i in eachindex(N.posteriors)
            if ismissing(N.posteriors[i])
                ap = AllCell(unfold(apDNA))
                N.posteriors[i] = ap
                append!(NN_collection, [ap])
                return nothing
            end
        end
    end
end
function update_fitness!(s::Synaps, activated::Bool, fitness_decay::FloatN)
    s.fitness *= fitness_decay
    if activated
        s.total_fitness += 1.
        s.fitness += 1.
    end
end

# VERIFICATION FUNCTIONS
function rectifyDNA!(NDNA::NeuronDNA, maxLifeTime::Integer)
    clamp!(NDNA.LifeTime.min, 1, maxLifeTime-1)
    clamp!(NDNA.LifeTime.max, NDNA.LifeTime.min, maxLifeTime)
end
function rectifyDNA!(DDNA::DendriteDNA, maxLifeTime::Integer); end
function rectifyDNA!(APDNA::AxonPointDNA, maxLifeTime::Integer); end
function rectifyDNA!(SDNA::SynapsDNA, maxLifeTime::Integer, qDecay_bounds::min_max_pair, threshold_bounds::min_max_pair)
    clamp!(SDNA.QDecay, qDecay_bounds.min, qDecay_bounds.max)
    clamp!(SDNA.THR, threshold_bounds.min, threshold_bounds.max)
    clamp!(SDNA.LifeTime, 1, maxLifeTime)
end
function rectifyDNA!(NDNA::NetworkDNA)
end
function rectifyInitializationPossition!(pos::InitializationPossition, network_max_range::FloatN)
end
