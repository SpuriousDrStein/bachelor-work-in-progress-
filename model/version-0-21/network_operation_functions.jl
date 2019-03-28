import Base.accumulate!

# SPATIAL UPDATE FUNCTIONS
function remove!(syn::Synaps);                      syn = missing; end
function remove!(den::Dendrite);                    den = missing; end
function remove!(ao::AxonPoint);                    ap = missing; end
function remove!(n::Neuron);                        n = missing; end


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
function updateQ!(syn::Synaps)
    decay_charge!(syn)
    syn.Q *= NT.strength
end
function accm!(N::Neuron, all_synapses::Array, dispersion_collection::Dict)
    N.Q = 0.

    input_syns = get_activatable_synapses(get_synapses(get_prior_all_cells(N)))
    input_nodes = get_prior_input_nodes(N)
    input_v = [[s.THR for s in input_syns]..., input_nodes...]

    if input_v != []
        N.Q = sum(input_v)
    else
        N.Q = 0.
    end

    for is in input_syns
        if (is.Q - is.THR) < 0.;  throw("Q - THR < 0 in synaps: $(is.id)");  end

        for s in get_synapses_in_range(Subnet(is.NT.dispersion_region, is.NT.range_scale * (is.Q - is.THR)), all_synapses)
            dispersion_collection[s] .+= (s.NT.strength, 1)
        end
    end   # dict of: synaps => neurotransmitter change
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
function NTChange!(syn::Synaps, input_strength::FloatN)
    syn.NT.strength = retination_percentage * syn.NT.strength + (1-syn.NT.retination_percentage) * input_strength
end
function value_step!(NN::Network, input::Array; accumulate_f=accf_sum)
    network_all_cells = get_all_all_cells(NN)
    in_nodes = get_all_input_nodes(NN)
    out_nodes = get_all_output_nodes(NN)
    neurons = get_all_neurons(NN)

    dens = get_dendrites(network_all_cells)
    APs = get_axon_points(network_all_cells)
    syns = get_synapses(network_all_cells)
    dispersion_collection = Dict()
    den_sinks = [] # array of dendrite sinks
    ap_sinks = [] # array of ap sinks

    if neurons == []
        return []
    elseif syns == []
        return [Sink(i_n.possition, i_n.value) for i_n in in_nodes], [Sink(o_n.possition, NN.ap_sink_attractive_force) for o_n in out_nodes]
    end


    # 1
    for i in eachindex(in_nodes, input)
        in_nodes[i].value = input[i]
    end

    # 2
    println("testing: $dispersion_collection")
    for n in neurons
        # accumulate!(n, dropout(synapses, NN.synapsesAccessDropout), dispersion_collection)
        accm!(n, syns, dispersion_collection) #all_synapses::Array{Synaps}, dispersion_collection::Dict{Synaps, Pair{FloatN, Integer}}
    end
    println("testing: $dispersion_collection")

    # 3
    # this puts the NT(t-1) into the calculation and then calculates NT(t)
    # this can be reversed
    for s in syns
        if s.Q >= s.THR
            s.Q = 0.
        else
            updateQ!(s)
        end

        dispersion_value, n = get(dispersion_collection, s, (1, 1))
        avg_NT_change = dispersion_value/n
        NTChange!(s, avg_NT_change)
    end

    # 4
    for n in neurons
        propergate!(n, den_sinks)
    end
    for d in dens
        append!(ap_sinks, [Sink(d.possition, NN.ap_sink_attractive_force)])
    end

    return den_sinks, ap_sinks
end
function state_step!(NN::Network, den_sinks, ap_sinks)
    # update spatial relation
    # - fuse!
    # - split!
    network_all_cells = get_all_all_cells(NN)
    in_nodes = get_all_input_nodes(NN)
    out_nodes = get_all_output_nodes(NN)
    neurons = get_all_neurons(NN)

    dens = get_dendrites(network_all_cells)
    APs = get_axon_points(network_all_cells)
    # syns = get_synapses(network_all_cells)


    for den in dens

        total_V = [0.,0.,0.]
        for d_sink in den_sinks
            norm_dist_v = normalize(distance(den.possition, d_sink.possition))
            total_V .+= norm_dist_v .* (1 + d_sink.strength)
        end
        decay_life!(den, NN.life_decay)
    end
    for ap in APs


        for ap_sink in ap_sinks

        end
        decay_life!(ap, NN.life_decay)
    end
    for n1 in neurons
        for n2 in neurons
            if n1 !== n2
                # repel neurons from each other
            end
        end
    end
end

# STRUCTURE GENERATION FUNCTIONS
function fuse!(den::AllCell, ap::AllCell, to::Synaps)
    if typeof(den.cell) != Dendrite || typeof(ap.cell) != AxonPoint; throw("incorect fuse!($(typeof(den.cell)), $(typeof(ap.cell)))"); end
    den.cell = to
    ap.cell = to
    return nothing
end
function addDendrite!(N::Neuron, denDNA::DendriteDNA)
    if any(ismissing.(N.priors))
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                N.priors[i] = AllCell(unfold(denDNA))
                return nothing
            end
        end
    end
    println("no Dendrite added to neuron: $(N.id)")
    return nothing
end
function addAxonPoint!(N::Neuron, apDNA::AxonPointDNA)
    if any(ismissing.(N.posteriors))
        for i in eachindex(N.posteriors)
            if ismissing(N.posteriors[i])
                N.posteriors[i] = AllCell(unfold(apDNA))
                return nothing
            end
        end
    end
    println("no AP added to neuron: $(N.id)")
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
