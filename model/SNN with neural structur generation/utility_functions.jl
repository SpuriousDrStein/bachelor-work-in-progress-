# basic utility
function to_degree(x); x*180/π; end

# query functions
get_dendrites(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Dendrite]
get_axon_points(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == AxonPoint]
get_synapses(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Synaps]
get_input_nodes(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == InputNode]
get_output_nodes(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == OutputNode]

get_dendrites_in_all(x::Array{AllCell}) = [n for n in x if typeof(n.cell) == Dendrite]
get_axon_points_in_all(x::Array{AllCell}) = [n for n in x if typeof(n.cell) == AxonPoint]
get_synapses_in_all(x::Array{AllCell}) = [n for n in x if typeof(n.cell) == Synaps]

get_activatable_synapses(x::Array{Synaps}) = [s for s in x if s.Q >= s.THR]

get_all_all_cells(NN::Network) = [n for n in NN.components if typeof(n) == AllCell] #..., NN.IO_components...]
get_all_neurons(NN::Network) = [n for n in NN.components if typeof(n) == Neuron]
get_all_neuron_indecies(NN::Network) = [i for i in 1:length(NN.components)][typeof.(NN.components) .== Neuron]
get_input_nodes(NN::Network) = [n.cell for n in NN.IO_components if typeof(n.cell) == InputNode]
get_output_nodes(NN::Network) = [n.cell for n in NN.IO_components if typeof(n.cell) == OutputNode]
get_input_nodes_in_all(NN::Network) = [n for n in NN.IO_components if typeof(n.cell) == InputNode]
get_output_nodes_in_all(NN::Network) = [n for n in NN.IO_components if typeof(n.cell) == OutputNode]

has_empty_prior(N::Neuron) = any([ismissing(p) for p in N.priors])
has_empty_post(N::Neuron) = any([ismissing(p) for p in N.posteriors])
get_prior_all_cells(N::Neuron) = [n for n in skipmissing(N.priors) if typeof(n) == AllCell]
get_posterior_all_cells(N::Neuron) = [n for n in skipmissing(N.posteriors) if typeof(n) == AllCell]

get_neurons(subnet::Subnet, neuron_collection::Array{Neuron}) = [n for n in skipmissing(neuron_collection) if distance(n.position, subnet.position) < subnet.range]
get_dendrites(subnet::Subnet, den_collection::Array{Dendrite}) = [d for d in skipmissing(den_collection) if distance(d.position, subnet.position) < subnet.range]
get_axon_points(subnet::Subnet, ap_collection::Array{AxonPoint}) = [ap for ap in skipmissing(ap_collection) if distance(ap.position, subnet.position) <= subnet.range]
get_synapses(subnet::Subnet, syn_collection::Array{Synaps}) = [syn for syn in skipmissing(syn_collection) if distance(syn.position, subnet.position) <= subnet.range]

# disjointed fitness accumulation
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


# position functions
direction(from::Position, to::Position) = [to.x, to.y] .- [from.x, from.y]
distance(p1::Position, p2::Position) = sqrt(sum(direction(p1, p2).^2))
vector_length(p::Position) = sqrt(sum([p.x, p.y].^2))
vector_length(v::Vector) = sqrt(sum(v.^2))
vec_mean(p1::Position, p2::Position) = Position((p1 + p2) ./ 2.)
function get_random_position(range)
    return Position(rand(Distributions.Uniform(-range, range)), rand(Distributions.Uniform(-range, range)))
end
function change_length(position::Position, length)
    Position((normalize(position) .* length)...)
end
function rectify_position(p::Position, len::FloatN)
    if distance(p, Position(0,0)) > len
        return Position((normalize(p) * len)...)
    else
        return p
    end
end
function get_all_relations(NN::Network)
    np = []
    app = []
    denp = []
    synp = []
    inp = [[i.position, i.value] for i in get_input_nodes(NN)]
    outp = [[o.position, o.value] for o in get_output_nodes(NN)]
    cons = []

    all_all = get_all_all_cells(NN)
    all_n = get_all_neurons(NN)

    if all_n == []
        return [[[],[]], [[],[]], [[],[]], [[],[]], inp, outp], []
    else
        for n in all_n
            append!(np, [[n.position, n.Q]])

            for pr in skipmissing(get_prior_all_cells(n))
                append!(cons, [[n.position + Position(direction(n.position, pr.cell.position)...), n.position]])
            end
            for po in skipmissing(get_posterior_all_cells(n))
                append!(cons, [[n.position, n.position + Position(direction(n.position, po.cell.position)...)]])
            end
        end
        if all_all != []
            for ac in all_all
                if typeof(ac.cell) == Dendrite
                    append!(denp, [[ac.cell.position, 1]])
                elseif typeof(ac.cell) == AxonPoint
                    append!(app, [[ac.cell.position, 1]])
                elseif typeof(ac.cell) == Synaps
                    append!(synp, [[ac.cell.position, ac.cell.Q]])
                end
            end
        end
        return [np, app, denp, synp, inp, outp], cons
    end
end


# initialization functions
function initialize_network(
                net_size::FloatN,
                global_stdv::FloatN,
                # nlife::FloatN,
                # slife::FloatN,
                # dlife::FloatN,
                # aplife::FloatN,
                min_fuse_distance::FloatN,
                ap_sink_attractive_force::FloatN,
                ap_surge_repulsive_force::FloatN,
                den_surge_repulsive_force::FloatN,
                in_attractive_force::FloatN,
                out_attractive_force::FloatN,
                # nrf::FloatN,
                max_nt_strength::FloatN,
                max_n_threshold::FloatN,
                max_s_threshold::FloatN,
                # life_decay::FloatN,
                nt_retain_percentage::FloatN,
                # den_and_ap_init_range::FloatN,
                # neuron_init_interval::Integer,
                # ap_den_init_interval::Integer,
                max_priors::Integer,
                max_posteriors::Integer,
                n_dest_thresh::FloatN,
                s_dest_thresh::FloatN,
                dna_stack::DNAStack)

    return Network(net_size,
                    global_stdv,
                    # nlife,
                    # slife,
                    # dlife,
                    # aplife,
                    min_fuse_distance,
                    ap_sink_attractive_force,
                    ap_surge_repulsive_force,
                    den_surge_repulsive_force,
                    in_attractive_force,
                    out_attractive_force,
                    # nrf,
                    max_nt_strength,
                    max_n_threshold,
                    max_s_threshold,
                    # life_decay,
                    nt_retain_percentage,
                    # neuron_init_interval,
                    # ap_den_init_interval,
                    max_priors,
                    max_posteriors,
                    dna_stack,
                    [], [],
                    0, 0, 0, 0, 0,
                    n_dest_thresh, s_dest_thresh)
end
function unfold(dna::NeuroTransmitterDNA, NN::Network)::NeuroTransmitter
    str = clamp(dna.init_strength, 0.1, NN.max_nt_strength)
    return NeuroTransmitter(str)
end
function unfold(dna::SynapsDNA, layer::Integer, pos::Position, NT::NeuroTransmitter, NN::Network)::Synaps
    thr = clamp(dna.THR, 0.5, NN.max_s_threshold)
    r_rec = max(1.1, dna.r_rec) # 1.1 because at 1 it has to increase
    maxR = max(1., dna.maxR)

    return Synaps(layer, pos, NT, 0, thr, 1, r_rec, maxR, 0, 0, NN.s_destruction_threshold)
end
function unfold(dna::NeuronDNA, layer::Integer, pos::Position, NN::Network)::Neuron
    thr = clamp(dna.THR, 0.5, NN.max_n_threshold)

    return Neuron(layer, pos, 0., thr, [missing for _ in 1:NN.max_num_priors],
                    [missing for _ in 1:NN.max_num_posteriors],
                    0., 0., NN.n_destruction_threshold)
end


# rectification functions
function rectify!(dna::NeuroTransmitterDNA, NN::Network)
    a = copy(dna.init_strength)

    dna.init_strength = clamp(dna.init_strength, 0.1, NN.max_nt_strength)

    return a != dna.init_strength
end
function rectify!(dna::SynapsDNA, NN::Network)
    thr = copy(dna.THR)
    r_rec = copy(dna.r_rec)
    maxR = copy(dna.maxR)

    dna.THR = clamp(dna.THR, 0.5, NN.max_s_threshold)
    dna.r_rec = max(1.1, dna.r_rec) # 1.1 because at 1 it has to increase
    dna.maxR = max(1., dna.maxR)

    return sum([thr != dna.THR, r_rec != dna.r_rec, maxR != dna.maxR])
end
function rectify!(dna::NeuronDNA, NN::Network)
    thr = copy(dna.THR)
    dna.THR = clamp(dna.THR, 0.5, NN.max_n_threshold)
    return thr != dna.THR
end
function rectify!(dna::DNAStack, NN::Network)
    s = 0
    for nt in dna.nt_dna_samples
        s += rectify!(nt, NN)
    end
    for syn in dna.syn_dna_samples
        s += rectify!(syn, NN)
    end
    for n in dna.n_dna_samples
        s += rectify!(n, NN)
    end
    return s
end
