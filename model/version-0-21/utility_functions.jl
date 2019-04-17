# BASIC
has_empty_prior(N::Neuron) = any([ismissing(p) for p in N.priors])
has_empty_post(N::Neuron) = any([ismissing(p) for p in N.posteriors])

# QUERY FUNCTIONS
get_dendrites(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Dendrite]
get_axon_points(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == AxonPoint]
get_synapses(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Synaps]
get_input_nodes(NN::Network) = [n.cell for n in NN.IO_components if typeof(n.cell) == InputNode]
get_output_nodes(NN::Network) = [n.cell for n in NN.IO_components if typeof(n.cell) == OutputNode]
get_input_nodes(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == InputNode]
get_output_nodes(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == OutputNode]


get_dendrites_in_all(x::Array{AllCell}) = [n for n in x if typeof(n.cell) == Dendrite]
get_axon_points_in_all(x::Array{AllCell}) = [n for n in x if typeof(n.cell) == AxonPoint]
get_synapses_in_all(x::Array{AllCell}) = [n for n in x if typeof(n.cell) == Synaps]
get_input_nodes_in_all(NN::Network) = [n for n in NN.IO_components if typeof(n.cell) == InputNode]
get_output_nodes_in_all(NN::Network) = [n for n in NN.IO_components if typeof(n.cell) == OutputNode]

get_activatable_synapses(x::Array{Synaps}) = [s for s in x if s.Q >= s.THR]

get_all_all_cells(NN::Network) = [n for n in NN.components if typeof(n) == AllCell] #..., NN.IO_components...]
get_all_neurons(NN::Network) = [n for n in NN.components if typeof(n) == Neuron]
get_all_neuron_indecies(NN::Network) = [i for i in 1:length(NN.components)][typeof.(NN.components) .== Neuron]

get_prior_all_cells(N::Neuron) = [n for n in skipmissing(N.priors) if typeof(n) == AllCell]
get_posterior_all_cells(N::Neuron) = [n for n in skipmissing(N.posteriors) if typeof(n) == AllCell]

get_neurons(subnet::Subnet, neuron_collection::Array{Neuron}) = [n for n in skipmissing(neuron_collection) if distance(n.position, subnet.position) < subnet.range]
get_dendrites(subnet::Subnet, den_collection::Array{Dendrite}) = [d for d in skipmissing(den_collection) if distance(d.position, subnet.position) < subnet.range]
get_axon_points(subnet::Subnet, ap_collection::Array{AxonPoint}) = [ap for ap in skipmissing(ap_collection) if distance(ap.position, subnet.position) <= subnet.range]
get_synapses(subnet::Subnet, syn_collection::Array{Synaps}) = [syn for syn in skipmissing(syn_collection) if distance(syn.position, subnet.position) <= subnet.range]


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


# SPATIAL FUNCTIONS
direction(from::Position, to::Position) = [to.x, to.y] .- [from.x, from.y]
distance(p1::Position, p2::Position) = sqrt(sum(direction(p1, p2).^2))
vector_length(p::Position) = sqrt(sum([p.x, p.y].^2))
vector_length(v::Vector) = sqrt(sum(v.^2))
normalize(p::Position) = [p.x, p.y] ./ vector_length(p)
normalize(v::Vector) = v ./ vector_length(v)

function get_random_position(range)
    return Position(rand(Uniform(-range, range)), rand(Uniform(-range, range)))
end

function get_all_positions(NN::Network)
    np = []
    app = []
    denp = []
    synp = []
    inp = [i.cell.position for i in get_input_nodes_in_all(NN)]
    outp = [o.cell.position for o in get_output_nodes_in_all(NN)]

    all_all = get_all_all_cells(NN)
    all_n = get_all_neurons(NN)

    if all_n == []
        return [], [], [], [], inp, outp
    else
        for n in all_n
            append!(np, [n.position])
        end
        if all_all != []
            for ac in all_all
                if typeof(ac.cell) == Dendrite
                    append!(denp, [ac.cell.position])
                elseif typeof(ac.cell) == AxonPoint
                    append!(app, [ac.cell.position])
                elseif typeof(ac.cell) == Synaps
                    append!(synp, [ac.cell.position])
                end
            end
        end
        return np, app, denp, synp, inp, outp
    end
end


# INITIALIZATION SAMPELING
function initialize_network(
                net_size::FloatN,
                global_stdv::FloatN,
                mNlife::FloatN,
                mSlife::FloatN,
                mDlife::FloatN,
                mAlife::FloatN,
                min_fuse_distance::FloatN,
                ap_sink_attractive_force::FloatN,
                ap_surge_repulsive_force::FloatN,
                den_surge_repulsive_force::FloatN,
                in_attractive_force::FloatN,
                out_attractive_force::FloatN,
                nrf::FloatN,
                max_nt_strength::FloatN,
                max_n_threshold::FloatN,
                max_s_threshold::FloatN,
                random_fluctuation_scale::FloatN,
                light_life_decay::FloatN,
                heavy_life_decay::FloatN,
                nt_retain_percentage::FloatN,
                neuron_init_interval::Integer,
                min_ap_den_init_interval::Integer,
                n_dest_thresh::FloatN,
                s_dest_thresh::FloatN,
                component_stack::DNAStack,
                init_fitness=0)

    return Network(net_size,
                    global_stdv,
                    mNlife,
                    mSlife,
                    mDlife,
                    mAlife,
                    min_fuse_distance,
                    ap_sink_attractive_force,
                    ap_surge_repulsive_force,
                    den_surge_repulsive_force,
                    in_attractive_force,
                    out_attractive_force,
                    nrf,
                    max_nt_strength,
                    max_n_threshold,
                    max_s_threshold,
                    random_fluctuation_scale,
                    light_life_decay,
                    heavy_life_decay,
                    nt_retain_percentage,
                    neuron_init_interval,
                    min_ap_den_init_interval,
                    component_stack,
                    [], [],
                    init_fitness,
                    0, 0, 0, 0,
                    n_dest_thresh, s_dest_thresh)
end


function sample(mean::FloatN, global_stdv::FloatN)
    rand(Normal(mean, global_stdv))
end
function rectify_position(p::Position, nn_size::FloatN)
    if distance(p, Position(0,0)) > nn_size
        return Position((normalize(p) * nn_size)...)
    else
        return p
    end
end
function unfold(dna::DendriteDNA, position::Position, NN::Network)::Dendrite
    max_length = max(1., dna.max_length)
    lifetime = clamp(dna.lifeTime, 1., NN.maxDendriteLifeTime)

    return Dendrite(max_length, lifetime, position)
end
function unfold(dna::AxonPointDNA, position::Position, NN::Network)::AxonPoint
    max_length = max(1., dna.max_length, NN.global_stdv)
    lifetime = clamp(dna.lifeTime, 1., NN.maxAxonPointLifeTime)

    return AxonPoint(max_length, lifetime, position)
end
function unfold(dna::NeuroTransmitterDNA, NN::Network)::NeuroTransmitter
    str = clamp(dna.init_strength, 0.1, NN.max_nt_strength)
    return NeuroTransmitter(str)
end
function unfold(dna::SynapsDNA, pos::Position, NT::NeuroTransmitter, NN::Network)::Synaps
    lifetime = clamp(dna.lifeTime, 1., NN.maxSynapsLifeTime)
    thr = clamp(dna.THR, 0.5, NN.max_s_threshold)
    r_rec = max(1.1, dna.r_rec) # 1.1 because at 1 it has to increase
    maxR = max(1., dna.maxR)

    return Synaps(lifetime, pos, NT, 0, thr, 1, r_rec, maxR, 0, 0, NN.s_destruction_threshold)
end
function unfold(dna::NeuronDNA, pos::Position, NN::Network)::Neuron
    lifetime = clamp(dna.lifeTime, 1., NN.maxNeuronLifeTime)
    den_init_interval = round(max(dna.den_init_interval, NN.min_ap_den_init_interval))
    ap_init_interval = round(max(dna.ap_init_interval, NN.min_ap_den_init_interval))
    den_and_ap_init_range = max(1., dna.den_and_ap_init_range)

    max_num_priors = round(max(1, dna.max_num_priors))
    max_num_posteriors = round(max(1, dna.max_num_posteriors))

    thr = clamp(dna.THR, 0.5, NN.max_n_threshold)

    return Neuron(den_init_interval, ap_init_interval, den_and_ap_init_range, pos, lifetime, 0., thr,
                    [missing for _ in 1:max_num_priors], [missing for _ in 1:max_num_posteriors],
                    0., 0., NN.n_destruction_threshold)
end
