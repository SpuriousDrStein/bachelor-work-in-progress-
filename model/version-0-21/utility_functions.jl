# BASIC
to_stdv(var::FloatN) = sqrt(var)
is_activatable(syn::Synaps) = syn.Q >= syn.THR
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

get_all_positions(NN::Network) = begin
    a = []
    for c in skipmissing(NN.components)
        if typeof(c) == Neuron
            append!(a, c.position)
        else
            append!(a, c.cell.position)
        end
    end
end



# SPATIAL FUNCTIONS
direction(from::Position, to::Position) = [to.x, to.y, to.z] .- [from.x, from.y, from.z]
distance(p1::Position, p2::Position) = sqrt(sum(direction(p1,p2).^2))
vector_length(p::Position) = sqrt(sum([p.x, p.y, p.z].^2))
vector_length(v::Vector) = sqrt(sum(v.^2))
normalize(p::Position) = [p.x, p.y, p.z] ./ vector_length(p)
normalize(v::Vector) = v ./ vector_length(v)


function get_random_position(range)
    return Position(rand(Uniform(-range, range)), rand(Uniform(-range, range)), rand(Uniform(-range, range)))
end

# INITIALIZATION SAMPELING
function sample(mean::FloatN, global_stdv::FloatN)
    rand(Normal(mean, global_stdv))
end

# DNA GENERATOR FUNCTIONS
function unfold(dna::DendriteDNA, position::Position, NN::Network)::Dendrite
    max_length = max(1., sample(dna.max_length, NN.global_stdv))
    lifetime = clamp(sample(dna.lifeTime, NN.global_stdv), 1., NN.maxDendriteLifeTime)

    return Dendrite(max_length, lifetime, position)
end
function unfold(dna::AxonPointDNA, position::Position, NN::Network)::AxonPoint
    max_length = max(1., sample(dna.max_length, NN.global_stdv))
    lifetime = clamp(sample(dna.lifeTime, NN.global_stdv), 1., NN.maxAxonPointLifeTime)

    return AxonPoint(max_length, lifetime, position)
end
function unfold(dna::SynapsDNA, position::Position, NT::NeuroTransmitter, NN::Network)::Synaps
    q_dec = clamp(sample(dna.QDecay, NN.global_stdv), 0.1, 0.99)
    thr = clamp(sample(dna.THR, NN.global_stdv), 0.1, NN.max_threshold)
    lifetime = clamp(sample(dna.lifeTime, NN.global_stdv), 1., NN.maxSynapsLifeTime)

    return Synaps(copy(NN.s_id_counter), thr, q_dec, lifetime, 0, position, NT, 0.)
end
function unfold(dna::NeuronDNA, pos::Position, NN::Network)::Neuron
    lifetime = clamp(sample(dna.lifeTime, NN.global_stdv), 1., NN.maxNeuronLifeTime)
    num_priors = round(max(1, sample(dna.max_num_priors, NN.global_stdv)))
    num_posteriors = round(max(1, sample(dna.max_num_posteriors, NN.global_stdv)))
    den_and_ap_init_range = max(1., sample(dna.den_and_ap_init_range, NN.global_stdv))
    den_init_interval = round(max(sample(dna.den_init_interval, NN.global_stdv), NN.min_ap_den_init_interval))
    ap_init_interval = round(max(sample(dna.ap_init_interval, NN.global_stdv), NN.min_ap_den_init_interval))

    return Neuron(copy(NN.n_id_counter), den_init_interval, ap_init_interval, den_and_ap_init_range, pos, 0., lifetime, [missing for _ in 1:num_priors], [missing for _ in 1:num_posteriors], 0., 0.)
end
function unfold(dna::NeuroTransmitterDNA, NN::Network)::NeuroTransmitter
    str = clamp(sample(dna.init_strength, NN.global_stdv), 0.5, NN.max_nt_strength)
    retain_percentage = clamp(sample(dna.retain_percentage, NN.global_stdv), 0, 0.5) # 0.5 -> there should be minimum loss of own value for nt interations

    return NeuroTransmitter(str, retain_percentage)
end
function unfold(dna::NetworkDNA,
                net_size::FloatN,
                global_stdv::FloatN,
                mNlife::FloatN,
                mSlife::FloatN,
                mDlife::FloatN,
                mAlife::FloatN,
                min_fuse_distance::FloatN,
                life_decay::FloatN,
                max_nt_strength::FloatN,
                max_threshold::FloatN,
                random_fluctuation_scale::FloatN,
                fitness_decay::FloatN,
                neuron_init_interval::Integer,
                min_ap_den_init_interval::Integer,
                dna_stack;
                init_fitness=0)

    sink_force = sample(dna.ap_sink_force, global_stdv)
    nrf = sample(dna.neuron_repel_force, global_stdv)

    return Network(net_size, global_stdv, mNlife, mSlife, mDlife, mAlife, min_fuse_distance,
                    sink_force, nrf, max_nt_strength,
                    max_threshold, fitness_decay, random_fluctuation_scale,
                    neuron_init_interval, min_ap_den_init_interval, dna_stack, [], [],
                    life_decay, init_fitness, 0, 0)
end
