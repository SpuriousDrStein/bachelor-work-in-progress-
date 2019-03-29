# BASIC
to_stdv(var::FloatN) = sqrt(var)
is_activatable(syn::Synaps) = syn.Q >= syn.THR

# QUERY FUNCTIONS
get_dendrites(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Dendrite]
get_axon_points(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == AxonPoint]
get_synapses(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Synaps]
get_activatable_synapses(x::Array{Synaps}) = [s for s in x if s.Q >= s.THR]

get_all_all_cells(NN::Network) = [n for n in NN.components if typeof(n) == AllCell]
get_all_neurons(NN::Network) = [n for n in NN.components if typeof(n) == Neuron]
get_all_input_nodes(NN::Network) = [n for n in NN.components if typeof(n) == InputNode]
get_all_output_nodes(NN::Network) = [n for n in NN.components if typeof(n) == OutputNode]

get_input_nodes(N::Neuron) = [n.cell for n in skipmissing(N.priors) if typeof(n.cell) == InputNode]
get_output_nodes(N::Neuron) = [n.cell for n in skipmissing(N.posteriors) if typeof(n.cell) == OutputNode]


get_prior_all_cells(N::Neuron) = [n for n in skipmissing(N.priors) if typeof(n) == AllCell]
get_posterior_all_cells(N::Neuron) = [n for n in skipmissing(N.posteriors) if typeof(n) == AllCell]

get_neurons(subnet::Subnet, neuron_collection::Array{Neuron}) = [n for n in skipmissing(neuron_collection) if distance(n.possition, subnet.possition) < subnet.range]
get_dendrites(subnet::Subnet, den_collection::Array{Dendrite}) = [d for d in skipmissing(den_collection) if distance(d.possition, subnet.possition) < subnet.range]
get_axon_points(subnet::Subnet, ap_collection::Array{AxonPoint}) = [ap for ap in skipmissing(ap_collection) if distance(ap.possition, subnet.possition) <= subnet.range]
get_synapses(subnet::Subnet, syn_collection::Array{Synaps}) = [syn for syn in skipmissing(syn_collection) if distance(syn.possition, subnet.possition) <= subnet.range]


# SPATIAL FUNCTIONS
direction(from::Possition, to::Possition) = [to.x, to.y, to.z] .- [from.x, from.y, from.z]
distance(p1::Possition, p2::Possition) = sqrt(sum(direction(p1,p2).^2))
vector_length(p::Possition) = sqrt(sum([p.x, p.y, p.z].^2))
vector_length(v::Vector) = sqrt(sum(v.^2))
normalize(p::Possition) = [p.x, p.y, p.z] ./ vector_length(p)
normalize(v::Vector) = v ./ vector_length(v)


# INITIALIZATION SAMPELING
function get_random_init_possition(center::Possition, range::FloatN)
    return InitializationPossition(m_v_pair(center.x, range), m_v_pair(center.y, range), m_v_pair(center.z, range))
end
function get_random_init_sub_possition(center::Possition, range::FloatN, sub_sample_range::FloatN)
    sub_pos = sample(get_random_init_possition(center, range))
    return get_random_init_possition(sub_pos, sub_sample_range)
end

function sample(init_pos::InitializationPossition)
    Possition(rand(Normal(init_pos.x.mean, to_stdv(init_pos.x.variance))), rand(Normal(init_pos.y.mean, to_stdv(init_pos.y.variance))), rand(Normal(init_pos.z.mean, to_stdv(init_pos.z.variance))))
end
function sample(min_max::min_max_pair)
    rand(min_max.min:min_max.max)
end
function sample(m_v::m_v_pair)
    rand(Normal(m_v.mean, to_stdv(m_v.variance)))
end


# DNA GENERATOR FUNCTIONS
function unfold(dna::DendriteDNA)
    return Dendrite(sample(dna.max_length), sample(dna.lifeTime), sample(dna.init_pos))
end
function unfold(dna::SynapsDNA, s_id::Integer, possition::Possition, NT::NeuroTransmitter, life_decay::Integer)::Synaps
    q_dec = sample(dna.QDecay)
    thr = sample(dna.THR)
    lifetime = sample(dna.lifeTime)
    nt = sample(dna.NT)
    return Synaps(s_id, thr, q_dec, lifetime, 0, possition, nt)
end
function unfold(dna::AxonPointDNA)
    return AxonPoint(sample(dna.max_length), sample(dna.lifeTime), sample(dna.init_pos))
end
function unfold(dna::NeuronDNA, n_id::Integer)::Neuron
    pos = sample(dna.init_pos)
    lifetime = sample(dna.lifeTime)
    num_priors = sample(dna.max_num_priors)
    num_posteriors = sample(dna.max_num_posteriors)
    return Neuron(n_id, pos, 0., lifetime, [missing for _ in 1:num_priors], [missing for _ in 1:num_posteriors], 0., 0.)
end
function unfold(dna::NeuroTransmitterDNA, init_region_center::Possition)
    pos = init_region_center + sample(dna.dispersion_region)
    return NeuroTransmitter(sample(dna.strength), pos)
end
function unfold(dna::NeuroTransmitterDNA)
    pos = sample(dna.dispersion_region)
    return NeuroTransmitter(sample(dna.init_strength), pos, sample(dna.dispersion_strength_scale), sample(dna.retain_percentage))
end
function unfold(dna::NetworkDNA, min_fuse_distance::FloatN, init_life_decay::Integer, components)
    size = sample(dna.networkSize)
    mNlife = sample(dna.maxNeuronLifeTime)
    mSlife = sample(dna.maxSynapsLifeTime)
    mDlife = sample(dna.maxDendriteLifeTime)
    mAlife = sample(dna.maxAxonPointLifeTime)
    sink_force = sample(dna.ap_sink_force)
    nrf = sample(dna.neuron_repel_force)
    return Network(size, mNlife, mSlife, mDlife, mAlife, min_fuse_distance, sink_force, nrf, components, init_life_decay)
end


# ACCUMULATION FUNCTIONS
accf_sum(x) = sum(x)
accf_mean(x) = sum(x)/length(x)
