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

get_neurons(subnet::Subnet, neuron_collection::Array{Neuron}) = [n for n in skipmissing(neuron_collection) if distance(n.possition, subnet.possition) < subnet.range]
get_dendrites(subnet::Subnet, den_collection::Array{Dendrite}) = [d for d in skipmissing(den_collection) if distance(d.possition, subnet.possition) < subnet.range]
get_axon_points(subnet::Subnet, ap_collection::Array{AxonPoint}) = [ap for ap in skipmissing(ap_collection) if distance(ap.possition, subnet.possition) <= subnet.range]
get_synapses(subnet::Subnet, syn_collection::Array{Synaps}) = [syn for syn in skipmissing(syn_collection) if distance(syn.possition, subnet.possition) <= subnet.range]

get_all_possitions(NN::Network) = begin
    a = []
    for c in skipmissing(NN.components)
        if typeof(c) == Neuron
            append!(a, c.possition)
        else
            append!(a, c.cell.possition)
        end
    end
end



# SPATIAL FUNCTIONS
direction(from::Possition, to::Possition) = [to.x, to.y, to.z] .- [from.x, from.y, from.z]
distance(p1::Possition, p2::Possition) = sqrt(sum(direction(p1,p2).^2))
vector_length(p::Possition) = sqrt(sum([p.x, p.y, p.z].^2))
vector_length(v::Vector) = sqrt(sum(v.^2))
normalize(p::Possition) = [p.x, p.y, p.z] ./ vector_length(p)
normalize(v::Vector) = v ./ vector_length(v)




# INITIALIZATION SAMPELING
function get_random_init_possition(range::Number)
    return InitializationPossition(min_max_pair(-range, range), min_max_pair(-range, range), min_max_pair(-range, range))
end

function sample(init_pos::InitializationPossition)
    Possition(rand(Uniform(init_pos.x.min, init_pos.x.max)), rand(Uniform(init_pos.y.min, init_pos.y.max)), rand(Uniform(init_pos.z.min, init_pos.z.max)))
end
function sample(min_max::min_max_pair)
    rand(Uniform(min_max.min, min_max.max))
end

# DNA GENERATOR FUNCTIONS
function unfold(dna::DendriteDNA, possition::Possition)::Dendrite
    return Dendrite(sample(dna.max_length), sample(dna.lifeTime), possition)
end
function unfold(dna::AxonPointDNA, possition::Possition)::AxonPoint
    return AxonPoint(sample(dna.max_length), sample(dna.lifeTime), possition)
end
function unfold(dna::SynapsDNA, s_id::Integer, possition::Possition, NT::NeuroTransmitter, life_decay::FloatN)::Synaps
    q_dec = sample(dna.QDecay)
    thr = sample(dna.THR)
    lifetime = sample(dna.lifeTime)
    return Synaps(s_id, thr, q_dec, lifetime, 0, possition, NT, 0.)
end
function unfold(dna::NeuronDNA, pos::Possition, n_id::Integer)::Neuron
    lifetime = sample(dna.lifeTime)
    num_priors = sample(dna.max_num_priors)
    num_posteriors = sample(dna.max_num_posteriors)
    den_and_ap_init_range = sample(dna.den_and_ap_init_range)
    den_init_interval = round(sample(dna.den_init_interval))
    ap_init_interval = round(sample(dna.ap_init_interval))


    # println(pos, lifetime, num_priors, num_posteriors)
    return Neuron(n_id, den_init_interval, ap_init_interval, den_and_ap_init_range, pos, 0., lifetime, [missing for _ in 1:num_priors], [missing for _ in 1:num_posteriors], 0., 0.)
end
function unfold(dna::NeuroTransmitterDNA, init_region_center::Possition)
    pos = init_region_center + sample(dna.dispersion_region)
    return NeuroTransmitter(sample(dna.strength), pos)
end
function unfold(dna::NeuroTransmitterDNA)
    pos = sample(dna.dispersion_region)
    return NeuroTransmitter(sample(dna.init_strength), pos, sample(dna.dispersion_strength_scale), sample(dna.retain_percentage))
end
function unfold(dna::NetworkDNA,
                mNlife::FloatN,
                mSlife::FloatN,
                mDlife::FloatN,
                mAlife::FloatN,
                min_fuse_distance::FloatN,
                init_life_decay::FloatN,
                max_nt_dispersion_strength_scale::FloatN,
                max_threshold::FloatN,
                random_fluctuation_scale::FloatN,
                neuron_init_interval::Integer,
                min_ap_den_init_interval::Integer,
                dna_stack;
                fitness_decay=0.99,
                init_fitness=0)
    size = sample(dna.networkSize)
    sink_force = sample(dna.ap_sink_force)
    nrf = sample(dna.neuron_repel_force)

    return Network(size, mNlife, mSlife, mDlife, mAlife, min_fuse_distance,
                    sink_force, nrf, max_nt_dispersion_strength_scale,
                    max_threshold, fitness_decay, random_fluctuation_scale,
                    neuron_init_interval, min_ap_den_init_interval, dna_stack, [], [],
                    init_life_decay, init_fitness, 0, 0)
end


# ACCUMULATION FUNCTIONS
accf_sum(x) = sum(x)
accf_mean(x) = sum(x)/length(x)
