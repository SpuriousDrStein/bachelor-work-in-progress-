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


function get_random_possition(range)
    return Possition(rand(Uniform(-range, range)), rand(Uniform(-range, range)), rand(Uniform(-range, range)))
end

# INITIALIZATION SAMPELING
function sample(mean::Possition, global_variance::FloatN)
    Possition(rand(Normal(mean.x, global_variance)), rand(Normal(mean.y, global_variance)), rand(Normal(mean.z, global_variance)))
end
function sample(mean::FloatN, global_variance::FloatN)
    rand(Normal(min_max, global_variance))
end

# DNA GENERATOR FUNCTIONS
function unfold(dna::DendriteDNA, possition::Possition, global_variance::FloatN)::Dendrite
    return Dendrite(sample(dna.max_length, global_variance), sample(dna.lifeTime, global_variance), possition)
end
function unfold(dna::AxonPointDNA, possition::Possition, global_variance::FloatN)::AxonPoint
    return AxonPoint(sample(dna.max_length, global_variance), sample(dna.lifeTime, global_variance), possition)
end
function unfold(dna::SynapsDNA, s_id::Integer, possition::Possition, NT::NeuroTransmitter, life_decay::FloatN, global_variance::FloatN)::Synaps
    q_dec = sample(dna.QDecay, global_variance)
    thr = sample(dna.THR, global_variance)
    lifetime = sample(dna.lifeTime, global_variance)
    return Synaps(s_id, thr, q_dec, lifetime, 0, possition, NT, 0.)
end
function unfold(dna::NeuronDNA, pos::Possition, n_id::Integer, global_variance::FloatN)::Neuron
    lifetime = sample(dna.lifeTime, global_variance)
    num_priors = sample(dna.max_num_priors, global_variance)
    num_posteriors = sample(dna.max_num_posteriors, global_variance)
    den_and_ap_init_range = sample(dna.den_and_ap_init_range, global_variance)
    den_init_interval = round(sample(dna.den_init_interval, global_variance))
    ap_init_interval = round(sample(dna.ap_init_interval, global_variance))

    return Neuron(n_id, den_init_interval, ap_init_interval, den_and_ap_init_range, pos, 0., lifetime, [missing for _ in 1:num_priors], [missing for _ in 1:num_posteriors], 0., 0.)
end
function unfold(dna::NeuroTransmitterDNA, init_region_center::Possition, global_variance::FloatN)::NeuroTransmitter
    pos = init_region_center + sample(dna.dispersion_region, global_variance)
    return NeuroTransmitter(sample(dna.strength, global_variance), pos)
end
function unfold(dna::NeuroTransmitterDNA, global_variance::FloatN)
    pos = sample(dna.dispersion_region, global_variance)
    return NeuroTransmitter(sample(dna.init_strength, global_variance), pos, sample(dna.retain_percentage, global_variance))
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

    sink_force = sample(dna.ap_sink_force, global_variance)
    nrf = sample(dna.neuron_repel_force, global_variance)

    return Network(net_size, global_stdv, mNlife, mSlife, mDlife, mAlife, min_fuse_distance,
                    sink_force, nrf, max_nt_strength,
                    max_threshold, fitness_decay, random_fluctuation_scale,
                    neuron_init_interval, min_ap_den_init_interval, dna_stack, [], [],
                    life_decay, init_fitness, 0, 0)
end
