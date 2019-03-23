# QUERY FUNCTIONS
function get_all_neurons(NN::Network)
    [n for n in NN if typeof(n) == Neuron]
end
function get_all_input_nodes(NN::Network)
    [n for n in NN if typeof(n) == InputNode]
end
function get_all_outputNodes(NN::Network)
    [n for n in NN if typeof(n) == OutputNode]
end
function get_all_dendrites(NN::Network)
    [n.cell for n in NN if typeof(n) == AllCell && typeof(n.cell) == Dendrite]
end
function get_all_axon_points(NN::Network)
    [n.cell for n in NN if typeof(n) == AllCell && typeof(n.cell) == AxonPoint]
end

function get_dendrites(N::Neuron)
    [n.cell for n in skipmissing(N.priors) if typeof(n.cell) == Dendrite]
end
function get_axon_points(N::Neuron)
    [n.cell for n in skipmissing(N.posteriors) if typeof(n.cell) == AxonPoint]
end
function get_activatable_prior_synapses(N::Neuron)
    [n.cell for n in skipmissing(N.priors) if typeof(n.cell) == Synaps && is_activated(n.cell)]
end
function get_all_prior_synapses(N::Neuron)
    [n.cell for n in skipmissing(N.priors) if typeof(n.cell) == Synaps]
end
function get_activatable_post_synapses(N::Neuron)
    [n.cell for n in skipmissing(N.posteriors) if typeof(n.cell) == Synaps && is_activated(n.cell)]
end
function get_all_post_synapses(N::Neuron)
    [n.cell for n in skipmissing(N.posteriors) if typeof(n.cell) == Synaps]
end
function get_neuron_input_vector(N::Neuron)
    [s.cell.NT(s.cell.Q) for s in skipmissing(N.priors) if typeof(s.cell) == Synaps && is_activated(s.cell)]
end



function get_neurons_in_subnet(subnet::Subnet, neuron_collection::Array{Neuron})
    den_collection[[scalar_distance(n.possition, subnet.possition) < subnet.range for n in skipmissing(neuron_collection)]]
end
function get_dendrites_in_subnet(subnet::Subnet, den_collection::Array{Dendrite})
    den_collection[[scalar_distance(dp.possition, subnet.possition) < subnet.range for dp in skipmissing(den_collection)]]
end
function get_axon_points_in_subnet(subnet::Subnet, ap_collection::Array{AxonPoint})
    ap_collection[[scalar_distance(ap.possition, subnet.possition) <= subnet.range for ap in skipmissing(ap_collection)]]
end
function get_synapses_in_subnet(subnet::Subnet, syn_collection::Array{Synaps})
    syn_collection[[scalar_distance(syn.possition, subnet.possition) <= subnet.range for syn in skipmissing(syn_collection)]]
end


# UTILITY FUNCTIONS
function is_activated(syn::Synaps)
    syn.Q >= syn.THR
end

function direction(from::Possition, to::Possition)
    [to.x, to.y, to.z] .- [from.x, from.y, from.z]
end

function scalar_distance(p1::Possition, p2::Possition)
    sqrt(sum([p1.x, p1.y, p1.z] .- [p2.x, p2.y, p2.z]).^2)
end

function to_stdv(var::FloatN)
    sqrt(var)
end

function get_random_init_possition(mean::FloatN, variance::FloatN)
    InitializationPossition(m_v_pair(mean, variance), m_v_pair(mean, variance), m_v_pair(mean, variance))
end

function sample(init_pos::InitializationPossition)
    Possition(rand(Normal(init_pos.x.mean, init_pos.x.variance)), rand(Normal(init_pos.y.mean, init_pos.y.variance)), rand(Normal(init_pos.z.mean, init_pos.z.variance)))
end
function sample(min_max::min_max_pair)
    rand(min_max.min:min_max.max)
end
function sample(m_v::m_v_pair)
    rand(Normal(m_v.mean, to_stdv(m_v.variance)))
end


# ACCUMULATION FUNCTIONS
accf_sum(x) = sum(x)
accf_mean(x) = sum(x)/length(x)
