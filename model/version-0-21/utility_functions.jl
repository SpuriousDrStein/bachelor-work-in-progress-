# BASIC
to_stdv(var::FloatN) = sqrt(var)
is_activatable(syn::Synaps) = syn.Q >= syn.THR

# QUERY FUNCTIONS
get_all_neurons(NN::Network) = [n for n in NN if typeof(n) == AllCell && typeof(n.cell) == Neuron]
get_all_input_nodes(NN::Network) = [n for n in NN if typeof(n) == InputNode]
get_all_output_nodes(NN::Network) = [n for n in NN if typeof(n) == OutputNode]
get_all_dendrites(NN::Network) = [n.cell for n in NN if typeof(n) == AllCell && typeof(n.cell) == Dendrite]
get_all_axon_points(NN::Network) = [n.cell for n in NN if typeof(n) == AllCell && typeof(n.cell) == AxonPoint]


get_input_nodes(N::Neuron) = [n.cell for n in skipmissing(N.priors) if typeof(n.cell) == InputNode]
get_output_nodes(N::Neuron) = [n.cell for n in skipmissing(N.posteriors) if typeof(n.cell) == OutputNode]

get_prior_all_cells(N::Neuron) = [n for n in skipmissing(N.priors) if typeof(n) == AllCell]
get_posterior_all_cells(N::Neuron) = [n for n in skipmissing(N.posteriors) if typeof(n) == AllCell]
get_dendrites(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Dendrite]
get_axon_points(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == AxonPoint]
get_synapses(x::Array{AllCell}) = [n.cell for n in x if typeof(n.cell) == Synaps]

get_activatable_synapses(N::Array{Synaps}) = [s for s in N if  s.Q >= s.THR]

get_neurons(subnet::Subnet, neuron_collection::Array{Neuron}) = [n for n in skipmissing(neuron_collection) if distance(n.possition, subnet.possition) < subnet.range]
get_dendrites(subnet::Subnet, den_collection::Array{Dendrite}) = [d for d in skipmissing(den_collection) if distance(d.possition, subnet.possition) < subnet.range]
get_axon_points(subnet::Subnet, ap_collection::Array{AxonPoint}) = [ap for ap in skipmissing(ap_collection) if distance(ap.possition, subnet.possition) <= subnet.range]
get_synapses(subnet::Subnet, syn_collection::Array{Synaps}) = [syn for syn in skipmissing(syn_collection) if distance(syn.possition, subnet.possition) <= subnet.range]



# SPATIAL FUNCTIONS
direction(from::Possition, to::Possition) = [to.x, to.y, to.z] .- [from.x, from.y, from.z]
distance(p1::Possition, p2::Possition) = sqrt(sum([p1.x, p1.y, p1.z] .- [p2.x, p2.y, p2.z]).^2)



# INITIALIZATION SAMPELING
function get_random_init_possition(center::Possition, range::FloatN)
    InitializationPossition(m_v_pair(center.x, range), m_v_pair(center.y, range), m_v_pair(center.z, range))
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



# ACCUMULATION FUNCTIONS
accf_sum(x) = sum(x)
accf_mean(x) = sum(x)/length(x)
