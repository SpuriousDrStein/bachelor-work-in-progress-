# version 2-_ are:
# exploring alternative solutions for threshold learning
# initial solutions for structural adaptation
using Distributions


FloatN = Float32

mutable struct Possition
    x::FloatN
    y::FloatN
    z::FloatN
end

mutable struct Force # identity vector with a force
    x::FloatN
    y::FloatN
    z::FloatN
    strength::FloatN
end

mutable struct NeuroTransmitter # small possitive or small negative
    # addition for future versions --> have a function that gets the specific Synaps -> for (propergate, disperse, etc...)
    strength::FloatN
end

mutable struct Dendrite
    possition::Possition
    force::Force # influence by, for example, neurotransmitters
end

mutable struct AxonPoint
    possition::Possition
    force::Force
end

mutable struct Synaps
    possition::Possition
    Q::FloatN # charge at t
    QDecay::FloatN
    THR::FloatN # threshold
    NT::NeuroTransmitter # NT for different functionalities

    # values to manage synaps life cycles
    lifeTime::FloatN
    lifeDecay::FloatN

    numActivation::Integer
end

mutable struct AllCell
    cell::Union{AxonPoint, Dendrite, Synaps}
end

mutable struct Neuron
    possition::Possition
    force::Force
    priors::Array{Union{Missing, AllCell}, 1}
    posterior::Array{Union{Missing, AllCell}, 1}
    NT::NeuroTransmitter # NT for different functionalities

    lifeTime::FloatN
    lifeDecay::FloatN
    fitness::FloatN
end

mutable struct InputNode
    possition::Possition
    value::FloatN
end

mutable struct OutputNode
    possition::Possition
    value::FloatN
end

mutable struct Subnet # may be used for more update by predetermined references or for neurotransmitter dispersion
    possition::Possition
    range::FloatN
end

mutable struct Network
    enteries::Array{Union{AllCell, Neuron, InputNode, OutputNode}, 1}
end


# three kinds of updates
# S - structure - when structurally important states change
# T - temporal - when computationally important states change

# BASE OVERLOAD
import Base.+, Base.-, Base./
import Base.convert
function +(poss::Possition, f::Force); [poss.x, poss.y, poss.z] .+ ([f.x, f.y, f.z] .* f.strength); end
function +(p1::Possition, p2::Possition); [p1.x, p1.y, p1.z] .+ [p2.x, p2.y, p2.z]; end
function -(poss::Possition, f::Force); [poss.x, poss.y, poss.z] .- ([f.x, f.y, f.z] .* f.strength); end
function /(poss::Possition, n::Number); [poss.x, poss.y, poss.z] ./ n; end
function mean(p1::Possition, p2::Possition); (p1 + p2) / 2; end
# function convert(T::Type{Synaps}, den::Dendrite, synaps_params)
#     Synaps(synaps_params...)
# end
# function convert(T::Type{Synaps}, ap::AxonPoint, synaps_params)
#     Synaps(synaps_params...)
# end



# UPDATE FUNCTIONS
function V_update_axon_point!(AP::AxonPoint, force::FloatN)
    AP.force = force
    return nothing
end # basically only updates the force (3/19/19/1:52)
function S_update_axon_point!(AP::AxonPoint)
    AP.possition += AP.force
    return nothing
end # basically only updates the possition based on the force at t [so V_update has to be called first] (3/19/19/1:52)
function V_update_dendrite!(D::Dendrite, force)
    D.force = force
    return nothing
end
function S_update_dendrite!(D::Dendrite)
    D.possition += D.force
    return nothing
end

# ?> function S_update_synaps!(syn::Synaps); end
function V_update_neuron!(N::Neuron, force::FloatN)
    N.force = force
    N.lifeTime -= N.lifeDecay
    return nothing
end
function S_update_neuron!(N::Neuron)
    N.possition += N.force
    return nothing
end

function decay_charge!(syn::Synaps)
    syn.Q *= syn.QDecay
end
function kill_synaps!(syn::Synaps)
    syn = missing
end
function decay_life!(syn::Synaps)
    syn.lifeTime -= syn.lifeDecay
    syn.lifeDecay = 0.
    if syn.lifeTime <= 0
        kill_synaps!(syn)
    end
end
function V_update_synaps!(syn::Synaps, input::FloatN, Δlife_decay::FloatN)
    syn.lifeDecay = life_decay
    decay_charge!(syn)
    decay_life!(syn)
    syn.Q += input
    return nothing
end
function activate_synapses!(syn::Synaps)
    for s in syn
        if is_activated(ps)
            s.Q -= s.THR
        end
    end
end


# # test fuse function
# t_den = AllCell(Dendrite(Possition(0,0,0), Force(0,0,0,0)))
# t_ap = AllCell(AxonPoint(Possition(0,0,1), Force(0,0,0,0)))
# t_syn = Synaps(0,0,1,-0.01,5,1,1000)
# println(t_den.cell)
# fuse!(t_den, t_ap, t_syn)
# println(t_den.cell)
#
# # test collection rewireing
# a = []
# append!(a, [t_den])
# a[1].cell.possition = Possition(1,1,1)
# t_den.cell.possition

# ANEALING FUNCTIONS
function fuse!(den::AllCell, ap::AllCell, to::Synaps)
    if typeof(den.cell) != Dendrite || typeof(ap.cell) != AxonPoint; throw("incorect fuse!($(typeof(den.cell)), $(typeof(ap.cell)))"); end
    den.cell = to
    ap.cell = to
    return nothing
end

# function defuse!(syn::Synaps)


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

function get_neuron_input_vector(N::Neuron)
    [s.cell.NT(s.cell.Q) for s in skipmissing(N.priors) if typeof(s.cell) == Synaps && is_activated(s.cell)]
end
function get_all_prior_dendrites(N::Neuron)
    [n.cell for n in N.prior if typeof(n.cell) == Dendrite]
end
function get_all_posterior_dendrites(N::Neuron)
    [n.cell for n in N.posterior if typeof(n.cell) == Dendrite]
end
function get_all_axon_points(N::Neuron)
    [n.cell for n in NN.posterior if typeof(n.cell) == AxonPoint]
end
function get_activateable_synapses(N::Neuron)
    [is_activated(n.cell) for n in N.prior if typeof(n.cell) == Dendrite]
end

function get_neurons_in_range(possition::Possition, range::FloatN, neuron_collection::Array{Neuron})
    den_collection[[scalar_distance(n.possition, possition) < range for n in neuron_collection]]
end
function get_dendrites_in_range(possition::Possition, range::FloatN, den_collection::Array{Dendrite})
    den_collection[[scalar_distance(dp.possition, possition) < range for dp in den_collection]]
end
function get_axon_points_in_range(possition::Possition, range::FloatN, ap_collection::Array{AxonPoint})
    ap_collection[[scalar_distance(ap.possition, possition) <= range for ap in ap_collection]]
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


# PROPERGATION FUNCTIONS
function propergate!(N::Neuron, input_accumulate::Function)
    input_v = get_neuron_input_vector(N)
    a = input_accumulate(input)
    prior_activatable = get_activateable_synapses(get_all_prior_synapses(N))
    disperse!(prior_activatable)
    reset_synapses!(prior_activatable)

    posterior_synapses = get_all_posterior_synapses(N)
    V_update_synaps!.(posterior_synapses, a/length(posterior_synapses))
end


# INIT FUNCTIONS


# GENERATOR FUNCTIONS

mutable struct m_v_pair
    mean::FloatN
    variance::FloatN
end

mutable struct min_max_pair
    min::Number
    max::Number
end

mutable struct InitializationPossition
    x::m_v_pair
    y::m_v_pair
    z::m_v_pair
end

mutable struct NetworkDNA
    networkSize::FloatN # i.e. range; centered at 0

    maxNeuronLifeTime::min_max_pair
    maxSynapsLifeTime::min_max_pair
    NeuronLifeTimeDecay::FloatN # >0; <1
    SynapsLifeTimeDecay::FloatN # >0; <1

    NeuronAccessDropout::FloatN # dropout probability for unspecific neuron selections (1 for early tests)

end

mutable struct SynapsDNA
    QDecay::m_v_pair
    THR::m_v_pair
    LifeTime::min_max_pair
end

mutable struct NeuronDNA
    possition::InitializationPossition
    lifeTime::min_max_pair
    num_priors::min_max_pair
    num_posteriors::min_max_pair
    neuroTransmitter::NeuroTransmitter
end

function unfold(dna::SynapsDNA, possition::FloatN, NT::NeuroTransmitter)::Synaps
    q_dec = rand(Normal(dna.QDecay.mean, to_stdv(dna.QDecay.variance)))
    thr = rand(Normal(dna.THR.mean, to_stdv(dna.THR.variance)))
    lifetime = rand(dna.LifeTime.min:dna.LifeTime.max)
    return Synaps(possition, 0., q_dec, thr, NT, lifetime, 0)
end
function unfold(dna::NeuronDNA, possition::FloatN, NT::NeuroTransmitter)::Neuron
    lifetime = rand(dna.lifeTime.min:dna.lifeTime.max)
    num_priors = (dna.num_priors.min:dna.num_priors.max)
    num_posteriors = (dna.num_posteriors.min:dna.num_posteriors.max)
    return Neuron(possition, Force(0,0,0,0), [missing for _ in 1:num_priors], [missing for _ in 1:num_posteriors], NT, lifetime, 0)
end



# CONTROL FUNCTIONS
function addDendrite!(N::Neuron, dna::DendriteDNA); end
function addAxonPoint!(N::Neuron, dna::AxonPointDNA); end
    # add axon point and add N.NT reference to it
end



# VALIDATION FUNCTIONS
function validateDNA!(NDNA::NeuronDNA, maxLifeTime::FloatN)
    clamp!(NDNA.LifeTime.min, 1, maxLifeTime-1)
    clamp!(NDNA.LifeTime.max, NDNA.LifeTime.min, maxLifeTime)
end

function validateInRangeInitialization!(pos::Possition, network_range::FloatN)
end



# TESTING
t_NT = NeuroTransmitter(1)

t_neu = Neuron(Possition(0,0,0), Force(0,0,0,0), [missing for _ in 1:10], [missing for _ in 1:3], t_NT, 0)
