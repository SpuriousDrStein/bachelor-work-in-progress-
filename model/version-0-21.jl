# version 2-_ are:
# exploring alternative solutions for threshold learning
# initial solutions for structural adaptation

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


mutable struct Synaps
    Q::FloatN # charge at t
    QDecay::FloatN
    THR::FloatN # threshold
    NT::NeuroTransmitter # NT for different functionalities

    # values to manage synaps life cycles
    lifeTime::FloatN
    lifeDecay::FloatN
    fitness::FloatN
end

mutable struct Dendrite
    possition::Possition
    force::Force # influence by, for example, neurotransmitters
end

mutable struct AxonPoint
    possition::Possition
    force::Force
end

mutable struct AllCell
    cell::Union{AxonPoint, Dendrite, Synaps, Neuron}
end

mutable struct Neuron
    possition::Possition
    force::Force
    priors::Array{Union{Missing, AllCell}, 1}
    posterior::Array{Union{Missing, AllCell}, 1}
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
    enteries::Array{Union{AllCell, InputNode, OutputNode}, 1}
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
function convert(T::Type{Synaps}, den::Dendrite, synaps_params)
    Synaps(synaps_params...)
end
function convert(T::Type{Synaps}, ap::AxonPoint, synaps_params)
    Synaps(synaps_params...)
end



# UPDATE FUNCTIONS
function propergate!(N::Neuron, input_accumulate::Function)
    input_v = get_neuron_input_vector(N)
    a = input_accumulate(input)
    activate_synapses!(N)
    for ap in N.posterior
        if ap.

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
function V_update_synaps!(syn::Synaps, input::FloatN)
    decay_charge!(syn)
    decay_life!(syn)
    syn.Q += input
    return nothing
end
function activate_synapses!(N)



# ?> function S_update_synaps!(syn::Synaps); end
function V_update_neuron!(N::Neuron, force::FloatN)
    N.force = force
    return nothing
end
function S_update_neuron!(N::Neuron)
    N.possition += N.force
    return nothing
end

function decay_charge!(syn::Synaps)
    syn.Q *= syn.QDecay
end

function decay_life!(syn::Synaps)
    syn.lifeTime -= syn.lifeDecay
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


# QUERY FUNCTIONS
function get_neuron_input_vector(N::Neuron)
    [s.cell.NT(s.cell.Q) for s in skipmissing(N.priors) if typeof(s.cell) == Synaps && is_activated(s.cell)]
end
function get_all_neurons(NN::Network)
    [n.cell for n in NN if typeof(n.cell) == Neuron]
end
function get_all_dendrites(NN::Network)
    [n.cell for n in NN if typeof(n.cell) == Neuron]
end
function get_all_axon_points(NN::Network)
    [n.cell for n in NN if typeof(n.cell) == Neuron]
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
