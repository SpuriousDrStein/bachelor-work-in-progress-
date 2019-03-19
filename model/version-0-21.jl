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
    x::Int16
    y::Int16
    z::Int16
    strength::FloatN
end

mutable struct Synaps
    Q::FloatN # charge at t
    AP::FloatN # action potential at t
    THR::FloatN # threshold
    NT::NeuroTransmitter # NT for different functionalities

    # values to manage synaps life cycles
    Life::FloatN
    Decay::FloatN
end

mutable struct Dendrite
    possition::Possition
    force::Force # influence by, for example, neurotransmitters
end

mutable struct AxonPoint
    possition::Possition
    force::Force
end

mutable struct Neuron
    possition::Possition
    priors::Array{Union{Missing, Dendrite, Synaps}, 3}
    posterior::Array{Union{Missing, AxonPoint, Synaps}, 3}
end

mutable struct Subnet # may be used for more update by predetermined references or for neurotransmitter dispersion
    possition::Possition
    range::FloatN
end




# three kinds of updates
# S - structure - when structurally important states change
# T - temporal - when computationally important states change

# UPDATE FUNCTIONS
function V_update_axon_point!(AP::AxonPoint); end # basically only updates the force (3/19/19/1:52)
function S_update_axon_point!(AP::AxonPoint); end # basically only updates the possition based on the force at t [so V_update has to be called first] (3/19/19/1:52)
    function V_update_dendrite!(D::Dendrite); end
    function S_update_dendrite!(D::Dendrite); end
function V_update_synaps!(syn::Synaps); end
# ?> function S_update_synaps!(syn::Synaps); end
# ?> function V_update_neuron!(N::Neuron); end
function S_update_neuron!(N::Neuron); end


# QUERY FUNCTIONS
function get_all_neurons; end
function get_all_dendrites; end
function get_all_axon_points; end
function get_neurons_in_range(possition::Possition, range::FloatN); end
function get_dendrites_in_range(possition::Possition, range::FloatN); end
function get_axon_points_in_range(possition::Possition, range::FloatN); end
