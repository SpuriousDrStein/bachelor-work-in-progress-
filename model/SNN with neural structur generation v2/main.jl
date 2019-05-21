FloatN = Float16

# R = V / I
# V = R * I
# I = V / R
# Q = C * V
# (C = Q / V)
# Q = C * R * I
# I = Q / (C * R)
# (Q = V * t)

# R proportional to V
# I proportional to V
# I proportional to Q
# R inv. proportional to I

function base_update(syn::Neuron, input::Real)

    syn.I = syn.V / syn.R
    dV = syn.I * syn.R
    syn.V += dV
    syn.Q = syn.C * syn.V

    if syn.Q >= syn.THR
        syn.V = 0
    end
end # maybe deprecated

function calculate_syn_I(syn::Synaps)
    syn.preN.Q/(cp.C * cp.R)
end

function calculate_postN_I(syn::Synaps)

    syn.I

    dV = syn.I * syn.R

    syn.V += dV

    syn.postN.I += (syn.V >= syn.THR) * (syn.preN.Q/(cp.C * cp.R))
end


# STRUCTURE
mutable struct Vec3
    x::FloatN
    y::FloatN
    z::FloatN
end
mutable struct Neuron
# constant parameters
    position::Vec3
    orientation::Vec3
    THR::FloatN
    C::FloatN

# volitile parameters
    Q::FloatN
end

mutable struct Axon
    points::Array{Vec3}
    R::FloatN
    C::FloatN
    neuron::Neuron
end

mutable struct Synaps
    position::Vec3

    V::FloatN
    C::FloatN
    THR::FloatN

    preN::Neuron
    postN::Neuron
    cp::Axon
end

mutable struct Dendrite
    position::Vec3
    neuron::Neuron
end


mutable struct NetworkParameters
end

# updates
#=

    for each neuron:

        dQ/d



=#
