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



# PROPERGATION FUNCTIONS
function propergate!(N::Neuron, input_accumulate_f::Function)
    input_v = get_neuron_input_vector(N)
    println(typeof(input_v), input_v)
    a = input_accumulate_f(input_v)
    prior_activatable = get_activateable_synapses(get_all_prior_synapses(N))
    disperse!(prior_activatable)
    reset_synapses!(prior_activatable)

    posterior_synapses = get_all_posterior_synapses(N)
    V_update_synaps!.(posterior_synapses, a/length(posterior_synapses))
end
