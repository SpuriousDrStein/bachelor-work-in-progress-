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
function activate_synaps!(syn::Synaps)
    if (s.Q - s.THR) < 0
        throw("unexpected behaviour |> calling activation without threshold reached")
    else
        s.Q -= s.THR
        s.numActivation += 1
        return copy(s.THR)
    end
end

function reset_synaps!(syn::Synaps)
    syn.Q = 0.
end

# function disperse!(charge::FloatN, region::Subnet, synapses::Array{Synaps})
#     valid_syns = get_synapses_in_range(region.possition, region.range, synapses)
#     for i in eachindex(valid_syns)
#         valid_syns[i].Q += charge/length(valid_syns)
#         # redraw NT
#     end
# end


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



# Main LOOP FUNCTIONS
function Base.accumulate!(N::Neuron, accumulate_f::Function, )::Array{Tuple{FloatN, Array{Synaps}}}
    input_syns = get_activatable_prior_synapses(N)
    input_v = [activate_synaps!(s) for s in input_syns]
    if input_v != []
        N.Q = accumulate_f(input_v)
    else
        N.Q = 0.
    end

    dispersion = [(is.NT.strength => get_synapses_in_range(Subnet(is.possition), all_synapses_for_dispersion)) for is in input_syns]
    # basicly stores array of: Q-effect -> effected synapses

    reset_synaps!.(input_syns)
    return input_syns, dispersion
end

function propergate!(N::Neuron)

    posterior_synapses = get_all_post_synapses(N)
    V_update_synaps!.(posterior_synapses, a/length(posterior_synapses))
    return nothing
end
