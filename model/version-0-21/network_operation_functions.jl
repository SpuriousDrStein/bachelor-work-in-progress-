# SPATIAL UPDATE FUNCTIONS
function apply_force!(AP::AxonPoint, force::Force); AP.Possition += force; end
function apply_force!(D::Dendrite, force::Force);   D.possition += force; end
function apply_force!(N::Neuron, force::Force);     N.Possition += force; end
function remove!(syn::Synaps);                      syn = missing; end
function remove!(den::Dendrite);                    den = missing; end
function remove!(ao::AxonPoint);                    ap = missing; end
function remove!(n::Neuron);                        n = missing; end


# VALUE UPDATE FUNCTIONS
function decay_charge!(syn::Synaps)
    syn.Q *= syn.QDecay
end
function decay_life!(syn::Synaps, Δlife::Integer)
    syn.lifeTime -= Δlife
    if syn.lifeTime <= 0
        remove!(syn)
    end
end
function decay_life!(den::Dendrite, Δlife::Integer)
    den.lifeTime -= Δlife
    if den.lifeTime <= 0
        remove!(den)
    end
end
function decay_life!(ap::AxonPoint, Δlife::Integer)
    ap.lifeTime -= Δlife
    if ap.lifeTime <= 0
        remove!(ap)
    end
end
function updateQ!(syn::Synaps, input::FloatN)
    syn.Q += input
    decay_charge!(syn)
    syn.Q *= NT.strength
end
function activate!(syn::Synaps)
    if syn.Q < syn.THR
        println("activate_synaps called but no activation occured in synaps: $(syn.id)")
        return 0.
    else
        if syn.activated
            println("accessing already activated synaps: $(syn.id)")
            return 0.
        else
            syn.Q -= syn.THR
            syn.numActivation += 1
            syn.activated = true
            return copy(syn.THR)
        end
    end
end
function setActivation!(syn::Synaps); syn.activated = true; end
function unsetActivation!(syn::Synaps); syn.activated = false; end
function accumulate!(N::Neuron, accumulate_f::Function, all_synapses_for_dispersion::Array{Synaps})::Array{Tuple{Array{Synaps}, FloatN}}
    N.Q = 0.
    input_syns = get_activatable_prior_synapses(N)
    input_v = [activate!(s) for s in input_syns]

    if input_v != []
        N.Q = accumulate_f(input_v)
    else
        N.Q = 0.
    end

    dispersion = [(get_synapses_in_range(Subnet(is.possition, is.NT.dispersion_range), all_synapses_for_dispersion) => is.NT.strength) for is in input_syns]
    # basicly stores array of: effected synapses -> Q-effect

    reset_synaps!.(input_syns)
    return dispersion
end

function propergate!(N::Neuron, dispersion::Array{Tuple{Array{Synaps}, FloatN}})
    # 1. update NeuroTransmitter in from dispersion effected synapses
    # 2.

    posterior_synapses = get_all_post_synapses(N)
    V_update_synaps!.(posterior_synapses, a/length(posterior_synapses))
    return nothing
end



# STRUCTURE GENERATION FUNCTIONS
function fuse!(den::AllCell, ap::AllCell, to::Synaps)
    if typeof(den.cell) != Dendrite || typeof(ap.cell) != AxonPoint; throw("incorect fuse!($(typeof(den.cell)), $(typeof(ap.cell)))"); end
    den.cell = to
    ap.cell = to
    return nothing
end
function addDendrite!(N::Neuron, denDNA::DendriteDNA)
    if any(ismissing.(N.priors))
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                N.priors[i] = AllCell(unfold(denDNA))
                return nothing
            end
        end
    end
    println("no den added to neuron: $(N.id)")
    return nothing
end
function addAxonPoint!(N::Neuron, dna::AxonPointDNA)
    # add axon point and add N.NT reference to it
end



# VERIFICATION FUNCTIONS
function rectifyDNA!(NDNA::NeuronDNA, maxLifeTime::FloatN)
    clamp!(NDNA.LifeTime.min, 1, maxLifeTime-1)
    clamp!(NDNA.LifeTime.max, NDNA.LifeTime.min, maxLifeTime)
    return nothing
end

function rectifyDNA!(DDNA::DendriteDNA, maxLifeTime::FloatN); end

function rectifyDNA!(SDNA::SynapsDNA, maxLifeTime::Integer, qDecay_bounds::min_max_pair, threshold_bounds::min_max_pair)
    clamp!(SDNA.QDecay, qDecay_bounds.min, qDecay_bounds.max)
    clamp!(SDNA.THR, threshold_bounds.min, threshold_bounds.max)
    clamp!(SDNA.LifeTime, 1, maxLifeTime)
end

function rectifyInRangeInitialization!(pos::Possition, network_range::FloatN, min_distance_between_components::FloatN)
end
