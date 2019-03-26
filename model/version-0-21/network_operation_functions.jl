import Base.accumulate!

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
function updateQ!(syn::Synaps)
    decay_charge!(syn)
    syn.Q *= NT.strength
end
function Base.accumulate!(N::Neuron, synapses_for_dispersion::Array{Synaps}, dispersion_collection::Dict{Synaps, Pair{FloatN, Integer}}; accumulate_f=accf_sum)
    N.Q = 0.

    input_syns = get_activatable_synapses(get_synapses(get_prior_all_cells(N)))
    input_nodes = get_prior_input_nodes(N)
    input_v = [[s.THR for s in input_syns]..., input_nodes...]

    if input_v != []
        N.Q = accumulate_f(input_v)
    else
        N.Q = 0.
    end

    for is in input_syns
        if (is.Q - is.THR) < 0.;  throw("Q - THR < 0 in synaps: $(is.id)");  end

        for s in get_synapses_in_range(Subnet(is.NT.dispersion_region, is.NT.range_scale * (is.Q - is.THR)), synapses_for_dispersion)
            dispersion_collection[s] .+= (s.NT.strength, 1)
        end
    end   # dict of: synaps => neurotransmitter change
end
function propergate!(N::Neuron, division_f::Function)
    post_syns = get_synapses(get_posterior_all_cells(N))
    for s in post_syns
        s.Q += N.Q/length(post_syns)
    end
end
function NTChange!(syn::Synaps, input_strength::FloatN)
    syn.NT.strength = retination_percentage * syn.NT.strength + (1-syn.NT.retination_percentage) * input_strength
end

function value_step!(NN::Network, input::Array)
    neurons = get_all_neurons(NN)
    dens = get_all_dendrites(NN)
    APs = get_all_axon_points(NN)
    syns = get_all_synapses(NN)
    in_nodes = get_all_input_nodes(NN)
    out_nodes = get_all_output_nodes(NN)
    dispersion_collection = Dict()

    in_nodes .= input

    println("testing: $dispersion_collection")
    for n in neurons
        accumulate!(n, dropout(synapses, NN.synapsesAccessDropout), dispersion_collection)
    end
    println("testing: $dispersion_collection")

    # this puts the NT(t-1) into the calculation and then calculates NT(t)
    # this can be reversed
    for s in syns
        if s.Q >= s.THR
            s.Q = 0.
        else
            updateQ!(s)
        end

        dispersion_value, n = get(dispersion_collection, s, (1, 1))
        avg_NT_change = dispersion_value/n
        NTChange!(s, avg_NT_change)
    end

    for n in neurons
        propergate!(n)
    end
end
function state_step!(NN::Network)
    # update spatial relation
    # - decay_life!(s, NN.life_decay)
    # - fuse!
    # - split!
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
    println("no Dendrite added to neuron: $(N.id)")
    return nothing
end
function addAxonPoint!(N::Neuron, apDNA::AxonPointDNA)
    if any(ismissing.(N.posteriors))
        for i in eachindex(N.posteriors)
            if ismissing(N.posteriors[i])
                N.posteriors[i] = AllCell(unfold(apDNA))
                return nothing
            end
        end
    end
    println("no AP added to neuron: $(N.id)")
end



# VERIFICATION FUNCTIONS
function rectifyDNA!(NDNA::NeuronDNA, maxLifeTime::Integer)
    clamp!(NDNA.LifeTime.min, 1, maxLifeTime-1)
    clamp!(NDNA.LifeTime.max, NDNA.LifeTime.min, maxLifeTime)
end
function rectifyDNA!(DDNA::DendriteDNA, maxLifeTime::Integer); end
function rectifyDNA!(APDNA::AxonPointDNA, maxLifeTime::Integer); end
function rectifyDNA!(SDNA::SynapsDNA, maxLifeTime::Integer, qDecay_bounds::min_max_pair, threshold_bounds::min_max_pair)
    clamp!(SDNA.QDecay, qDecay_bounds.min, qDecay_bounds.max)
    clamp!(SDNA.THR, threshold_bounds.min, threshold_bounds.max)
    clamp!(SDNA.LifeTime, 1, maxLifeTime)
end
function rectifyDNA!(NDNA::NetworkDNA)
end
function rectifyInitializationPossition!(pos::InitializationPossition, network_max_range::FloatN)
end
