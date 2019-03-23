
function unfold(dna::DendriteDNA)
    return Dendrite(sample(dna.possition), sample(dna.max_length), sample(dna.lifeTime))
end
function unfold(dna::SynapsDNA, possition::FloatN, NT::NeuroTransmitter)::Synaps
    q_dec = sample(dna.QDecay)
    thr = sample(dna.THR)
    lifetime = sample(dna.LifeTime)
    return Synaps(possition, 0., q_dec, thr, NT, lifetime, 0)
end
function unfold(dna::NeuronDNA, NT::NeuroTransmitter, life_decay)::Neuron
    pos = sample(dna.possition)
    lifetime = sample(dna.lifeTime)
    num_priors = sample(dna.num_priors)
    num_posteriors = sample(dna.num_posteriors)
    return Neuron(pos, Force(0,0,0,0), [missing for _ in 1:num_priors], [missing for _ in 1:num_posteriors], NT, lifetime, life_decay, 0)
end



# RECTIFATION FUNCTIONS
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
