
function unfold(dna::DendriteDNA)
    return Dendrite(sample(dna.possition), sample(dna.max_length), Force(0,0,0,0), sample(dna.lifeTime))
end

function unfold(dna::SynapsDNA, possition::Possition, NT::NeuroTransmitter, life_decay::Integer)::Synaps
    q_dec = sample(dna.QDecay)
    thr = sample(dna.THR)
    lifetime = sample(dna.LifeTime)
    return Synaps(possition, 0., q_dec, thr, NT, lifetime, life_decay, 0)
end

function unfold(dna::NeuronDNA, NT::NeuroTransmitter, life_decay::FloatN; id_counter=N_id_counter)::Neuron
    id = copy(id_counter)
    id_counter += 1

    pos = sample(dna.possition)
    lifetime = sample(dna.lifeTime)
    num_priors = sample(dna.num_priors)
    num_posteriors = sample(dna.num_posteriors)
    return Neuron(pos, Force(0,0,0,0), 0., [missing for _ in 1:num_priors], [missing for _ in 1:num_posteriors], NT, lifetime, life_decay, 0, id)
end
