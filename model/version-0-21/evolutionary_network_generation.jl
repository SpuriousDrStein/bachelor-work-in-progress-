import Flux

NETWORK_SIZE
maxNeuronLifeTime               = FloatN(100000)
maxSynapsLifeTime               = FloatN(100000)
maxDendriteLifeTime             = FloatN(10000)
maxAxonPointLifeTime            = FloatN(10000)
minFuseDistance                 = FloatN(0.1)
ap_sink_attractive_force        = # force: AxonPoint's -> ap_sinks
neuron_repel_force              = FloatN(0.05)
max_nt_dispersion_strength_scale= FloatN(2.0)
max_threshold                   = FloatN(10)
life_decay                      = FloatN(0.1)
fitness_decay                   = FloatN(0.99)
init_num_neurons                = 20
dna_sample_array_length         = 4
init_max_posteriors             = 5 # how many ap's can be created at instantiation time
init_max_priors                 = 5


function initialize(net_dna)
    rectifyDNA!(net_dna)
    nn = unfold(net_dna,
                min_fuse_distance,
                init_life_decay,
                max_nt_dispersion_strength_scale,
                max_threshold,
                dna_stack,
                fitness_decay=fitness_decay)

    rectifyDNA!(nn.dna_stack, nn)
    populate_network!(nn, init_num_neurons, init_max_priors, init_max_posteriors)

    return nn
end

# sample dna_stack samples
# create net_dna with hp's and dna_stack
# watch performance n times
# train on m best samples

nt_dna_samples ->
    init_strength,
    dispersion_region,
    dispersion_strength_scale,
    retain_percentage
ap_dna_samples ->
    max_length
    lifeTime
    init_pos
den_dna_samples ->
    max_length
    lifeTime
    init_pos
syn_dna_samples ->
    THR
    QDecay
    lifeTime
    NT
n_dna_samples ->
    init_pos
    max_num_priors
    max_num_posteriors
    lifeTime
