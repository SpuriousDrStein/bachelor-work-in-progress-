import Flux

NETWORK_SIZE                        = FloatN(1000)
MAX_NEURON_LIFETIME                 = FloatN(100000)
MAX_SYNAPTIC_LIFETIME               = FloatN(100000)
MAX_DENDRITE_LIFETIME               = FloatN(10000)
MAX_AXONPOINT_LIFETIME              = FloatN(10000)
MIN_FUSE_DISTANCE                   = FloatN(0.1)
AP_SINK_ATTRACTIVE_FORCE            = FloatN(0.5) # force: AxonPoint's -> ap_sinks
NEURON_REPEL_FORCE                  = FloatN(0.05)
MAX_NT_DISPERSION_STRENGTH_SCALE    = FloatN(2.0)
MAX_THRESHOLD                       = FloatN(10)
LIFE_DECAY                          = FloatN(0.1)
FITNESS_DECAY                       = FloatN(0.99)
RANDOM_FLUCTUATION                  = FloatN(0.05)
INIT_NUM_NEURONS                    = 20
DNA_SAMPLE_SIZE                     = 4
INIT_MAX_PRIORS                     = 5
INIT_MAX_POSTERIORS                 = 5 # how many ap's can be created at instantiation time


function initialize(net_dna, dna_stack)
    rectifyDNA!(net_dna)
    nn = unfold(net_dna,
                MIN_FUSE_DISTANCE,
                LIFE_DECAY,
                MAX_NT_DISPERSION_STRENGTH_SCALE,
                MAX_THRESHOLD,
                RANDOM_FLUCTUATION,
                dna_stack,
                fitness_decay=fitness_decay)

    rectifyDNA!(nn.dna_stack, nn)
    populate_network!(nn, INIT_NUM_NEURONS, INIT_MAX_PRIORS, INIT_MAX_POSTERIORS)
    return nn
end

# sample dna_stack samples
# create net_dna with hp's and dna_stack
# watch performance n times
# train on m best samples

# THE REAL NETWORK
output_size = [DNA_SAMPLE_SIZE, 18] # 18 parameter





# NN = unfold(NN_dna,
#             min_fuse_distance,
#             init_life_decay,
#             max_nt_dispersion_strength_scale,
#             max_threshold,
#             dna_stack,
#             fitness_decay=0.99)

# nt_dna_samples ->
#     init_strength,
#     dispersion_region,
#     dispersion_strength_scale,
#     retain_percentage
# ap_dna_samples ->
#     max_length
#     lifeTime
#     init_pos
# den_dna_samples ->
#     max_length
#     lifeTime
#     init_pos
# syn_dna_samples ->
#     THR
#     QDecay
#     lifeTime
#     NT
# n_dna_samples ->
#     init_pos
#     max_num_priors
#     max_num_posteriors
#     lifeTime
