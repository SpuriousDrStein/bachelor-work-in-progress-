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

NET_DNA = NetworkDNA(NETWORK_SIZE,
                    MAX_NEURON_LIFETIME,
                    MAX_SYNAPTIC_LIFETIME,
                    MAX_DENDRITE_LIFETIME,
                    MAX_AXONPOINT_LIFETIME,
                    AP_SINK_ATTRACTIVE_FORCE,
                    NEURON_REPEL_FORCE)


# THE NETWORK PREDICTING NETWORK
ap_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2 # dna samples for 1 netwprl * 3 features * 2 for min and max values
den_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2
nt_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2 + (2 * 3) # + (2 * 3) for 3 min-max pairs per possition in init_pos
syn_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2
n_sample_output_size = DNA_SAMPLE_SIZE * 5 * 2 + (2 * 3) # the same as nt
net_sample_output_size = 3 # net_size, ap_sink_force and neuron_repel_force

# input_size = 100
input_size = sum([ap_sample_output_size..., den_sample_output_size..., nt_sample_output_size..., syn_sample_output_size..., n_sample_output_size...])
latent_size = 40
latent_activation = Flux.sigmoid

DECODER_HIDDENS = [50, 30, 20]
ENCODER_HIDDENS = [90, 60, 40, 30]


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

encoder_model = Flux.Chain(
                Flux.Dense(input_size, ENCODER_HIDDENS[1]),
                [Flux.Dense(ENCODER_HIDDENS[i-1], ENCODER_HIDDENS[i], Flux.relu) for i in 2:length(ENCODER_HIDDENS)]...,
                Flux.Dense(ENCODER_HIDDENS[end], latent_size, latent_activation))

nt_model = Flux.Chain(
            Flux.Dense(latent_size, DECODER_HIDDENS[1], Flux.relu),
            [Flux.Dense(DECODER_HIDDENS[i-1], DECODER_HIDDENS[i], Flux.relu) for i in 2:length(DECODER_HIDDENS)]...,
            Flux.Dense(DECODER_HIDDENS[end], nt_sample_output_size, Flux.relu))

ap_model = Flux.Chain(
            Flux.Dense(latent_size, DECODER_HIDDENS[1], Flux.relu),
            [Flux.Dense(DECODER_HIDDENS[i-1], DECODER_HIDDENS[i], Flux.relu) for i in 2:length(DECODER_HIDDENS)]...,
            Flux.Dense(DECODER_HIDDENS[end], ap_sample_output_size, Flux.relu))

den_model = Flux.Chain(
            Flux.Dense(latent_size, DECODER_HIDDENS[1], Flux.relu),
            [Flux.Dense(DECODER_HIDDENS[i-1], DECODER_HIDDENS[i], Flux.relu) for i in 2:length(DECODER_HIDDENS)]...,
            Flux.Dense(DECODER_HIDDENS[end], den_sample_output_size, Flux.relu))

syn_model = Flux.Chain(
            Flux.Dense(latent_size, DECODER_HIDDENS[1], Flux.relu),
            [Flux.Dense(DECODER_HIDDENS[i-1], DECODER_HIDDENS[i], Flux.relu) for i in 2:length(DECODER_HIDDENS)]...,
            Flux.Dense(DECODER_HIDDENS[end], syn_sample_output_size, Flux.relu))

neuron_model = Flux.Chain(
            Flux.Dense(latent_size, DECODER_HIDDENS[1], Flux.relu),
            [Flux.Dense(DECODER_HIDDENS[i-1], DECODER_HIDDENS[i], Flux.relu) for i in 2:length(DECODER_HIDDENS)]...,
            Flux.Dense(DECODER_HIDDENS[end], n_sample_output_size, Flux.relu))

net_model = Flux.Chain(
            Flux.Dense(latent_size, DECODER_HIDDENS[1], Flux.relu),
            [Flux.Dense(DECODER_HIDDENS[i-1], DECODER_HIDDENS[i], Flux.relu) for i in 2:length(DECODER_HIDDENS)]...,
            Flux.Dense(DECODER_HIDDENS[end], net_sample_output_size, Flux.relu))


z1 = encoder_model(rand(input_size))

net_dna = net_model(z1)
nt_dna = nt_model(z1)
ap_dna = ap_model(z1)
den_dna = den_model(z1)
syn_dna = syn_model(z1)
n_dna = neuron_model(z1)



function collect_dna(NN::Network)
    collection = []
    den_samples = NN.dna_stack.den_dna_samples
    syn_samples = NN.dna_stack.syn_dna_samples
    n_samples = NN.dna_stack.n_dna_samples

    for nts in NN.dna_stack.nt_dna_samples
        append!(collection, nts.init_strength.min)
        append!(collection, nts.init_strength.max)
        append!(collection, nts.dispersion_region.x.min)
        append!(collection, nts.dispersion_region.x.max)
        append!(collection, nts.dispersion_region.y.min)
        append!(collection, nts.dispersion_region.y.max)
        append!(collection, nts.dispersion_region.z.min)
        append!(collection, nts.dispersion_region.z.max)
        append!(collection, nts.dispersion_strength_scale.min)
        append!(collection, nts.dispersion_strength_scale.max)
    end
    for aps in NN.dna_stack.ap_dna_samples
        append!(collection, aps.max_length.min)
end


# reihenfolge

# dna_stack
#   nt_dna_samples
#       init_strength,
#       dispersion_region,
#       dispersion_strength_scale,
#       retain_percentage
#   ap_dna_samples
#       max_length
#       lifeTime
#   den_dna_samples
#       max_length
#       lifeTime
#   syn_dna_samples
#       THR
#       QDecay
#       lifeTime
#   n_dna_samples
#       max_num_priors
#       max_num_posteriors
#       lifeTime
#       dna_and_ap_init_range
#       den_init_interval
#       ap_init_interval
