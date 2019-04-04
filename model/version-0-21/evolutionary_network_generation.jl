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
NEURON_INIT_INTERVAL                = 100
MIN_AP_DEN_INIT_INTERVAL            = 20 # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep


# THE NETWORK PREDICTING NETWORK
ap_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2 # dna samples for 1 netwprl * 3 features * 2 for min and max values
den_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2
nt_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2 + (2 * 3) # + (2 * 3) for 3 min-max pairs per possition in init_pos
syn_sample_output_size = DNA_SAMPLE_SIZE * 3 * 2
n_sample_output_size = DNA_SAMPLE_SIZE * 5 * 2 + (2 * 3) # the same as nt
net_sample_output_size = 3 * 2 # net_size, ap_sink_force and neuron_repel_force

# input_size = 100
input_size = ap_sample_output_size + den_sample_output_size + nt_sample_output_size + syn_sample_output_size + n_sample_output_size
latent_size = 40
latent_activation = Flux.sigmoid
MIN_RECONSTRUCTION_LOSS = 10 # if above this threshold - do more reconstruction effort
OUTPUT_SCALE = 10 # coefficient to scale output of dna prediction on to create a more truthfull reconstruction of the high variance space that is the networks parameters

DECODER_HIDDENS = [50, 30, 20]
ENCODER_HIDDENS = [90, 60, 40, 30]


function initialize(net_dna, dna_stack)
    rectifyDNA!(net_dna)
    nn = unfold(net_dna,
                MAX_NEURON_LIFETIME,
                MAX_SYNAPTIC_LIFETIME,
                MAX_DENDRITE_LIFETIME,
                MAX_AXONPOINT_LIFETIME,
                MIN_FUSE_DISTANCE,
                LIFE_DECAY,
                MAX_NT_DISPERSION_STRENGTH_SCALE,
                MAX_THRESHOLD,
                RANDOM_FLUCTUATION,
                NEURON_INIT_INTERVAL,
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


function train(episodes, iterations, data)
    # devide data

    # "   "

    best_network_dna = []

    # instantiate network
    for e in 1:episodes
        false_x = rand(input_size)
        z1 = encoder_model(false_x)

        nt_dna1 = nt_model(z1)
        ap_dna1 = ap_model(z1)
        den_dna1 = den_model(z1)
        syn_dna1 = syn_model(z1)
        n_dna1 = neuron_model(z1)
        net_dna1 = net_model(z1)
        f_y = [nt_dna1..., ap_dna1..., den_dna1..., syn_dna1..., n_dna1..., net_dna1...] .* OUTPUT_SCALE

        # train encoder and decoder on reconstruction loss
        z2 = encoder_model(f_y)

        nt_dna2 = nt_model(z2)
        ap_dna2 = ap_model(z2)
        den_dna2 = den_model(z2)
        syn_dna2 = syn_model(z2)
        n_dna2 = neuron_model(z2)
        net_dna2 = net_model(z2)

        rec_f_y = [nt_dna2..., ap_dna2..., den_dna2..., syn_dna2..., n_dna2..., net_dna2...] .* OUTPUT_SCALE

        if Flux.mse(f_y, rec_f_y) >= MIN_RECONSTRUCTION_LOSS
            reconstruction_loss = Flux.mse(f_y, rec_f_y)

            # train encoder and decoder on reconstruction loss
        end


        # use network
        for i in 1:iterations


            # train decoder
            # observe efficiency of dna
            den_sinks, ap_sinks = value_step!(NN, [x])
            state_step!(NN, den_sinks, ap_sinks)
            clean_network_components!(NN)
            runtime_instantiate_components!(NN, i)


        end

        # save (total_reward -> network_dna)

    end
end



# run training for N number of networks
# select top percentile based on NN.total_fitness

# train decoder on these top percentiles




# order:
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
#   nn_dna
#       networkSize
#       ap_sink_force
#       neuron_repel_force


function collect_dna(NN::Network, nn_dna::NetworkDNA)
    collection = []
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
        append!(collection, nts.retain_percentage.min)
        append!(collection, nts.retain_percentage.max)
    end
    for aps in NN.dna_stack.ap_dna_samples
        append!(collection, aps.max_length.min)
        append!(collection, aps.max_length.max)
        append!(collection, aps.lifeTime.min)
        append!(collection, aps.lifeTime.max)
    end
    for dens in NN.dna_stack.den_dna_samples
        append!(collection, dens.max_length.min)
        append!(collection, dens.max_length.max)
        append!(collection, dens.lifeTime.min)
        append!(collection, dens.lifeTime.max)
    end
    for syns in NN.dna_stack.syn_dna_samples
        append!(collection, syns.THR.min)
        append!(collection, syns.THR.max)
        append!(collection, syns.QDecay.min)
        append!(collection, syns.QDecay.max)
        append!(collection, syns.lifeTime.min)
        append!(collection, syns.lifeTime.max)
    end
    for ns in NN.dna_stack.n_dna_samples
        append!(collection, ns.max_num_priors.min)
        append!(collection, ns.max_num_priors.max)
        append!(collection, ns.max_num_posteriors.min)
        append!(collection, ns.max_num_posteriors.max)
        append!(collection, ns.lifeTime.min)
        append!(collection, ns.lifeTime.max)
        append!(collection, ns.dna_and_ap_init_range.min)
        append!(collection, ns.dna_and_ap_init_range.max)
        append!(collection, ns.den_init_interval.min)
        append!(collection, ns.den_init_interval.max)
        append!(collection, ns.ap_init_interval.min)
        append!(collection, ns.ap_init_interval.max)
    end

    append!(collection, nn_dna.networkSize.min)
    append!(collection, nn_dna.networkSize.max)
    append!(collection, nn_dna.ap_sink_force.min)
    append!(collection, nn_dna.ap_sink_force.max)
    append!(collection, nn_dna.neuron_repel_force.min)
    append!(collection, nn_dna.neuron_repel_force.max)
    return collection
end
