# version 2

# specifications:
# 1. dendrite and axonPoint max is bound by the network size / 2

import Plots

include("structure.jl")
include("functions.jl")

# NETWORK HP's
init_params =   Dict("MAX_NEURON_LIFETIME"          => FloatN(100000),
                "MAX_SYNAPTIC_LIFETIME"             => FloatN(100000),
                "MAX_DENDRITE_LIFETIME"             => FloatN(10000),
                "MAX_AXONPOINT_LIFETIME"            => FloatN(10000),
                "MIN_FUSE_DISTANCE"                 => FloatN(0.5),
                "AP_SINK_ATTRACTIVE_FORCE"          => FloatN(0.7), # force: AxonPoint's -> ap_sinks
                "NEURON_REPEL_FORCE"                => FloatN(0.05),
                "MAX_NT_DISPERSION_STRENGTH_SCALE"  => FloatN(2.0),
                "MAX_THRESHOLD"                     => FloatN(10),
                "LIFE_DECAY"                        => FloatN(0.1),
                "FITNESS_DECAY"                     => FloatN(0.8),
                "RANDOM_FLUCTUATION"                => FloatN(0.1),
                "LEARNING_RATE"                     => 0.02,
                "INIT_NUM_NEURONS"                  => 20,
                "INIT_MAX_PRIORS"                   => 5,
                "INIT_MAX_POSTERIORS"               => 5, # how many ap's can be created at instantiation time
                "NEURON_INIT_INTERVAL"              => 50,
                "MIN_AP_DEN_INIT_INTERVAL"          => 10, # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep
                "MIN_RECONSTRUCTION_LOSS"           => 20, # if above this threshold - do more reconstruction effort
                "DATA_INPUT_SIZE"                   => 4,
                "DATA_OUTPUT_SIZE"                  => 2,
                "OUTPUT_SCALE"                      => 100, # coefficient to scale output of dna prediction on to create a more truthfull reconstruction of the high variance space that is the networks parameters
                "LATENT_SIZE"                       => 30,
                "LATENT_ACTIVATION"                 => Flux.sigmoid,
                "DNA_SAMPLE_SIZE"                   => 3,
                "DECODER_HIDDENS"                   => [50, 100, 200],
                "ENCODER_HIDDENS"                   => [200, 100, 50])



net_episodes = 200
env_episodes = 30
iterations = 50         # for unsupervised can break early
parallel_networks = 5  # how many networks at one time (no multi-threading)
env = :CartPole
v = :v0

best_dna, metrics = unsupervised_train(net_episodes, env_episodes, iterations, parallel_networks, env, v, init_params)

Plots.plot(metrics["best_net_fitness"], label="best network", xlabel="episodes", ylabel="fitness")
Plots.plot(metrics["net_1_final_neurons"], labels=["net 1"], xlabel="episodes", ylabel="remaining neurons")

get_dna(best_dna[2], init_params["DNA_SAMPLE_SIZE"])

unsupervised_testing(env_episodes, iterations, best_dna[2], env, v, init_params)
