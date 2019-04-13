# version 2

# specifications:
# 1. dendrite and axonPoint max is bound by the network size / 2

import Plots

include("structure.jl")
include("functions.jl")

# NETWORK HP's
init_params =   Dict("NETWORK_SIZE"                 => FloatN(5),
                "GLOBAL_STDV"                       => FloatN(10),
                "MAX_NEURON_LIFETIME"               => FloatN(10000),
                "MAX_SYNAPTIC_LIFETIME"             => FloatN(10000),
                "MAX_DENDRITE_LIFETIME"             => FloatN(1000),
                "MAX_AXONPOINT_LIFETIME"            => FloatN(1000),
                "MIN_FUSE_DISTANCE"                 => FloatN(0.2),
                "AP_SINK_ATTRACTIVE_FORCE"          => FloatN(0.4), # force:    AxonPoint's -> ap_sinks
                "DEN_SURGE_REPULSIVE_FORCE"         => FloatN(0.08), # repulsive force of dendrites
                "MAX_THRESHOLD"                     => FloatN(5),
                "RANDOM_FLUCTUATION"                => FloatN(0.2),
                "LITE_LIFE_DECAY"                   => FloatN(2.),
                "HEAVY_LIFE_DECAY"                  => FloatN(15.),
                "NEURON_DESTRUCTION_THRESHOLD"      => FloatN(2.),
                "SYNAPS_DESTRUCTION_THRESHOLD"      => FloatN(0.7),
                "MAX_NT_STRENGTH"                   => FloatN(1.5),
                "INIT_NUM_NEURONS"                  => 1,
                "INIT_MAX_PRIORS"                   => 7,
                "INIT_MAX_POSTERIORS"               => 2, # how many ap's can be created at instantiation time
                "NEURON_INIT_INTERVAL"              => 70,
                "MIN_AP_DEN_INIT_INTERVAL"          => 15, # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep
                "DATA_INPUT_SIZE"                   => 4,
                "DATA_OUTPUT_SIZE"                  => 2,
                "DNA_SAMPLE_SIZE"                   => 3,
                "TOP_BUFFER_LENGTH"                 => 5,
                "NT_RETAIN_PERCENTAGE"              => FloatN(0.8),
                "NEURON_REPEL_FORCE"                => FloatN(0.05))

                # "LEARNING_RATE"                     => 0.02,
                # "MIN_RECONSTRUCTION_LOSS"           => 10, # if above this threshold - do more reconstruction effort
                # "OUTPUT_SCALE"                      => 1, # coefficient to scale output of dna prediction on to create a more truthfull reconstruction of the high variance space that is the networks parameters
                # "LATENT_SIZE"                       => 10,
                # "LATENT_ACTIVATION"                 => Flux.sigmoid,
                # "DECODER_HIDDENS"                   => [15, 30, 35, 40],
                # "ENCODER_HIDDENS"                   => [40, 35, 30, 15])


net_episodes = 300
env_episodes = 50
iterations = 30         # for unsupervised environment (s,a,s') transition can break early
parallel_networks = 4  # how many networks at one time (no multi-threading)
env = :CartPole
v = :v0

best_dna, metrics = unsupervised_train(net_episodes, env_episodes, iterations, parallel_networks, env, v, init_params)

get_dna(maximum(best_dna)[2], init_params)

Plots.plot([metrics["current_net_$(n)_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="fitness")
Plots.plot([metrics["net_$(n)_final_syns"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of synapses (end of iteration)")
Plots.plot([metrics["net_$(n)_final_neurons"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of neurons (end of iteration)")
Plots.plot([metrics["net_$(n)_final_aps"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of axon points (end of iteration)")
Plots.plot([metrics["net_$(n)_final_dens"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of dendrites (end of iteration)")
Plots.plot([metrics["net_$(n)_neuron_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="sum neuron fitness (end of iteration)")
Plots.plot([metrics["net_$(n)_synaps_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="sum synaps fitness (end of iteration)")


unsupervised_testing(env_episodes, iterations, best_dna[2], env, v, init_params)
