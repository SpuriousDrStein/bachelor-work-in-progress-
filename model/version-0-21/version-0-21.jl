# version 2

# specifications:
# 1. dendrite and axonPoint max is bound by the network size / 2

import Plots

include("structure.jl")
include("functions.jl")

# NETWORK HP's
init_params =   Dict("NETWORK_SIZE"                 => FloatN(10),
                "MAX_NEURON_LIFETIME"               => FloatN(1000),
                "MAX_SYNAPTIC_LIFETIME"             => FloatN(1000),
                "MAX_DENDRITE_LIFETIME"             => FloatN(100),
                "MAX_AXONPOINT_LIFETIME"            => FloatN(100),
                "MIN_FUSE_DISTANCE"                 => FloatN(0.2),
                "AP_SINK_ATTRACTIVE_FORCE"          => FloatN(0.3), # force: AxonPoint's -> ap_sinks
                # "MAX_NT_DISPERSION_STRENGTH_SCALE"  => FloatN(2.0),
                "MAX_THRESHOLD"                     => FloatN(10),
                "LIFE_DECAY"                        => FloatN(0.1),
                "GLOBAL_STDV"                       => FloatN(5),
                "FITNESS_DECAY"                     => FloatN(0.8),
                "RANDOM_FLUCTUATION"                => FloatN(0.1),
                "LEARNING_RATE"                     => 0.02,
                "INIT_NUM_NEURONS"                  => 20,
                "INIT_MAX_PRIORS"                   => 5,
                "INIT_MAX_POSTERIORS"               => 5, # how many ap's can be created at instantiation time
                "NEURON_INIT_INTERVAL"              => 50,
                "MAX_NT_STRENGTH"                   => FloatN(1.5),
                "MIN_AP_DEN_INIT_INTERVAL"          => 10, # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep
                "MIN_RECONSTRUCTION_LOSS"           => 10, # if above this threshold - do more reconstruction effort
                "DATA_INPUT_SIZE"                   => 4,
                "DATA_OUTPUT_SIZE"                  => 2,
                "OUTPUT_SCALE"                      => 5, # coefficient to scale output of dna prediction on to create a more truthfull reconstruction of the high variance space that is the networks parameters
                "LATENT_SIZE"                       => 10,
                "LATENT_ACTIVATION"                 => Flux.sigmoid,
                "DNA_SAMPLE_SIZE"                   => 1,
                "DECODER_HIDDENS"                   => [15, 20, 25, 30, 35, 40],
                "ENCODER_HIDDENS"                   => [40, 35, 30, 25, 20, 15])


println("network size = $(15 * init_params["DNA_SAMPLE_SIZE"] + 2)") # 15 = #parameters

net_episodes = 200
env_episodes = 50
iterations = 50         # for unsupervised environment (s,a,s') transition can break early
parallel_networks = 20  # how many networks at one time (no multi-threading)
env = :CartPole
v = :v0

best_dna, metrics = unsupervised_train(net_episodes, env_episodes, iterations, parallel_networks, env, v, init_params)

Plots.plot(metrics["best_net_fitness"], label="best network", xlabel="episodes", ylabel="fitness")
Plots.plot(metrics["rec_loss"], xlabel="episodes", ylabel="reconstruction loss")

get_dna(best_dna[2], init_params)

unsupervised_testing(env_episodes, iterations, best_dna[2], env, v, init_params)



import Flux
using Plots

encoder = Flux.Chain(
    Flux.Dense(45, 40, Flux.relu),
    Flux.Dense(40, 40, Flux.relu),
    Flux.Dense(40, 40, Flux.relu),
    Flux.Dense(40, 40, Flux.relu),
    Flux.Dense(40, 20, Flux.relu),
    Flux.Dense(20, 10, Flux.sigmoid))

decoder = Flux.Chain(
    Flux.Dense(10, 20, Flux.relu),
    Flux.Dense(20, 40, Flux.relu),
    Flux.Dense(40, 40, Flux.relu),
    Flux.Dense(40, 40, Flux.relu),
    Flux.Dense(40, 40, Flux.relu),
    Flux.Dense(40, 45, Flux.relu))

p = [Flux.params(decoder)...,Flux.params(encoder)...]

data = [Pair(repeat([rand(45).*50], 2)...) for _ in 1:1000]

loss(x,y) = Flux.mse(decoder(encoder(x)), y)

loss_met = []; for i in 1:200
    Flux.train!(loss, p, data, Flux.Descent(0.03))
    println("i $i -- loss = ",loss(rand(data)...))
    append!(loss_met, [Flux.Tracker.data(loss(rand(data)...))])
end

Plots.plot(loss_met)
