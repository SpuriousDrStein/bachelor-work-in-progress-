# version 2

# specifications:
# 1. dendrite and axonPoint max is bound by the network size / 2

import Plots

include("structure.jl")
include("functions.jl")

# NETWORK HP's
init_params =   Dict("NETWORK_SIZE"                 => FloatN(5),
                "GLOBAL_STDV"                       => FloatN(20),
                "MAX_NEURON_LIFETIME"               => FloatN(10000),
                "MAX_SYNAPTIC_LIFETIME"             => FloatN(10000),
                "MAX_DENDRITE_LIFETIME"             => FloatN(1000),
                "MAX_AXONPOINT_LIFETIME"            => FloatN(1000),
                "MIN_FUSE_DISTANCE"                 => FloatN(0.2),
                "AP_SINK_ATTRACTIVE_FORCE"          => FloatN(0.4), # force: AxonPoint's -> ap_sinks
                "MAX_THRESHOLD"                     => FloatN(3),
                "LIFE_DECAY"                        => FloatN(1),
                "FITNESS_DECAY"                     => FloatN(0.9),
                "RANDOM_FLUCTUATION"                => FloatN(0.05),
                "INIT_NUM_NEURONS"                  => 4,
                "INIT_MAX_PRIORS"                   => 5,
                "INIT_MAX_POSTERIORS"               => 5, # how many ap's can be created at instantiation time
                "NEURON_INIT_INTERVAL"              => 50,
                "MAX_NT_STRENGTH"                   => FloatN(1.5),
                "MIN_AP_DEN_INIT_INTERVAL"          => 10, # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep
                "DATA_INPUT_SIZE"                   => 4,
                "DATA_OUTPUT_SIZE"                  => 2,
                "DNA_SAMPLE_SIZE"                   => 10)
                # "LEARNING_RATE"                     => 0.02,
                # "MIN_RECONSTRUCTION_LOSS"           => 10, # if above this threshold - do more reconstruction effort
                # "OUTPUT_SCALE"                      => 1, # coefficient to scale output of dna prediction on to create a more truthfull reconstruction of the high variance space that is the networks parameters
                # "LATENT_SIZE"                       => 10,
                # "LATENT_ACTIVATION"                 => Flux.sigmoid,
                # "DECODER_HIDDENS"                   => [15, 30, 35, 40],
                # "ENCODER_HIDDENS"                   => [40, 35, 30, 15])


net_episodes = 300
env_episodes = 40
iterations = 30         # for unsupervised environment (s,a,s') transition can break early
parallel_networks = 7  # how many networks at one time (no multi-threading)
env = :CartPole
v = :v0

best_dna, metrics = unsupervised_train(net_episodes, env_episodes, iterations, parallel_networks, env, v, init_params)



Plots.plot(metrics["best_net_fitness"], label="best network", xlabel="episodes", ylabel="fitness")
Plots.plot([metrics["net_$(n)_final_syns"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of synapses (end of iteration)")
Plots.plot([metrics["net_$(n)_final_neurons"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of neurons (end of iteration)")
Plots.plot([metrics["net_$(n)_final_aps"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of axon points (end of iteration)")
Plots.plot([metrics["net_$(n)_final_dens"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of dendrites (end of iteration)")
Plots.plot([metrics["net_$(n)_neuron_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="sum neuron fitness (end of iteration)")
Plots.plot([metrics["net_$(n)_synaps_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="sum synaps fitness (end of iteration)")


get_dna(best_dna[2], init_params)

unsupervised_testing(env_episodes, iterations, best_dna[2], env, v, init_params)





# tests ...

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
    (x) -> x./mean(x),
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
