    include("structure.jl")
    include("functions.jl")

    # NETWORK HP's
    init_params =   Dict("NETWORK_SIZE"                 => FloatN(6),
                    "GLOBAL_STDV"                       => FloatN(3),
                    "MAX_NEURON_LIFETIME"               => FloatN(5000),
                    "MAX_SYNAPTIC_LIFETIME"             => FloatN(3000),
                    "MAX_DENDRITE_LIFETIME"             => FloatN(1000),
                    "MAX_AXONPOINT_LIFETIME"            => FloatN(1000),
                    "MIN_FUSE_DISTANCE"                 => FloatN(0.2),
                    "AP_SINK_ATTRACTIVE_FORCE"          => FloatN(1), # force:    AxonPoint's -> ap_sinks
                    "DEN_SURGE_REPULSIVE_FORCE"         => FloatN(0.02), # repulsive force of den/occupied input -> den
                    "AP_SURGE_REPULSIVE_FORCE"          => FloatN(0.02), # repulsive force of ap/occupied output -> ap
                    "MAX_SYNAPTIC_THRESHOLD"            => FloatN(2),
                    "MAX_NEURON_THRESHOLD"              => FloatN(1),
                    "RANDOM_FLUCTUATION"                => FloatN(0.08),
                    "LITE_LIFE_DECAY"                   => FloatN(1.),
                    "HEAVY_LIFE_DECAY"                  => FloatN(3.),
                    "NEURON_DESTRUCTION_THRESHOLD"      => FloatN(0.1),
                    "SYNAPS_DESTRUCTION_THRESHOLD"      => FloatN(0.05),
                    "MAX_NT_STRENGTH"                   => FloatN(1.5),
                    "NT_RETAIN_PERCENTAGE"              => FloatN(0.5),
                    "NEURON_REPEL_FORCE"                => FloatN(0),
                    "MAX_MAX_RESISTANCE"                => FloatN(5.),
                    "INIT_NUM_NEURONS"                  => 1,
                    "MAX_PRIORS"                        => 10,
                    "MAX_POSTERIORS"                    => 6,
                    "NEURON_INIT_INTERVAL"              => 100,
                    "MIN_AP_DEN_INIT_INTERVAL"          => 5, # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep
                    "TOP_BUFFER_LENGTH"                 => 10,
                    "DNA_SAMPLE_SIZE"                   => 4,
                    "DATA_INPUT_SIZE"                   => 4,
                    "DATA_OUTPUT_SIZE"                  => 3)


    net_episodes = 10
    env_episodes = 30
    iterations = 60
    parallel_networks = 4
    env = :Acrobot
    v = :v1
    # Acrobot state = [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

best_dna, metrics = unsupervised_train(net_episodes, env_episodes, iterations, parallel_networks, env, v, init_params)

display_network(metrics, 1, [net_episodes, net_episodes])


Plots.plot([metrics["net_$(n)_current_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="fitness")
Plots.plot([metrics["net_$(n)_num_syns"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of synapses ")
Plots.plot([metrics["net_$(n)_num_neurons"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of neurons ")
Plots.plot([metrics["net_$(n)_num_aps"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of axon points ")
Plots.plot([metrics["net_$(n)_num_dens"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="number of dendrites ")
Plots.plot([metrics["net_$(n)_neuron_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="sum neuron fitness ")
Plots.plot([metrics["net_$(n)_synaps_fitness"] for n in 1:parallel_networks], labels=["net $n" for n in 1:parallel_networks], xlabel="episodes", ylabel="sum synaps fitness ")
Plots.scatter(metrics["net_1_output_signals"], leg=false, xlabel="output node", ylabel="sum of output values", xticks=[1,2,3])

unsupervised_testing(env_episodes, iterations, best_dna[2], env, v, init_params)



# define the Lorenz attractor
mutable struct Lorenz
    dt; σ; ρ; β; x; y; z
end

function step!(l::Lorenz)
    dx = l.σ*(l.y - l.x)       ; l.x += l.dt * dx
    dy = l.x*(l.ρ - l.z) - l.y ; l.y += l.dt * dy
    dz = l.x*l.y - l.β*l.z     ; l.z += l.dt * dz
end

attractor = Lorenz((dt = 0.02, σ = 10., ρ = 28., β = 8//3, x = 1., y = 1., z = 1.)...)


# initialize a 3D plot with 1 empty series
plt = plot3d(1, xlim=(-25,25), ylim=(-25,25), zlim=(0,50),
                title = "Lorenz Attractor", marker = 2)

# build an animated gif by pushing new points to the plot, saving every 10th frame
@gif for i=1:1500
    step!(attractor)
    push!(plt, attractor.x, attractor.y, attractor.z)
end every 10
