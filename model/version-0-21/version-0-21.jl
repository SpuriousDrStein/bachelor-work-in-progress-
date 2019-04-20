include("structure.jl")
include("functions.jl")

# NETWORK HP's
init_params =   Dict("NETWORK_SIZE"                 => FloatN(5),
                "GLOBAL_STDV"                       => FloatN(1),
                "INIT_POSITION_STDV"                => FloatN(0.8),
                "NEURON_LIFETIME"                   => FloatN(10000),
                "SYNAPTIC_LIFETIME"                 => FloatN(10000),
                "DENDRITE_LIFETIME"                 => FloatN(10000),
                "MAX_AXONPOINT_LIFETIME"            => FloatN(10000),
                "MIN_FUSE_DISTANCE"                 => FloatN(0.2),
                "AP_SINK_ATTRACTIVE_FORCE"          => FloatN(0.9), # force:    AxonPoint's -> ap_sinks
                "DEN_SURGE_REPULSIVE_FORCE"         => FloatN(0.003), # repulsive force of den/occupied input -> den
                "AP_SURGE_REPULSIVE_FORCE"          => FloatN(0.005), # repulsive force of ap/occupied output -> ap
                "INPUT_ATTRACTIVE_FORCE"            => FloatN(5),
                "OUTPUT_ATTRACTIVE_FORCE"           => FloatN(5),
                "MAX_SYNAPTIC_THRESHOLD"            => FloatN(4),
                "MAX_NEURON_THRESHOLD"              => FloatN(6),
                "LITE_LIFE_DECAY"                   => FloatN(1.),
                "HEAVY_LIFE_DECAY"                  => FloatN(5.),
                "NEURON_DESTRUCTION_THRESHOLD"      => FloatN(0.1),
                "SYNAPS_DESTRUCTION_THRESHOLD"      => FloatN(0.05),
                "MAX_NT_STRENGTH"                   => FloatN(1.5),
                "NT_RETAIN_PERCENTAGE"              => FloatN(0.5),
                "MAX_RESISTANCE"                    => FloatN(1.6),
                "N_AP_DEN_INIT_RANGE"               => FloatN(0.5),
                "MAX_NUM_PRIORS"                    => 10,
                "MAX_NUM_POSTERIORS"                => 10,
                "INIT_NUM_NEURONS"                  => 3,
                "INIT_PRIORS"                       => 4,
                "INIT_POSTERIORS"                   => 4,
                "NEURON_INIT_INTERVAL"              => 10000,
                "AP_DEN_INIT_INTERVAL"              => 5000, # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep
                "TOP_BUFFER_LENGTH"                 => 10,
                "DNA_SAMPLE_SIZE"                   => 2,
                "DATA_INPUT_SIZE"                   => 4,
                "DATA_OUTPUT_SIZE"                  => 3)


net_episodes = 400
env_episodes = 5
iterations = 500
parallel_networks = 4
test_episodes = 1
test_iterations = 500
env = :Acrobot
v = :v1
# Acrobot state = [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

best_dna, best_init_pos, metrics = unsupervised_train(net_episodes, env_episodes, iterations, parallel_networks, env, v, init_params)




metrics2 = unsupervised_test(maximum(best_dna)[2], maximum(best_init_pos)[2], test_episodes, test_iterations, env, v, init_params, false)

for j in 1:test_episodes
    for i in 1:length(metrics2["episode_$(j)_positions"])
        t_p = metrics2["episode_$(j)_positions"][i]
        t_c = metrics2["episode_$(j)_connections"][i]

        display_timestep(t_p, t_c, init_params, j, i)
    end
end

# plot fitness and env reward
begin
    m1 = mean([metrics["net_$(n)_current_fitness"] for n in 1:parallel_networks])
    m2 = mean([metrics["net_$(n)_neuron_fitness"] for n in 1:parallel_networks])
    m3 = mean([metrics["net_$(n)_synaps_fitness"] for n in 1:parallel_networks])
    n = 3
    plot([(m1[i]+sum(m1[i-n:i-1])+sum(m1[i+1:i+n]))/(n*2+1) for i in n+1:length(m1)-n], xlabel="episodes", label="mean total fitness", linealpha=0.6, leg=:topleft)
    plot!([(m2[i]+sum(m2[i-n:i-1])+sum(m2[i+1:i+n]))/(n*2+1) for i in n+1:length(m2)-n], label="mean sum neuron fitness")
    plot!([(m3[i]+sum(m3[i-n:i-1])+sum(m3[i+1:i+n]))/(n*2+1) for i in n+1:length(m3)-n], label="mean sum synaps fitness", linealpha=0.7)
    plot!(mean([metrics["net_$(n)_env_reward"] for n in 1:parallel_networks]), label="environment reward", c="purple")
end


# plot computation time and number of synapses
begin
    l = @layout [a c; b]
    p1 = plot(mean([metrics["net_$(n)_num_syns"] for n in 1:parallel_networks]), leg=false, xlabel="episodes", linewidth=0.85, yticks=1:30)
    p2 = plot(mean([metrics["net_$(n)_execution_time"] for n in 1:parallel_networks]), leg=false, xlabel="episodes", c="green", linewidth=0.8)
    p3 = plot(grid=false, showaxis=false, xlims=(0,0))
    plot!([0], label="total number of synapses")
    plot!([0], label="execution time (seconds)", c="green")
    plot(p1, p2, p3, layout=l)
end
