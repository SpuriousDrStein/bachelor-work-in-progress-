include("structure.jl")
include("functions.jl")

# NETWORK HP's
init_params =   Dict("NETWORK_SIZE"                 => FloatN(5),
                "GLOBAL_STDV"                       => FloatN(0.5),
                "INIT_POSITION_STDV"                => FloatN(2),
                "NEURON_LIFETIME"                   => FloatN(100000),
                "SYNAPTIC_LIFETIME"                 => FloatN(100000),
                "DENDRITE_LIFETIME"                 => FloatN(100000),
                "AXONPOINT_LIFETIME"                => FloatN(100000),
                "MIN_FUSE_DISTANCE"                 => FloatN(0.3),
                "AP_SINK_ATTRACTIVE_FORCE"          => FloatN(0.9), # force:    AxonPoint's -> ap_sinks
                "DEN_SURGE_REPULSIVE_FORCE"         => FloatN(0.003), # repulsive force of den/occupied input -> den
                "AP_SURGE_REPULSIVE_FORCE"          => FloatN(0.005), # repulsive force of ap/occupied output -> ap
                "INPUT_ATTRACTIVE_FORCE"            => FloatN(7),
                "OUTPUT_ATTRACTIVE_FORCE"           => FloatN(5),
                "MAX_SYNAPTIC_THRESHOLD"            => FloatN(4),
                "MAX_NEURON_THRESHOLD"              => FloatN(6),
                "NEURON_REPEL_FORCE"                => FloatN(0),
                "LITE_LIFE_DECAY"                   => FloatN(1.),
                "HEAVY_LIFE_DECAY"                  => FloatN(1.),
                "NEURON_DESTRUCTION_THRESHOLD"      => FloatN(0.1),
                "SYNAPS_DESTRUCTION_THRESHOLD"      => FloatN(0.05),
                "MAX_NT_STRENGTH"                   => FloatN(1.5),
                "NT_RETAIN_PERCENTAGE"              => FloatN(0.8),
                "MAX_RESISTANCE"                    => FloatN(1.6),
                "N_AP_DEN_INIT_RANGE"               => FloatN(0.2),
                "MAX_NUM_PRIORS"                    => 5,
                "MAX_NUM_POSTERIORS"                => 5,
                "INIT_NUM_NEURONS"                  => 3,
                "INIT_PRIORS"                       => 3,
                "INIT_POSTERIORS"                   => 3,
                "NEURON_INIT_INTERVAL"              => 10000,
                "AP_DEN_INIT_INTERVAL"              => 5000, # a minimum to negate the possibility of calling the add_dendrite or add_axon_point function every timestep
                "TOP_BUFFER_LENGTH"                 => 10,
                "DNA_SAMPLE_SIZE"                   => 2,
                "DATA_INPUT_SIZE"                   => 6,
                "DATA_OUTPUT_SIZE"                  => 2)


net_episodes = 150
env_episodes = 400
iterations = 30
parallel_networks = 6
test_episodes = 400
env = :CartPole
v = :v1
# Acrobot state = [cos(theta1) sin(theta1) cos(theta2) sin(theta2) thetaDot1 thetaDot2]

best_dna, best_init_pos, metrics = unsupervised_train(net_episodes, env_episodes, iterations, parallel_networks, env, v, init_params)




println("execution time = ", sum(sum(metrics["net_$(n)_execution_time"] for n in 1:parallel_networks))/60, " minutes")

ind = 1; metrics2 = unsupervised_test(sort(best_dna)[ind][2], sort(best_init_pos)[ind][2], test_episodes, test_iterations, env, v, init_params, false)
for j in 1:30
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
    p1 = plot(mean([metrics["net_$(n)_num_syns"] for n in 1:parallel_networks]), leg=false, xlabel="episodes", c="red", linewidth=0.85, yticks=1:200)
    p2 = plot(mean([metrics["net_$(n)_execution_time"] for n in 1:parallel_networks]), leg=false, xlabel="episodes", c="green", linewidth=0.8)
    p3 = plot(grid=false, showaxis=false, xlims=(0,0))
    plot!([0], label="total number of synapses", c="red")
    plot!([0], label="execution time (seconds)", c="green")
    plot(p1, p2, p3, layout=l)
end



f(c) = begin
    wait(c)
    schedule(Task(sum(1:20)))
end

c = [Condition() for _ in 1:8]
T = [f(c[i]) for i in 1:8]

wait(T[1])
notify(c[1])




function unsupervised_train(net_episodes::Integer, env_episodes::Integer, iterations::Integer, parallel_networks::Integer, env, env_version, params::Dict)
    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    dna_ss = params["DNA_SAMPLE_SIZE"]
    best_nets_buf = [[-9999, get_random_set(params)[2]] for _ in 1:params["TOP_BUFFER_LENGTH"]]
    best_init_pos = [[-9999, get_random_init_positions(params)[2]] for _ in 1:params["TOP_BUFFER_LENGTH"]]

    metrics = Dict([("net_$(n)_current_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_neuron_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_synaps_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_num_neurons" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_num_syns" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_execution_time" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_env_reward" => []) for n in 1:parallel_networks]...)


    for e in 1:net_episodes
        nets = []
        net_poss = []
        println("episode: $e")

        for n in 1:parallel_networks
            if rand() > clamp(1/log(e), 0, 1)
                if rand() > clamp(1/log(e), 0, 1)
                    println("$(n) sample dna combinations")
                    dna_stack, x = sample_from_sets_random([bnb[2] for bnb in best_nets_buf], params)
                else
                    println("$(n) sample dna normal")
                    net_distro = softmax([bn[1]/mean([i[1] for i in best_nets_buf]) for bn in best_nets_buf])
                    net_params = Distributions.sample(Random.GLOBAL_RNG, best_nets_buf, StatsBase.Weights(net_distro))[2]
                    dna_stack, x = sample_from_set_scaled(net_params, params)
                end
            else
                println("$(n) sample dna random")
                dna_stack, x = get_random_set(params)
            end

            if rand() > clamp(1/log(e), 0, 1)
                if rand() > clamp(1/log(e), 0, 1)
                    println("$(n) sample position combinations")
                    init_positions, p = sample_init_positions_from_sets_random([bn[2] for bn in best_init_pos], params)
                else
                    println("$(n) sample position normal")
                    pos_distro = softmax([bp[1]/mean([i[1] for i in best_init_pos]) for bp in best_init_pos])
                    pos_params = Distributions.sample(Random.GLOBAL_RNG, best_init_pos, StatsBase.Weights(pos_distro))[2]
                    init_positions, p = sample_init_positions_from_set(pos_params, params)
                end
            else
                println("$(n) sample position random")
                init_positions, p = get_random_init_positions(params)
            end

            net = initialize(dna_stack, init_positions, params)
            _ = rectify!(net.dna_stack, net)

            I = 1 # for counting iterations
            t = time()
            sum_env_rewards = 0
            # total_output = [0. for _ in 1:params["DATA_OUTPUT_SIZE"]]

            # training
            for ee in 1:env_episodes
                s = OpenAIGym.reset!(env)

                # reset_network_components!(net)

                for i in 1:iterations
                    # for Acrobot
                    state = Array(s)
                    state = [s[1]>0,s[1]<0, s[2]>0,s[2]<0, s[3]>0,s[3]<0]# [(s[2] > 0), (s[2] < 0), (s[4] > 0), (s[4] < 0)]

                    for _ in 1:3
                        den_sinks, den_surges, ap_sinks, ap_surges = value_step!(net, state)
                        state_step!(net, den_sinks, den_surges, ap_sinks, ap_surges)
                        clean_network_components!(net)
                        # runtime_instantiate_components!(net, I)
                    end

                    I += 1


                    if I % 100 == 0
                        if !all([inn.referenced for inn in get_input_nodes(net)]) && get_all_neurons(net) != []
                            net.total_fitness -= 500
                            # add_dendrite!(net, rand(get_all_neurons(net)))
                        end
                        if !all([onn.referenced for onn in get_output_nodes(net)]) && get_all_neurons(net) != []
                            net.total_fitness -= 500
                            # add_axon_point!(net, rand(get_all_neurons(net)))
                        end
                    end

                    # println([nnn.Q for nnn in get_all_neurons(net)])

                    out = [on.value for on in get_output_nodes(net)]
                    a = action_space[argmax(out)]
                    r, s = OpenAIGym.step!(env, a)

                    net.total_fitness += r
                    sum_env_rewards += r
                    # if r > 0
                    # else
                    #     net.total_fitness += 0
                    # end

                    if env.done
                        break
                    end
                end
            end

            append!(metrics["net_$(n)_env_reward"], [sum_env_rewards])
            append!(metrics["net_$(n)_execution_time"], [time()-t])
            append!(metrics["net_$(n)_num_neurons"], [net.n_counter])
            append!(metrics["net_$(n)_num_syns"], [net.syn_counter])
            append!(metrics["net_$(n)_neuron_fitness"], [sum([n.total_fitness for n in get_all_neurons(net)])])
            if get_all_all_cells(net) != [] && get_synapses(get_all_all_cells(net)) != []
                append!(metrics["net_$(n)_synaps_fitness"], [sum([s.total_fitness for s in get_synapses(get_all_all_cells(net))])])
            else
                append!(metrics["net_$(n)_synaps_fitness"], [0])
            end

            tally_up_fitness!(net)


            append!(nets, [(net.total_fitness => copy(x))])
            append!(net_poss, [(net.total_fitness => copy(p))])
        end

        for cn in nets
            if cn[1] > sort(best_nets_buf)[1][1]
                sort(best_nets_buf)[1] .= copy.(cn)
            end
        end
        for cip in net_poss
            if cip[1] > best_init_pos[1][1]
                sort(best_init_pos)[1] .= copy.(cip)
            end
        end

        println("best_net_rewards: $([bn[1] for bn in best_nets_buf])")

        for pln in 1:parallel_networks
            append!(metrics["net_$(pln)_current_fitness"], [nets[pln][1]])
        end

    end
    return best_nets_buf, best_init_pos, metrics
end
