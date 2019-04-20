include(".\\..\\..\\global_utility_functions\\activation_functions.jl")

import OpenAIGym
import Random
import Distributions
import StatsBase

# FUNCTIONS
function initialize(dna_stack, init_positions, params)
    nn = initialize_network(
                params["NETWORK_SIZE"],
                params["GLOBAL_STDV"],
                params["MAX_NEURON_LIFETIME"],
                params["MAX_SYNAPTIC_LIFETIME"],
                params["MAX_DENDRITE_LIFETIME"],
                params["MAX_AXONPOINT_LIFETIME"],
                params["MIN_FUSE_DISTANCE"],
                params["AP_SINK_ATTRACTIVE_FORCE"],
                params["AP_SURGE_REPULSIVE_FORCE"],
                params["DEN_SURGE_REPULSIVE_FORCE"],
                params["INPUT_ATTRACTIVE_FORCE"],
                params["OUTPUT_ATTRACTIVE_FORCE"],
                params["NEURON_REPEL_FORCE"],
                params["MAX_NT_STRENGTH"],
                params["MAX_NEURON_THRESHOLD"],
                params["MAX_SYNAPTIC_THRESHOLD"],
                params["RANDOM_FLUCTUATION"],
                params["LITE_LIFE_DECAY"],
                params["HEAVY_LIFE_DECAY"],
                params["NT_RETAIN_PERCENTAGE"],
                params["NEURON_INIT_INTERVAL"],
                params["MIN_AP_DEN_INIT_INTERVAL"],
                params["NEURON_DESTRUCTION_THRESHOLD"],
                params["SYNAPS_DESTRUCTION_THRESHOLD"],
                dna_stack)

    input_nodes = [AllCell(InputNode(init_positions[i], 0., false)) for i in 1:params["DATA_INPUT_SIZE"]] # InputNode(Position(hns + rand(Uniform(0, 1))*hns, hns + rand(Uniform(0, 1))*hns)
    out_nodes = [AllCell(OutputNode(init_positions[params["DATA_INPUT_SIZE"]+i], 0., false)) for i in 1:params["DATA_OUTPUT_SIZE"]] # Position(-hns + rand(Uniform(-1, 0))*hns, -hns + rand(Uniform(-1, 0))*hns)

    populate_network!(nn, params["INIT_NUM_NEURONS"], params["INIT_PRIORS"], params["INIT_POSTERIORS"], init_positions[params["DATA_INPUT_SIZE"]+params["DATA_OUTPUT_SIZE"]:end])
    nn.IO_components = [input_nodes..., out_nodes...]
    return nn
end

function get_positions(x, params)
    inn = params["INIT_NUM_NEURONS"] * (params["INIT_PRIORS"] + params["INIT_POSTERIORS"]) + params["DATA_INPUT_SIZE"] + params["DATA_OUTPUT_SIZE"]
    ps = []
    for ip in 1:2:inn*2
        append!(ps, [Position(x[ip:ip+1]...)])
    end
    return ps
end

function sample_init_positions_from_set(x, params)
    inn = params["INIT_NUM_NEURONS"] * (params["INIT_PRIORS"] + params["INIT_POSTERIORS"]) + params["DATA_INPUT_SIZE"] + params["DATA_OUTPUT_SIZE"]
    ns = params["NETWORK_SIZE"]
    ps = []
    xx = [clamp(rand(Normal(ax, params["INIT_POSITION_STDV"])), -ns, ns) for ax in x]
    for ip in 1:2:inn*2
        append!(ps, [Position(xx[ip:ip+1]...)])
    end
    return ps, xx
end
function get_random_init_positions(params)
    inn = params["INIT_NUM_NEURONS"] * (params["INIT_PRIORS"] + params["INIT_POSTERIORS"]) + params["DATA_INPUT_SIZE"] + params["DATA_OUTPUT_SIZE"]
    ns = params["NETWORK_SIZE"]
    ps = []
    x = [rand(Uniform(-ns, ns)) for p in 1:inn*2]
    for ip in 1:2:inn*2
        append!(ps, [Position(x[ip:ip+1]...)])
    end
    return ps, x
end


get_nts_from_dna(dna, d) = [NeuroTransmitterDNA(dna[1:d][j]) for j in 1:d]
get_syns_from_dna(dna, d) = [SynapsDNA(dna[d+1:d*2][j], dna[d*2+1:d*3][j], dna[d*3+1:d*4][j]) for j in 1:d]
get_ns_from_dna(dna, d, max_pap) = [NeuronDNA(dna[d*4+1:d*5][j]) for j in 1:d]

function get_dna(x, params)
    d = params["DNA_SAMPLE_SIZE"] # because shorter indexing

    ntd = get_nts_from_dna(x, d)
    sd = get_syns_from_dna(x, d)
    nd = get_ns_from_dna(x, d)

    dna_stack = DNAStack(ntd, sd, nd)

    return dna_stack
end



function sample_from_set_plain(x, p)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Normal(x[i], stdv) for i in 1:d]

    syn_thr   = [rand(Normal(x[i], stdv) for i in d+1:(d*2)]
    syn_r_rec = [rand(Normal(x[i], stdv) for i in (d*2)+1:(d*3)]
    syn_maxR  = [rand(Normal(x[i], stdv) for i in (d*3)+1:(d*4)]

    n_thr  = [rand(Normal(x[i], stdv) for i in (d*4)+1:(d*5)]
    n_ap_den_init_r = [rand(Normal(x[i], stdv) for i in (d*7)+1:(d*8)]

    xx = [nt_init_strs..., syn_thr..., syn_r_rec..., syn_maxR..., n_thr..., n_den_init_int..., n_ap_init_int...]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_thr[i], p["MAX_NUM_PRIORS"], p["MAX_NUM_POSTERIORS"], p["AP_DEN_INIT_INTERVAL"], p["AP_DEN_INIT_INTERVAL"], p["N_AP_DEN_INIT_RANGE"])])
    end

    return dna_stack, xx
end

function get_random_set(p)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing

    nt_init_strs = [rand(Uniform(0.1, p["MAX_NT_STRENGTH"])) for i in 1:d]

    syn_thr   = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in d+1:(d*2)]
    syn_r_rec = [rand(Uniform(1.1, 100.)) for i in (d*2)+1:(d*3)]
    syn_maxR  = [rand(Uniform(1., p["MAX_MAX_RESISTANCE"])) for i in (d*3)+1:(d*4)]

    n_thr  = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in (d*4)+1:(d*5)]
    n_ap_den_init_r = [rand(Uniform(1., p["NETWORK_SIZE"])) for i in (d*7)+1:(d*8)]


    xx = [nt_init_strs..., syn_thr..., syn_r_rec..., syn_maxR..., n_thr..., n_den_init_int..., n_ap_init_int...]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_thr[i], p["MAX_NUM_PRIORS"], p["MAX_NUM_POSTERIORS"], p["AP_DEN_INIT_INTERVAL"], p["AP_DEN_INIT_INTERVAL"], p["N_AP_DEN_INIT_RANGE"])])
    end

    return dna_stack, xx
end

function sample_from_set_scaled(x, p; scl=1.)
    # d = p["DNA_SAMPLE_SIZE"] # because shorter indexing
    # stdv = p["GLOBAL_STDV"]
    #
    # nt_init_strs = [rand(Uniform(0.1, p["MAX_NT_STRENGTH"])) for i in 1:d]
    #
    # syn_thr   = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in d+1:(d*2)]
    # syn_r_rec = [rand(Uniform(1.1, 100.)) for i in (d*2)+1:(d*3)]
    # syn_maxR  = [rand(Uniform(1., p["MAX_MAX_RESISTANCE"])) for i in (d*3)+1:(d*4)]
    #
    # n_thr  = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in (d*4)+1:(d*5)]
    # n_ap_den_init_r = [rand(Uniform(1., p["NETWORK_SIZE"])) for i in (d*7)+1:(d*8)]
    #
    #
    # xx = [nt_init_strs..., syn_thr..., syn_r_rec..., syn_maxR..., n_thr..., n_den_init_int..., n_ap_init_int...]
    #
    # dna_stack = DNAStack([], [], [], [], [])
    # for i in 1:d
    #     append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
    #     append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i])])
    #     append!(dna_stack.n_dna_samples, [NeuronDNA(n_thr[i], p["MAX_NUM_PRIORS"], p["MAX_NUM_POSTERIORS"], p["AP_DEN_INIT_INTERVAL"], p["AP_DEN_INIT_INTERVAL"], p["N_AP_DEN_INIT_RANGE"])])
    # end
    #
    # return dna_stack, xx
end

function sample_from_set_decay(x, p, j)
end

function sample_from_set_scaled_decay(x, p, j; scl=0.3)
end


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
                net_distro = softmax([bn[1]/mean([i[1] for i in best_nets_buf]) for bn in best_nets_buf])
                net_params = Distributions.sample(Random.GLOBAL_RNG, best_nets_buf, StatsBase.Weights(net_distro))[2]
                dna_stack, x = sample_from_set_scaled(net_params, params)
            else
                dna_stack, x = get_random_set(params)
            end

            if rand() > clamp(1/log(e), 0, 1)
                pos_distro = softmax([bp[1]/mean([i[1] for i in best_init_pos]) for bp in best_init_pos])
                pos_params = Distributions.sample(Random.GLOBAL_RNG, best_init_pos, StatsBase.Weights(pos_distro))[2]
                init_positions, p = sample_init_positions_from_set(pos_params, params)
            else
                init_positions, p = get_random_init_positions(params)
            end

            net = initialize(dna_stack, init_positions, params)
            net.total_fitness -= rectify!(net.dna_stack, net) * 500

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
                    state = [(s[2] > 0), (s[2] < 0), (s[4] > 0), (s[4] < 0)]  # [(s[2] > 0) * abs(s[2]), (s[2] < 0) * abs(s[2]), (s[4] > 0) * abs(s[4]), (s[4] < 0) * abs(s[4])]

                    den_sinks, den_surges, ap_sinks, ap_surges = value_step!(net, state)
                    state_step!(net, den_sinks, den_surges, ap_sinks, ap_surges)
                    clean_network_components!(net)
                    runtime_instantiate_components!(net, I)
                    I += 1


                    if I % 100 == 0
                        if !all([inn.referenced for inn in get_input_nodes(net)]) && get_all_neurons(net) != []
                            add_dendrite!(net, rand(get_all_neurons(net)))
                        end
                        if !all([onn.referenced for onn in get_output_nodes(net)]) && get_all_neurons(net) != []
                            add_axon_point!(net, rand(get_all_neurons(net)))
                        end
                    end


                    out = [on.value for on in get_output_nodes(net)]
                    a = action_space[argmax(out)]
                    r, s = OpenAIGym.step!(env, a)

                    if r > 0
                        net.total_fitness += 10000
                        sum_env_rewards += 10000
                    else
                        net.total_fitness += r
                    end

                    if env.done
                        break
                    end
                end
            end

            # println("net: $n --- time: $(time()-t)")
            # println("#neurons     -- $(net.n_counter)")
            # println("#dendrites   -- $(net.den_counter)")
            # println("#axon points -- $(net.ap_counter)")
            # println("#synapses    -- $(net.syn_counter)")

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


function unsupervised_test(sample, init_positions, episodes::Integer, iterations::Integer, env, env_version, params::Dict, render)
    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    dna_stack = get_dna(sample, params)
    init_positions = get_positions(init_positions, params)

    net = initialize(dna_stack, init_positions, params)

    I = 1 # for counting iterations
    metrics = Dict()

    for e in 1:episodes
        s = OpenAIGym.reset!(env)
        # reset_network_components!(net)

        for i in 1:iterations
            # for Acrobot
            state = Array(s)
            state = [(s[2] > 0), (s[2] < 0), (s[4] > 0), (s[4] < 0)]  # [(s[2] > 0) * abs(s[2]), (s[2] < 0) * abs(s[2]), (s[4] > 0) * abs(s[4]), (s[4] < 0) * abs(s[4])]

            positions, connections = get_all_relations(net) # returns [np, app, denp, synp, inp, outp], connections
            if "episode_$(e)_positions" in keys(metrics)
                append!(metrics["episode_$(e)_positions"], [positions])
            else
                metrics["episode_$(e)_positions"] = [positions]
            end
            if "episode_$(e)_connections" in keys(metrics)
                append!(metrics["episode_$(e)_connections"], [connections])
            else
                metrics["episode_$(e)_connections"] = [connections]
            end


            den_sinks, den_surges, ap_sinks, ap_surges = value_step!(net, state)
            state_step!(net, den_sinks, den_surges, ap_sinks, ap_surges)
            clean_network_components!(net)
            runtime_instantiate_components!(net, I)
            I += 1



            if I % 100 == 0
                if !all([inn.referenced for inn in get_input_nodes(net)]) && get_all_neurons(net) != []
                    println("added dendrite")
                    add_dendrite!(net, rand(get_all_neurons(net)))
                end
                if !all([onn.referenced for onn in get_output_nodes(net)]) && get_all_neurons(net) != []
                    println("added ap")
                    add_axon_point!(net, rand(get_all_neurons(net)))
                end
            end




            out = [on.value for on in get_output_nodes(net)]
            a = action_space[argmax(out)]
            r, s = OpenAIGym.step!(env, a)

            if render
                OpenAIGym.render(env)
            end
            if env.done
                break
            end
        end
    end
    return metrics
end
