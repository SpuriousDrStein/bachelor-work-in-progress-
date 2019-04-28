

# FUNCTIONS
function initialize(dna_stack, init_positions, params)
    nn = initialize_network(
                params["NETWORK_SIZE"],
                params["GLOBAL_STDV"],
                # params["NEURON_LIFETIME"],
                # params["SYNAPTIC_LIFETIME"],
                # params["DENDRITE_LIFETIME"],
                # params["AXONPOINT_LIFETIME"],
                params["MIN_FUSE_DISTANCE"],
                params["AP_SINK_ATTRACTIVE_FORCE"],
                params["AP_SURGE_REPULSIVE_FORCE"],
                params["DEN_SURGE_REPULSIVE_FORCE"],
                params["INPUT_ATTRACTIVE_FORCE"],
                params["OUTPUT_ATTRACTIVE_FORCE"],
                # params["NEURON_REPEL_FORCE"],
                params["MAX_NT_STRENGTH"],
                params["MAX_NEURON_THRESHOLD"],
                params["MAX_SYNAPTIC_THRESHOLD"],
                # params["LIFE_DECAY"],
                params["NT_RETAIN_PERCENTAGE"],
                # params["NEURON_INIT_INTERVAL"],
                # params["AP_DEN_INIT_INTERVAL"],
                params["MAX_NUM_PRIORS"],
                params["MAX_NUM_POSTERIORS"],
                params["NEURON_DESTRUCTION_THRESHOLD"],
                params["SYNAPS_DESTRUCTION_THRESHOLD"],
                dna_stack)

    # if length(params["LAYERS"]) % 2 != 0
    #     throw("Layers have to be even in number")
    # end

    num_pr = params["INIT_PRIORS"]
    num_po = params["INIT_POSTERIORS"]
    ns = params["NETWORK_SIZE"]
    n_per_l = params["LAYERS"][2:end-1]
    is = params["LAYERS"][1]
    os = params["LAYERS"][end]
    lays = length(n_per_l) # basically #neuron_layers + #connection_layers

    I = 1
    for i in I:is
        append!(nn.IO_components, [AllCell(InputNode(lays*2+1, init_positions[i], 0., false))])
    end
    I += is
    for i in I:I+os-1
        append!(nn.IO_components, [AllCell(OutputNode(1, init_positions[is+i], 0., false))])
    end
    I += os

    # println(length(init_positions))

    # num_neurons = 7
    # num_connections = 6
    # num_layers = 7+6+2
    # [inp, n, c, n, c, n, c, n, out]

    for l in 2:2:(lays*2)
        num_n = n_per_l[Integer(ceil(l/2))]

        for i in I:1+num_pr+num_po:I+num_n+(num_n * num_pr)+(num_n * num_po)-1
            n = add_neuron!(nn, l, init_positions[i])

            for j in 1:num_pr
                add_dendrite!(nn, n, l+1, init_positions[i+j])
            end
            for k in 1:num_po
                add_axon_point!(nn, n, l-1, init_positions[i+num_pr+k])
            end
        end
        I += num_n + (num_n*num_pr) + (num_n*num_po)
    end

    # # populate network with segmentation [input positions, output positions, neuron positions, dendrite positions, axon point positions]
    # for dpi in is+os+i_nn:i_nn:is+os+i_nn+(i_nden*i_nn)
    #     for (i, n) in enumerate(get_all_neurons(nn))
    #         add_dendrite!(nn, n, init_positions[dpi+i])
    #     end
    # end
    # for appi in is+os+i_nn+(i_nden*i_nn):i_nn:is+os+i_nn+(i_nden*i_nn)+(i_nap*i_nn)-1
    #     for (i, n) in enumerate(get_all_neurons(nn))
    #         add_axon_point!(nn, n, init_positions[appi+i])
    #     end
    # end
    # nn.IO_components = [input_nodes..., out_nodes...]

    return nn
end




function get_positions(x, p)
    inn = p["LAYERS"][1] + p["LAYERS"][end] + sum(p["LAYERS"][2:end-1]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_PRIORS"]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_POSTERIORS"])
    # inn = sum(params["LAYERS"][2:end-1]) + (sum(params["LAYERS"][2:end-1]) * params["INIT_PRIORS"]) + (sum(params["LAYERS"][2:end-1]) * params["INIT_POSTERIORS"])
    ps = []
    for ip in 1:2:inn*2
        append!(ps, [Position(x[ip:ip+1]...)])
    end
    return ps
end
function sample_init_positions_from_set(x, p)
    inn = p["LAYERS"][1] + p["LAYERS"][end] + sum(p["LAYERS"][2:end-1]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_PRIORS"]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_POSTERIORS"])
    ns = p["NETWORK_SIZE"]
    ps = []
    xx = [clamp(rand(Distributions.Normal(ax, p["INIT_POSITION_STDV"])), -ns, ns) for ax in x]
    for ip in 1:2:inn*2
        append!(ps, [Position(xx[ip:ip+1]...)])
    end
    return ps, xx
end
function sample_init_positions_from_sets_random(sets, p)
    inn = p["LAYERS"][1] + p["LAYERS"][end] + sum(p["LAYERS"][2:end-1]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_PRIORS"]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_POSTERIORS"])
    ns = p["NETWORK_SIZE"]
    ps = []
    xx = []
    for ip in 1:2:inn*2
        pos = rand(sets)[ip:ip+1]
        append!(ps, [Position(pos...)])
        append!(xx, [pos...])
    end
    return ps, xx
end
function get_random_init_positions(p)
    inn = p["LAYERS"][1] + p["LAYERS"][end] + sum(p["LAYERS"][2:end-1]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_PRIORS"]) + (sum(p["LAYERS"][2:end-1]) * p["INIT_POSTERIORS"])
    ns = p["NETWORK_SIZE"]
    ps = []
    x = [rand(Uniform(-ns, ns)) for p in 1:inn*2]
    for ip in 1:2:inn*2
        append!(ps, [Position(x[ip:ip+1]...)])
    end
    return ps, x
end


function get_dna(x, params)
    d = params["DNA_SAMPLE_SIZE"] # because shorter indexing

    ntd = [NeuroTransmitterDNA(x[1:d][j]) for j in 1:d]
    sd = [SynapsDNA(x[d+1:d*2][j], x[d*2+1:d*3][j], x[d*3+1:d*4][j]) for j in 1:d]
    nd = [NeuronDNA(x[d*4+1:d*5][j]) for j in 1:d]

    dna_stack = DNAStack(ntd, sd, nd)

    return dna_stack
end



function sample_from_set_plain(x, p)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Distributions.Normal(x[i]), stdv) for i in 1:d]
    syn_thr   = [rand(Distributions.Normal(x[i]), stdv) for i in d+1:(d*2)]
    syn_r_rec = [rand(Distributions.Normal(x[i]), stdv) for i in (d*2)+1:(d*3)]
    syn_maxR  = [rand(Distributions.Normal(x[i]), stdv) for i in (d*3)+1:(d*4)]
    n_thr  = [rand(Distributions.Normal(x[i]), stdv) for i in (d*4)+1:(d*5)]

    xx = [nt_init_strs..., syn_thr..., syn_r_rec..., syn_maxR..., n_thr...]

    dna_stack = DNAStack([], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_thr[i])])
    end

    return dna_stack, xx
end

function get_random_set(p)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing

    nt_init_strs = [rand(Uniform(0.1, p["MAX_NT_STRENGTH"])) for i in 1:d]
    syn_thr   = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in d+1:(d*2)]
    syn_r_rec = [rand(Uniform(1.1, 100.)) for i in (d*2)+1:(d*3)]
    syn_maxR  = [rand(Uniform(1., p["MAX_RESISTANCE"])) for i in (d*3)+1:(d*4)]
    n_thr  = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in (d*4)+1:(d*5)]

    xx = [nt_init_strs..., syn_thr..., syn_r_rec..., syn_maxR..., n_thr...]

    dna_stack = DNAStack([], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_thr[i])])
    end

    return dna_stack, xx
end

function sample_from_set_scaled(x, p; scl=1.)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Distributions.Normal(x[i], stdv * scl * 0.4)) for i in 1:d]
    syn_thr   = [rand(Distributions.Normal(x[i], stdv * scl * 0.4)) for i in d+1:(d*2)]
    syn_r_rec = [rand(Distributions.Normal(x[i], stdv * scl * 0.4)) for i in (d*2)+1:(d*3)]
    syn_maxR  = [rand(Distributions.Normal(x[i], stdv * scl * 2)) for i in (d*3)+1:(d*4)]
    n_thr  = [rand(Distributions.Normal(x[i], stdv * scl * 0.4)) for i in (d*4)+1:(d*5)]

    xx = [nt_init_strs..., syn_thr..., syn_r_rec..., syn_maxR..., n_thr...]

    dna_stack = DNAStack([], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_thr[i])])
    end

    return dna_stack, xx
end

function sample_from_sets_random(sets, p)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(sets)[i] for i in 1:d]
    syn_thr   = [rand(sets)[i] for i in d+1:(d*2)]
    syn_r_rec = [rand(sets)[i] for i in (d*2)+1:(d*3)]
    syn_maxR  = [rand(sets)[i] for i in (d*3)+1:(d*4)]
    n_thr  = [rand(sets)[i] for i in (d*4)+1:(d*5)]

    xx = [nt_init_strs..., syn_thr..., syn_r_rec..., syn_maxR..., n_thr...]

    dna_stack = DNAStack([], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_thr[i])])
    end

    return dna_stack, xx
end

function sample_from_set_decay(x, p, j)
end

function sample_from_set_scaled_decay(x, p, j; scl=0.3)
end



function unsupervised_train(net_episodes::Integer, env_episodes::Integer, iterations::Integer, parallel_networks::Integer, env, env_version, params::Dict)
    env = GymEnv(env, env_version)
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
            if rand() > 1/log(max(ℯ, mean([bn[1] for bn in best_nets_buf])))
                if rand() > 1/log(max(ℯ, mean([bn[1] for bn in best_nets_buf])))
                    println("$(n) sample dna combinations")
                    dna_stack, x = sample_from_sets_random([bnb[2] for bnb in best_nets_buf], params)
                else
                    println("$(n) sample dna normal")
                    net_distro = softmax([bn[1]/mean([i[1] for i in best_nets_buf]) for bn in best_nets_buf])
                    net_params = sample(Random.GLOBAL_RNG, best_nets_buf, Weights(net_distro))[2]
                    dna_stack, x = sample_from_set_scaled(net_params, params)
                end
            else
                println("$(n) sample dna random")
                dna_stack, x = get_random_set(params)
            end

            if rand() > 1/log(max(ℯ, mean([bn[1] for bn in best_nets_buf])))
                if rand() > 1/log(max(ℯ, mean([bn[1] for bn in best_nets_buf])))
                    println("$(n) sample position combinations")
                    init_positions, p = sample_init_positions_from_sets_random([bn[2] for bn in best_init_pos], params)
                else
                    println("$(n) sample position normal")
                    pos_distro = softmax([bp[1]/mean([i[1] for i in best_init_pos]) for bp in best_init_pos])
                    pos_params = sample(Random.GLOBAL_RNG, best_init_pos, Weights(pos_distro))[2]
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
                s = reset!(env)

                reset_network_components!(net)

                for i in 1:iterations
                    # for Acrobot
                    s = Array(s)
                    state = [s[1]>0, s[1]<0, s[2]>0, s[2]<0, s[3]>0, s[3]<0]# [(s[2] > 0), (s[2] < 0), (s[4] > 0), (s[4] < 0)]

                    for _ in 1:length(params["LAYERS"][2:end-1])
                        den_sinks, den_surges, ap_sinks, ap_surges = value_step!(net, state)
                        state_step!(net, den_sinks, den_surges, ap_sinks, ap_surges)
                        clean_network_components!(net, ((length(params["LAYERS"])-2)*2+1))
                        # runtime_instantiate_components!(net, I)
                    end

                    I += 1


                    if I % 50 == 0
                        if !all([inn.referenced for inn in get_input_nodes(net)]) && get_all_neurons(net) != []
                            net.total_fitness -= 1000
                            # add_dendrite!(net, rand(get_all_neurons(net)))
                        end
                        if !all([onn.referenced for onn in get_output_nodes(net)]) && get_all_neurons(net) != []
                            net.total_fitness -= 1000
                            # add_axon_point!(net, rand(get_all_neurons(net)))
                        end
                    end

                    # println([nnn.Q for nnn in get_all_neurons(net)])

                    out = get_output_nodes(net)
                    out = [out[on].value + out[on+1].value for on in 1:2:length(out)]
                    a = action_space[argmax(out)]
                    r, s = step!(env, a)

                    if to_degree(s[3]) >= -12 && to_degree(s[3]) <= 12
                        net.total_fitness += 50
                        sum_env_rewards += 50
                    end
                    if s[1] >= 1 || s[1] <= -1
                        net.total_fitness -= 50
                        sum_env_rewards -= 50
                    end

                    # if env.done
                    #     break
                    # end
                end
            end
            close(env)
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
    env = GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    dna_stack = get_dna(sample, params)
    init_positions = get_positions(init_positions, params)

    net = initialize(dna_stack, init_positions, params)

    metrics = Dict()

    for e in 1:episodes
        s = reset!(env)
        reset_network_components!(net)

        for i in 1:iterations
            # for Acrobot
            state = Array(s)
            state = [s[1]>0,s[1]<0, s[2]>0,s[2]<0, s[3]>0,s[3]<0]# [(s[2] > 0), (s[2] < 0), (s[4] > 0), (s[4] < 0)]


            for _ in 1:length(params["LAYERS"][2:end-1])
                den_sinks, den_surges, ap_sinks, ap_surges = value_step!(net, state)
                state_step!(net, den_sinks, den_surges, ap_sinks, ap_surges)
                clean_network_components!(net, ((length(params["LAYERS"])-2)*2+1))
                # runtime_instantiate_components!(net, I)
            end

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
            # println([s.NT.strength for s in get_synapses(get_all_all_cells(net))])


            out = get_output_nodes(net)
            out = [out[on].value + out[on+1].value for on in 1:2:length(out)]
            a = action_space[argmax(out)]
            r, s = step!(env, a)

            println("out = $out")

            if render
                OpenAIGym.render(env)
            end
            # if env.done
            #     break
            # end
        end
    end

    close(env)
    return metrics
end
