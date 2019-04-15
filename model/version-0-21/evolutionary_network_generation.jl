include(".\\..\\..\\global_utility_functions\\activation_functions.jl")

import OpenAIGym
import Random
import Distributions
import StatsBase

# FUNCTIONS
function initialize(dna_stack, params)
    nn = initialize_network(
                params["NETWORK_SIZE"], # still missing
                params["GLOBAL_STDV"],
                params["MAX_NEURON_LIFETIME"],
                params["MAX_SYNAPTIC_LIFETIME"],
                params["MAX_DENDRITE_LIFETIME"],
                params["MAX_AXONPOINT_LIFETIME"],
                params["MIN_FUSE_DISTANCE"],
                params["AP_SINK_ATTRACTIVE_FORCE"],
                params["AP_SURGE_REPULSIVE_FORCE"],
                params["DEN_SURGE_REPULSIVE_FORCE"],
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

    ns = nn.size / 2.
    input_nodes = [AllCell(InputNode(Position(ns + rand(Uniform(-0.5, 1))*ns, ns + rand(Uniform(-0.5, 1))*ns, ns + rand(Uniform(-0.5, 1))*ns), 0.)) for i in 1:params["DATA_INPUT_SIZE"]]
    out_nodes = [AllCell(OutputNode(Position(-ns + rand(Uniform(-1, 0.5))*ns, -ns + rand(Uniform(-1, 0.5))*ns, -ns + rand(Uniform(-1, 0.5))*ns), 0.)) for i in 1:params["DATA_OUTPUT_SIZE"]]

    populate_network!(nn, params["INIT_NUM_NEURONS"], params["MAX_PRIORS"], params["MAX_POSTERIORS"])
    nn.IO_components = [input_nodes..., out_nodes...]
    return nn
end



function get_dna(x, params)
    d = params["DNA_SAMPLE_SIZE"] # because shorter indexing

    nt_init_strs = [x[i] for i in 1:d]

    ap_max_l  = [x[i] for i in (d)+1:(d*2)]
    ap_life   = [x[i] for i in (d*2)+1:(d*3)]

    den_max_l = [x[i] for i in (d*3)+1:(d*4)]
    den_life  = [x[i] for i in (d*4)+1:(d*5)]

    syn_thr   = [x[i] for i in (d*5)+1:(d*6)]
    syn_r_rec = [x[i] for i in (d*6)+1:(d*7)]
    syn_maxR  = [x[i] for i in (d*7)+1:(d*8)]
    syn_life  = [x[i] for i in (d*8)+1:(d*9)]

    n_life = [x[i] for i in (d*9)+1:(d*10)]
    n_thr  = [x[i] for i in (d*10)+1:(d*11)]
    n_max_pri = [x[i] for i in (d*11)+1:(d*12)]
    n_max_pos = [x[i] for i in (d*12)+1:(d*13)]
    n_den_init_int = [x[i] for i in (d*13)+1:(d*14)]
    n_ap_init_int = [x[i] for i in (d*14)+1:(d*15)]
    n_ap_den_init_r = [x[i] for i in (d*15)+1:(d*16)]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_life[i], n_thr[i], n_max_pri[i], n_max_pos[i], n_den_init_int[i], n_ap_init_int[i], n_ap_den_init_r[i])])
    end

    return dna_stack
end

function sample_from_set_plain(x, p)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Normal(x[i], stdv)) for i in 1:d]

    ap_max_l  = [rand(Normal(x[i], stdv)) for i in (d)+1:(d*2)]
    ap_life   = [rand(Normal(x[i], stdv)) for i in (d*2)+1:(d*3)]

    den_max_l = [rand(Normal(x[i], stdv)) for i in (d*3)+1:(d*4)]
    den_life  = [rand(Normal(x[i], stdv)) for i in (d*4)+1:(d*5)]

    syn_thr   = [rand(Normal(x[i], stdv)) for i in (d*5)+1:(d*6)]
    syn_r_rec = [rand(Normal(x[i], stdv)) for i in (d*6)+1:(d*7)]
    syn_maxR  = [rand(Normal(x[i], stdv)) for i in (d*7)+1:(d*8)]
    syn_life  = [rand(Normal(x[i], stdv)) for i in (d*8)+1:(d*9)]

    n_life = [rand(Normal(x[i], stdv)) for i in (d*9)+1:(d*10)]
    n_thr  = [rand(Normal(x[i], stdv)) for i in (d*10)+1:(d*11)]
    n_max_pri = [round(rand(Normal(x[i], stdv))) for i in (d*11)+1:(d*12)]
    n_max_pos = [round(rand(Normal(x[i], stdv))) for i in (d*12)+1:(d*13)]
    n_den_init_int = [round(rand(Normal(x[i], stdv))) for i in (d*13)+1:(d*14)]
    n_ap_init_int = [round(rand(Normal(x[i], stdv))) for i in (d*14)+1:(d*15)]
    n_ap_den_init_r = [round(rand(Normal(x[i], stdv))) for i in (d*15)+1:(d*16)]

    xx = [nt_init_strs..., ap_max_l..., ap_life..., den_max_l..., den_life...,
        syn_thr..., syn_r_rec..., syn_maxR..., syn_life...,
        n_life..., n_thr..., n_max_pri..., n_max_pos..., n_den_init_int..., n_ap_init_int..., n_ap_den_init_r...]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_life[i], n_thr[i], n_max_pri[i], n_max_pos[i], n_den_init_int[i], n_ap_init_int[i], n_ap_den_init_r[i])])
    end

    return dna_stack, xx
end

function get_random_set(p)
    d = p["DNA_SAMPLE_SIZE"] # because shorter indexing

    nt_init_strs = [rand(Uniform(0.1, p["MAX_NT_STRENGTH"])) for i in 1:d]

    ap_max_l  = [rand(Uniform(1., p["NETWORK_SIZE"])) for i in (d)+1:(d*2)]
    ap_life   = [rand(Uniform(1., p["MAX_AXONPOINT_LIFETIME"])) for i in (d*2)+1:(d*3)]

    den_max_l = [rand(Uniform(1., p["NETWORK_SIZE"])) for i in (d*3)+1:(d*4)]
    den_life  = [rand(Uniform(1., p["MAX_DENDRITE_LIFETIME"])) for i in (d*4)+1:(d*5)]

    syn_thr   = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in (d*5)+1:(d*6)]
    syn_r_rec = [rand(Uniform(1.1, 100.)) for i in (d*6)+1:(d*7)]
    syn_maxR  = [rand(Uniform(1., p["MAX_MAX_RESISTANCE"])) for i in (d*7)+1:(d*8)]
    syn_life  = [rand(Uniform(1., p["MAX_SYNAPTIC_LIFETIME"])) for i in (d*8)+1:(d*9)]

    n_life = [rand(Uniform(1., p["MAX_NEURON_LIFETIME"])) for i in (d*9)+1:(d*10)]
    n_thr  = [rand(Uniform(0.5, p["MAX_SYNAPTIC_THRESHOLD"])) for i in (d*10)+1:(d*11)]
    n_max_pri = [round(rand(Uniform(1, p["MAX_PRIORS"]))) for i in (d*11)+1:(d*12)]
    n_max_pos = [round(rand(Uniform(1, p["MAX_POSTERIORS"]))) for i in (d*12)+1:(d*13)]
    n_den_init_int = [round(rand(Uniform(p["MIN_AP_DEN_INIT_INTERVAL"], p["NEURON_INIT_INTERVAL"]))) for i in (d*13)+1:(d*14)]
    n_ap_init_int = [round(rand(Uniform(p["MIN_AP_DEN_INIT_INTERVAL"], p["NEURON_INIT_INTERVAL"]))) for i in (d*14)+1:(d*15)]
    n_ap_den_init_r = [rand(Uniform(1., p["NETWORK_SIZE"])) for i in (d*15)+1:(d*16)]

    xx = [nt_init_strs..., ap_max_l..., ap_life..., den_max_l..., den_life...,
        syn_thr..., syn_r_rec..., syn_maxR..., syn_life...,
        n_life..., n_thr..., n_max_pri..., n_max_pos..., n_den_init_int..., n_ap_init_int..., n_ap_den_init_r...]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_r_rec[i], syn_maxR[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_life[i], n_thr[i], n_max_pri[i], n_max_pos[i], n_den_init_int[i], n_ap_init_int[i], n_ap_den_init_r[i])])
    end

    return dna_stack, xx
end

function sample_from_set_scaled(x, p; scl=1.)
end

function sample_from_set_decay(x, p, j)
end

function sample_from_set_scaled_decay(x, p, j; scl=0.3)
end


function unsupervised_train(net_episodes::Integer, env_episodes::Integer, iterations::Integer, parallel_networks::Integer, env, env_version, params::Dict)
    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    best_nets = [[-99999999, get_random_set(params)[2]] for _ in 1:params["TOP_BUFFER_LENGTH"]]

    metrics = Dict([("net_$(n)_current_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_num_neurons" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_num_dens" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_num_aps" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_num_syns" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_neuron_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_synaps_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_output_signals" => []) for n in 1:parallel_networks]...)


    for e in 1:net_episodes
        nets = []
        xs = []
        println("episode: $e")

        for n in 1:parallel_networks
            if rand() > 0.5
                net_distro = softmax([bn[1]/mean([i[1] for i in best_nets]) for bn in best_nets])
                net_params = Distributions.sample(Random.GLOBAL_RNG, best_nets, StatsBase.Weights(net_distro))[2]
                dna_stack, x = sample_from_set_plain(net_params, params)
            else
                dna_stack, x = get_random_set(params)
            end

            append!(xs, [x])
            net = initialize(dna_stack, params)

            I = 1 # for counting iterations

            total_output = [0. for _ in 1:params["DATA_OUTPUT_SIZE"]]

            # training
            for ee in 1:env_episodes
                s = OpenAIGym.reset!(env)

                # reset_network_components!(net)

                for i in 1:iterations
                    # for Acrobot
                    state = [Array(s)[2] > 0, Array(s)[4] > 0]

                    den_sinks, den_surges, ap_sinks, ap_surges = value_step!(net, state)
                    state_step!(net, den_sinks, den_surges, ap_sinks, ap_surges)
                    clean_network_components!(net)
                    runtime_instantiate_components!(net, I)
                    I += 1

                    out = [on.value for on in get_output_nodes(net)]
                    a = action_space[argmax(out)]

                    r, s = OpenAIGym.step!(env, a)

                    net.total_fitness += r * i

                    total_output .+= [o_n.value for o_n in get_output_nodes(net)]

                    if env.done
                        break
                    end
                end
            end
            # println("net $n:")
            # println("#neurons     -- $(length(get_all_neurons(net)))")
            # println("#dendrites   -- $(length(get_dendrites(get_all_all_cells(net))))")
            # println("#axon points -- $(length(get_axon_points(get_all_all_cells(net))))")
            # println("#synapses    -- $(length(get_synapses(get_all_all_cells(net))))")


            tally_up_fitness!(net)
            append!(nets, [(net.total_fitness => copy(x))])

            append!(metrics["net_$(n)_output_signals"], [total_output])
            append!(metrics["net_$(n)_num_neurons"], [net.n_counter])
            append!(metrics["net_$(n)_num_dens"], [net.den_counter])
            append!(metrics["net_$(n)_num_aps"], [net.ap_counter])
            append!(metrics["net_$(n)_num_syns"], [net.syn_counter])
            append!(metrics["net_$(n)_neuron_fitness"], [sum([n.total_fitness for n in get_all_neurons(net)])])
            if get_synapses(get_all_all_cells(net)) != []
                append!(metrics["net_$(n)_synaps_fitness"], [sum([s.total_fitness for s in get_synapses(get_all_all_cells(net))])])
            else
                append!(metrics["net_$(n)_synaps_fitness"], [0])
            end
        end

        for cn in nets
            if cn[1] > sort(best_nets)[1][1]
                sort(best_nets)[1] .= copy.(cn)
            end
        end

        println("best_net_rewards: $([bn[1] for bn in best_nets])")

        for pln in 1:parallel_networks
            append!(metrics["net_$(pln)_current_fitness"], [nets[pln][1]])
        end

    end
    return best_nets, metrics
end


function unsupervised_testing(env_episodes::Integer, iterations::Integer, net_dna, dna_stack, env, env_version, params::Dict)
    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    net = initialize(net_dna, dna_stack, params)
    I = 1
    for e in 1:env_episodes
        s = OpenAIGym.reset!(env)

        reset_network_components!(net)

        for i in 1:iterations
            den_sinks, ap_sinks = value_step!(net, Array(s) .+ .5) # + .5 to produce preferable non negative values
            state_step!(net, den_sinks, ap_sinks)
            clean_network_components!(net)
            runtime_instantiate_components!(net, I)
            I += 1

            out = [on.value for on in get_output_nodes(net)]
            a = action_space[argmax(out)]

            r, s = OpenAIGym.step!(env, a)

            net.total_fitness += r * i

            OpenAIGym.render(env)

            if env.done
                break
            end
        end
    end
end
