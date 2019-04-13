include(".\\..\\..\\global_utility_functions\\activation_functions.jl")

import OpenAIGym
import Random
import Distributions
import StatsBase

# FUNCTIONS
function initialize(net_dna, dna_stack, params)
    rectifyDNA!(net_dna)
    nn = unfold(net_dna,
                params["NETWORK_SIZE"],
                params["GLOBAL_STDV"],
                params["MAX_NEURON_LIFETIME"],
                params["MAX_SYNAPTIC_LIFETIME"],
                params["MAX_DENDRITE_LIFETIME"],
                params["MAX_AXONPOINT_LIFETIME"],
                params["MIN_FUSE_DISTANCE"],
                params["LIFE_DECAY"],
                params["MAX_NT_STRENGTH"],
                params["MAX_THRESHOLD"],
                params["RANDOM_FLUCTUATION"],
                params["NEURON_INIT_INTERVAL"],
                params["MIN_AP_DEN_INIT_INTERVAL"],
                params["NEURON_DESTRUCTION_THRESHOLD"],
                params["SYNAPS_DESTRUCTION_THRESHOLD"],
                dna_stack)
    rectifyDNA!(nn.dna_stack, nn)

    ns = nn.size / 2.
    input_nodes = [AllCell(InputNode(Position(ns + rand(Uniform(-0.5, 1))*ns, ns + rand(Uniform(-0.5, 1))*ns, ns + rand(Uniform(-0.5, 1))*ns), 0.)) for i in 1:params["DATA_INPUT_SIZE"]]
    out_nodes = [AllCell(OutputNode(Position(-ns + rand(Uniform(-1, 0.5))*ns, -ns + rand(Uniform(-1, 0.5))*ns, -ns + rand(Uniform(-1, 0.5))*ns), 0.)) for i in 1:params["DATA_OUTPUT_SIZE"]]

    populate_network!(nn, params["INIT_NUM_NEURONS"], params["INIT_MAX_PRIORS"], params["INIT_MAX_POSTERIORS"])
    nn.IO_components = [input_nodes..., out_nodes...]
    return nn
end

function decode(z, nets, output_scale)
    # [nt_model, ap_model, den_model, syn_model, neuron_model, net_model]
    nt_dna = nets[1](z)
    ap_dna = nets[2](z)
    den_dna = nets[3](z)
    syn_dna = nets[4](z)
    n_dna = nets[5](z)
    net_dna = nets[6](z)

    return [nt_dna..., ap_dna..., den_dna..., syn_dna..., n_dna..., net_dna...] .* output_scale
end

function get_dna(x, params)
    d = params["DNA_SAMPLE_SIZE"] # because shorter indexing

    nt_init_strs = [x[i] for i in 1:d]
    nt_retain_ps = [x[i] for i in (d)+1:(d*2)]

    ap_max_l = [x[i] for i in (d*2)+1:(d*3)]
    ap_life = [x[i] for i in (d*3)+1:(d*4)]

    den_max_l = [x[i] for i in (d*4)+1:(d*5)]
    den_life = [x[i] for i in (d*5)+1:(d*6)]

    syn_thr = [x[i] for i in (d*6)+1:(d*7)]
    syn_Qd = [x[i] for i in (d*7)+1:(d*8)]
    syn_life = [x[i] for i in (d*8)+1:(d*9)]

    n_max_pri = [x[i] for i in (d*9)+1:(d*10)]
    n_max_pos = [x[i] for i in (d*10)+1:(d*11)]
    n_life = [x[i] for i in (d*11)+1:(d*12)]
    n_init_r = [x[i] for i in (d*12)+1:(d*13)]
    n_den_init_int = [x[i] for i in (d*13)+1:(d*14)]
    n_ap_init_int = [x[i] for i in (d*14)+1:(d*15)]

    ap_sink_f = x[d*15+1]
    nrf = x[d*15+2]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i], nt_retain_ps[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_Qd[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_max_pri[i], n_max_pos[i], n_life[i], n_init_r[i], n_den_init_int[i], n_ap_init_int[i])])
    end

    return NetworkDNA(ap_sink_f, nrf), dna_stack
end

function sample_from_set_scaled(x, p; scl=0.3)
    d = p["DNA_SAMPLE_SIZE"]
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Normal(x[i], stdv * scl)) for i in 1:d]
    nt_retain_ps = [rand(Normal(x[i], stdv * scl)) for i in (d)+1:(d*2)]

    ap_max_l = [rand(Normal(x[i], stdv * scl * sqrt(p["NETWORK_SIZE"]))) for i in (d*2)+1:(d*3)]
    ap_life = [rand(Normal(x[i], stdv * scl * sqrt(p["MAX_AXONPOINT_LIFETIME"]))) for i in (d*3)+1:(d*4)]

    den_max_l = [rand(Normal(x[i], stdv * scl * sqrt(p["NETWORK_SIZE"]))) for i in (d*4)+1:(d*5)]
    den_life = [rand(Normal(x[i], stdv * scl * sqrt(p["MAX_DENDRITE_LIFETIME"]))) for i in (d*5)+1:(d*6)]

    syn_thr = [rand(Normal(x[i], stdv * scl * sqrt(p["MAX_THRESHOLD"]))) for i in (d*6)+1:(d*7)]
    syn_Qd = [rand(Normal(x[i], stdv * scl)) for i in (d*7)+1:(d*8)]
    syn_life = [rand(Normal(x[i], stdv * scl * sqrt(p["MAX_SYNAPTIC_LIFETIME"]))) for i in (d*8)+1:(d*9)]

    n_max_pri = [rand(Normal(x[i], stdv * scl * sqrt(50))) for i in (d*9)+1:(d*10)]
    n_max_pos = [rand(Normal(x[i], stdv * scl * sqrt(50))) for i in (d*10)+1:(d*11)]
    n_life = [rand(Normal(x[i], stdv * scl * sqrt(p["MAX_NEURON_LIFETIME"]))) for i in (d*11)+1:(d*12)]
    n_init_r = [rand(Normal(x[i], stdv * scl * sqrt(p["NETWORK_SIZE"]/2))) for i in (d*12)+1:(d*13)]
    n_den_init_int = [rand(Normal(x[i] , stdv * scl * sqrt(p["MIN_AP_DEN_INIT_INTERVAL"]+50))) for i in (d*13)+1:(d*14)]
    n_ap_init_int = [rand(Normal(x[i] , stdv * scl * sqrt(p["MIN_AP_DEN_INIT_INTERVAL"]+50))) for i in (d*14)+1:(d*15)]

    ap_sink_f = rand(Normal(x[d*15+1], stdv * scl * sqrt(p["AP_SINK_ATTRACTIVE_FORCE"]+2.5)))
    nrf = rand(Normal(x[d*15+2], stdv * scl))

    x = [nt_init_strs..., nt_retain_ps..., ap_max_l...,
        ap_life..., den_max_l..., den_life..., syn_thr...,
        syn_Qd..., syn_life..., n_max_pri..., n_max_pos...,
        n_life..., n_init_r..., n_den_init_int..., n_ap_init_int...,
        ap_sink_f..., nrf...]

    # println(x)

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i], nt_retain_ps[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_Qd[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_max_pri[i], n_max_pos[i], n_life[i], n_init_r[i], n_den_init_int[i], n_ap_init_int[i])])
    end
    return NetworkDNA(ap_sink_f, nrf), dna_stack, x
end

function sample_from_set_decay(x, p, j)
    d = p["DNA_SAMPLE_SIZE"]
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in 1:d]
    nt_retain_ps = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d)+1:(d*2)]

    ap_max_l = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*2)+1:(d*3)]
    ap_life = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*3)+1:(d*4)]

    den_max_l = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*4)+1:(d*5)]
    den_life = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*5)+1:(d*6)]

    syn_thr = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*6)+1:(d*7)]
    syn_Qd = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*7)+1:(d*8)]
    syn_life = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*8)+1:(d*9)]

    n_max_pri = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*9)+1:(d*10)]
    n_max_pos = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*10)+1:(d*11)]
    n_life = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*11)+1:(d*12)]
    n_init_r = [rand(Normal(x[i], stdv * (1/(log(j)+1)))) for i in (d*12)+1:(d*13)]
    n_den_init_int = [rand(Normal(x[i] , stdv * (1/(log(j)+1)))) for i in (d*13)+1:(d*14)]
    n_ap_init_int = [rand(Normal(x[i] , stdv * (1/(log(j)+1)))) for i in (d*14)+1:(d*15)]

    ap_sink_f = rand(Normal(x[d*15+1], stdv * (1/(log(j)+1))))
    nrf = rand(Normal(x[d*15+2], stdv * (1/(log(j)+1))))

    x = [nt_init_strs..., nt_retain_ps..., ap_max_l...,
        ap_life..., den_max_l..., den_life..., syn_thr...,
        syn_Qd..., syn_life..., n_max_pri..., n_max_pos...,
        n_life..., n_init_r..., n_den_init_int..., n_ap_init_int...,
        ap_sink_f..., nrf...]

    # println(x)

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i], nt_retain_ps[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_Qd[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_max_pri[i], n_max_pos[i], n_life[i], n_init_r[i], n_den_init_int[i], n_ap_init_int[i])])
    end
    return NetworkDNA(ap_sink_f, nrf), dna_stack, x
end

function sample_from_set_scaled_decay(x, p, j; scl=0.3)
    d = p["DNA_SAMPLE_SIZE"]
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Normal(x[i],       stdv * scl * (1/(log(j)+1)))) for i in 1:d]
    nt_retain_ps = [rand(Normal(x[i],       stdv * scl * (1/(log(j)+1)))) for i in (d)+1:(d*2)]

    ap_max_l = [rand(Normal(x[i],           stdv * scl * (1/(log(j)+1)) * sqrt(p["NETWORK_SIZE"]))) for i in (d*2)+1:(d*3)]
    ap_life = [rand(Normal(x[i],            stdv * scl * (1/(log(j)+1)) * sqrt(p["MAX_AXONPOINT_LIFETIME"]))) for i in (d*3)+1:(d*4)]

    den_max_l = [rand(Normal(x[i],          stdv * scl * (1/(log(j)+1)) * sqrt(p["NETWORK_SIZE"]))) for i in (d*4)+1:(d*5)]
    den_life = [rand(Normal(x[i],           stdv * scl * (1/(log(j)+1)) * sqrt(p["MAX_DENDRITE_LIFETIME"]))) for i in (d*5)+1:(d*6)]

    syn_thr = [rand(Normal(x[i],            stdv * scl * (1/(log(j)+1)) * sqrt(p["MAX_THRESHOLD"]))) for i in (d*6)+1:(d*7)]
    syn_Qd = [rand(Normal(x[i],             stdv * scl * (1/(log(j)+1)))) for i in (d*7)+1:(d*8)]
    syn_life = [rand(Normal(x[i],           stdv * scl * (1/(log(j)+1)) * sqrt(p["MAX_SYNAPTIC_LIFETIME"]))) for i in (d*8)+1:(d*9)]

    n_max_pri = [rand(Normal(x[i],          stdv * scl * (1/(log(j)+1)) * sqrt(50))) for i in (d*9)+1:(d*10)]
    n_max_pos = [rand(Normal(x[i],          stdv * scl * (1/(log(j)+1)) * sqrt(50))) for i in (d*10)+1:(d*11)]
    n_life = [rand(Normal(x[i],             stdv * scl * (1/(log(j)+1)) * sqrt(p["MAX_NEURON_LIFETIME"]))) for i in (d*11)+1:(d*12)]
    n_init_r = [rand(Normal(x[i],           stdv * scl * (1/(log(j)+1)) * sqrt(p["NETWORK_SIZE"]/2))) for i in (d*12)+1:(d*13)]
    n_den_init_int = [rand(Normal(x[i] ,    stdv * scl * (1/(log(j)+1)) * sqrt(p["MIN_AP_DEN_INIT_INTERVAL"]+50))) for i in (d*13)+1:(d*14)]
    n_ap_init_int = [rand(Normal(x[i] ,     stdv * scl * (1/(log(j)+1)) * sqrt(p["MIN_AP_DEN_INIT_INTERVAL"]+50))) for i in (d*14)+1:(d*15)]

    ap_sink_f = rand(Normal(x[d*15+1], stdv * scl * sqrt(p["AP_SINK_ATTRACTIVE_FORCE"]+2.5) * (1/(log(j)+1))))
    nrf = rand(Normal(x[d*15+2], stdv * scl * (1/(log(j)+1))))

    x = [nt_init_strs..., nt_retain_ps..., ap_max_l...,
        ap_life..., den_max_l..., den_life..., syn_thr...,
        syn_Qd..., syn_life..., n_max_pri..., n_max_pos...,
        n_life..., n_init_r..., n_den_init_int..., n_ap_init_int...,
        ap_sink_f..., nrf...]

    # println(x)

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i], nt_retain_ps[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_Qd[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_max_pri[i], n_max_pos[i], n_life[i], n_init_r[i], n_den_init_int[i], n_ap_init_int[i])])
    end
    return NetworkDNA(ap_sink_f, nrf), dna_stack, x
end


function sample_from_set(x, p)
    d = p["DNA_SAMPLE_SIZE"]
    stdv = p["GLOBAL_STDV"]

    nt_init_strs = [rand(Normal(x[i], stdv)) for i in 1:d]
    nt_retain_ps = [rand(Normal(x[i], stdv)) for i in (d)+1:(d*2)]

    ap_max_l = [rand(Normal(x[i], stdv)) for i in (d*2)+1:(d*3)]
    ap_life = [rand(Normal(x[i], stdv)) for i in (d*3)+1:(d*4)]

    den_max_l = [rand(Normal(x[i], stdv)) for i in (d*4)+1:(d*5)]
    den_life = [rand(Normal(x[i], stdv)) for i in (d*5)+1:(d*6)]

    syn_thr = [rand(Normal(x[i], stdv)) for i in (d*6)+1:(d*7)]
    syn_Qd = [rand(Normal(x[i], stdv)) for i in (d*7)+1:(d*8)]
    syn_life = [rand(Normal(x[i], stdv)) for i in (d*8)+1:(d*9)]

    n_max_pri = [rand(Normal(x[i], stdv)) for i in (d*9)+1:(d*10)]
    n_max_pos = [rand(Normal(x[i], stdv)) for i in (d*10)+1:(d*11)]
    n_life = [rand(Normal(x[i], stdv)) for i in (d*11)+1:(d*12)]
    n_init_r = [rand(Normal(x[i], stdv)) for i in (d*12)+1:(d*13)]
    n_den_init_int = [rand(Normal(x[i] , stdv)) for i in (d*13)+1:(d*14)]
    n_ap_init_int = [rand(Normal(x[i] , stdv)) for i in (d*14)+1:(d*15)]

    ap_sink_f = rand(Normal(x[d*15+1], stdv))
    nrf = rand(Normal(x[d*15+2], stdv))

    x = [nt_init_strs..., nt_retain_ps..., ap_max_l...,
        ap_life..., den_max_l..., den_life..., syn_thr...,
        syn_Qd..., syn_life..., n_max_pri..., n_max_pos...,
        n_life..., n_init_r..., n_den_init_int..., n_ap_init_int...,
        ap_sink_f..., nrf...]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i], nt_retain_ps[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_Qd[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_max_pri[i], n_max_pos[i], n_life[i], n_init_r[i], n_den_init_int[i], n_ap_init_int[i])])
    end
    return NetworkDNA(ap_sink_f, nrf), dna_stack, x
end

function get_random_set(p)
    d = p["DNA_SAMPLE_SIZE"]

    nt_init_strs = [rand(Uniform(0.5, 1.5)) for i in 1:d]
    nt_retain_ps = [rand(Uniform(0.1, 0.9)) for i in (d)+1:(d*2)]

    ap_max_l = [rand(Uniform(1., p["NETWORK_SIZE"])) for i in (d*2)+1:(d*3)]
    ap_life = [rand(Uniform(1., p["MAX_AXONPOINT_LIFETIME"])) for i in (d*3)+1:(d*4)]

    den_max_l = [rand(Uniform(1., p["NETWORK_SIZE"])) for i in (d*4)+1:(d*5)]
    den_life = [rand(Uniform(1., p["MAX_DENDRITE_LIFETIME"])) for i in (d*5)+1:(d*6)]

    syn_thr = [rand(Uniform(0.5, p["MAX_THRESHOLD"])) for i in (d*6)+1:(d*7)]
    syn_Qd = [rand(Uniform(0.5, 0.99)) for i in (d*7)+1:(d*8)]
    syn_life = [rand(Uniform(1., p["MAX_SYNAPTIC_LIFETIME"])) for i in (d*8)+1:(d*9)]

    n_max_pri = [rand(Uniform(1, 50)) for i in (d*9)+1:(d*10)]
    n_max_pos = [rand(Uniform(1, 50)) for i in (d*10)+1:(d*11)]
    n_life = [rand(Uniform(1., p["MAX_NEURON_LIFETIME"])) for i in (d*11)+1:(d*12)]
    n_init_r = [rand(Uniform(0.5, p["NETWORK_SIZE"]/2)) for i in (d*12)+1:(d*13)]
    n_den_init_int = [rand(Uniform(p["MIN_AP_DEN_INIT_INTERVAL"], p["MIN_AP_DEN_INIT_INTERVAL"]+100)) for i in (d*13)+1:(d*14)]
    n_ap_init_int = [rand(Uniform(p["MIN_AP_DEN_INIT_INTERVAL"], p["MIN_AP_DEN_INIT_INTERVAL"]+100)) for i in (d*14)+1:(d*15)]

    ap_sink_f = rand(Uniform(p["AP_SINK_ATTRACTIVE_FORCE"], p["AP_SINK_ATTRACTIVE_FORCE"]+5.))
    nrf = rand(Uniform(0., 1.))

    x = [nt_init_strs..., nt_retain_ps..., ap_max_l...,
        ap_life..., den_max_l..., den_life..., syn_thr...,
        syn_Qd..., syn_life..., n_max_pri..., n_max_pos...,
        n_life..., n_init_r..., n_den_init_int..., n_ap_init_int...,
        ap_sink_f..., nrf...]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i], nt_retain_ps[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_Qd[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_max_pri[i], n_max_pos[i], n_life[i], n_init_r[i], n_den_init_int[i], n_ap_init_int[i])])
    end
    return NetworkDNA(ap_sink_f, nrf), dna_stack, x
end

# function supervised_train(episodes, iterations, data_input, data_output)
#
#     for e in 1:episodes
#         nets = []
#
#         for n in 1:PARALLEL_NETWORKS
#             rand_z = rand(net_latent_size)
#
#             rec_x = decode(rand_z, OUTPUT_SCALE)
#             net_dna, dna_stack = get_dna(rec_x)
#
#
#             net = initialize(net_dna, dna_stack)
#
#             for i in 1:iterations
#                 rand_n = rand(1:length(data_output))
#                 sample_x = data_input[rand_n]
#                 sample_y = data_output[rand_n]
#
#                 den_sinks, ap_sinks = value_step!(net, sample_x)
#                 state_step!(net, den_sinks, ap_sinks)
#                 clean_network_components!(net)
#                 runtime_instantiate_components!(net, i)
#
#                 # loss(output_nodes, y) -> network performance
#                 y_hat = get_output_nodes(get_all_all_cells(net))
#                 loss = sum((y_hat .- sample_y).^2)
#
#                 net.total_fitness -= loss
#             end
#
#             append!(nets, (rand_z => net.total_fitness))
#         end
#
#         # loss()
#
#     end
# end


function unsupervised_train(net_episodes::Integer, env_episodes::Integer, iterations::Integer, parallel_networks::Integer, env, env_version, params::Dict)

    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    best_nets = [[-99999999, get_random_set(params)[3]] for _ in 1:params["TOP_BUFFER_LENGTH"]]
    metrics = Dict([("current_net_$(n)_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_final_neurons" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_final_dens" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_final_aps" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_final_syns" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_neuron_fitness" => []) for n in 1:parallel_networks]...,
                    [("net_$(n)_synaps_fitness" => []) for n in 1:parallel_networks]...)

    for e in 1:net_episodes
        nets = []
        xs = []
        println("episode: $e")

        for n in 1:parallel_networks
            net_distro = softmax([bn[1]/mean([i[1] for i in best_nets]) for bn in best_nets])
            net_params = Distributions.sample(Random.GLOBAL_RNG, best_nets, StatsBase.Weights(net_distro))[2]
            net_dna, dna_stack, x = sample_from_set_scaled_decay(net_params, params, e)
            append!(xs, [Flux.Tracker.data.(x)])
            net = initialize(net_dna, dna_stack, params)

            I = 1 # for counting iterations

            # training
            for ee in 1:env_episodes
                s = OpenAIGym.reset!(env)

                # reset_network_components!(net)

                for i in 1:iterations
                    den_sinks, ap_sinks = value_step!(net, (Array(s) .+ 1) .* 10) # + .1 to produce preferable non negative values
                    state_step!(net, den_sinks, ap_sinks)
                    clean_network_components!(net)
                    runtime_instantiate_components!(net, I)
                    I += 1

                    out = [on.value for on in get_output_nodes(net)]
                    a = action_space[argmax(out)]

                    r, s = OpenAIGym.step!(env, a)

                    net.total_fitness += r * i

                    if env.done
                        break
                    end
                end

                # if get_input_nodes(get_all_all_cells(net)) == []
                #     println("input assigned")
                # else
                #     println("---- input assigned")
                # end
                #
                # if get_output_nodes(get_all_all_cells(net)) == []
                #     println("no outputs")
                # else
                #     println("---- output assigned")
                # end
                if get_output_nodes(get_all_all_cells(net)) != []
                    println([" $(v.cell.value)," for v in get_output_nodes_in_all(net)]...)
                end
            end
            # println("net $n:")
            # println("#neurons     -- $(length(get_all_neurons(net)))")
            # println("#dendrites   -- $(length(get_dendrites(get_all_all_cells(net))))")
            # println("#axon points -- $(length(get_axon_points(get_all_all_cells(net))))")
            # println("#synapses    -- $(length(get_synapses(get_all_all_cells(net))))")


            append!(nets, [(net.total_fitness => copy(Flux.Tracker.data.(x)))])

            append!(metrics["net_$(n)_final_neurons"], [length(get_all_neurons(net))])
            append!(metrics["net_$(n)_final_dens"], [length(get_dendrites(get_all_all_cells(net)))])
            append!(metrics["net_$(n)_final_aps"], [length(get_axon_points(get_all_all_cells(net)))])
            append!(metrics["net_$(n)_final_syns"], [length(get_synapses(get_all_all_cells(net)))])
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
            append!(metrics["current_net_$(pln)_fitness"], [nets[pln][1]])
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
