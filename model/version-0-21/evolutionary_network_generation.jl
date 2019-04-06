import Flux
import OpenAIGym


function init_network_generating_network(params::Dict)
    ap_sample_output_size = params["DNA_SAMPLE_SIZE"] * 3 * 2 # dna samples for 1 netwprl * 3 features * 2 for min and max values
    den_sample_output_size = params["DNA_SAMPLE_SIZE"] * 3 * 2
    nt_sample_output_size = params["DNA_SAMPLE_SIZE"] * 3 * 2 + (2 * 3) # + (2 * 3) for 3 min-max pairs per possition in init_pos
    syn_sample_output_size = params["DNA_SAMPLE_SIZE"] * 3 * 2
    n_sample_output_size = params["DNA_SAMPLE_SIZE"] * 5 * 2 + (2 * 3) # the same as nt
    net_sample_output_size = 3 * 2 # net_size, ap_sink_force and neuron_repel_force

    net_input_size = ap_sample_output_size + den_sample_output_size + nt_sample_output_size + syn_sample_output_size + n_sample_output_size
    net_latent_size = params["LATENT_SIZE"]
    latent_activation = params["LATENT_ACTIVATION"]

    encoder_model = Flux.Chain(
        Flux.Dense(net_input_size, params["ENCODER_HIDDENS"][1]),
        [Flux.Dense(params["ENCODER_HIDDENS"][i-1], params["ENCODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["ENCODER_HIDDENS"])]...,
        Flux.Dense(params["ENCODER_HIDDENS"][end], net_latent_size, latent_activation))

    nt_model = Flux.Chain(
        Flux.Dense(net_latent_size, params["DECODER_HIDDENS"][1], Flux.relu),
        [Flux.Dense(params["DECODER_HIDDENS"][i-1], params["DECODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["DECODER_HIDDENS"])]...,
        Flux.Dense(params["DECODER_HIDDENS"][end], nt_sample_output_size, Flux.relu))

    ap_model = Flux.Chain(
        Flux.Dense(net_latent_size, params["DECODER_HIDDENS"][1], Flux.relu),
        [Flux.Dense(params["DECODER_HIDDENS"][i-1], params["DECODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["DECODER_HIDDENS"])]...,
        Flux.Dense(params["DECODER_HIDDENS"][end], ap_sample_output_size, Flux.relu))

    den_model = Flux.Chain(
        Flux.Dense(net_latent_size, params["DECODER_HIDDENS"][1], Flux.relu),
        [Flux.Dense(params["DECODER_HIDDENS"][i-1], params["DECODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["DECODER_HIDDENS"])]...,
        Flux.Dense(params["DECODER_HIDDENS"][end], den_sample_output_size, Flux.relu))

    syn_model = Flux.Chain(
        Flux.Dense(net_latent_size, params["DECODER_HIDDENS"][1], Flux.relu),
        [Flux.Dense(params["DECODER_HIDDENS"][i-1], params["DECODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["DECODER_HIDDENS"])]...,
        Flux.Dense(params["DECODER_HIDDENS"][end], syn_sample_output_size, Flux.relu))

    neuron_model = Flux.Chain(
        Flux.Dense(net_latent_size, params["DECODER_HIDDENS"][1], Flux.relu),
        [Flux.Dense(params["DECODER_HIDDENS"][i-1], params["DECODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["DECODER_HIDDENS"])]...,
        Flux.Dense(params["DECODER_HIDDENS"][end], n_sample_output_size, Flux.relu))

    net_model = Flux.Chain(
        Flux.Dense(net_latent_size, params["DECODER_HIDDENS"][1], Flux.relu),
        [Flux.Dense(params["DECODER_HIDDENS"][i-1], params["DECODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["DECODER_HIDDENS"])]...,
        Flux.Dense(params["DECODER_HIDDENS"][end], net_sample_output_size, Flux.relu))

    decoder_params = [Flux.params(net_model)..., Flux.params(neuron_model)..., Flux.params(syn_model)..., Flux.params(den_model)..., Flux.params(ap_model)..., Flux.params(nt_model)...]
    encoder_params = [Flux.params(encoder_model)]

    return encoder_model, [nt_model, ap_model, den_model, syn_model, neuron_model, net_model], [encoder_params, decoder_params]
end


# FUNCTIONS
function initialize(net_dna, dna_stack, params)
    rectifyDNA!(net_dna)
    nn = unfold(net_dna,
                params["MAX_NEURON_LIFETIME"],
                params["MAX_SYNAPTIC_LIFETIME"],
                params["MAX_DENDRITE_LIFETIME"],
                params["MAX_AXONPOINT_LIFETIME"],
                params["MIN_FUSE_DISTANCE"],
                params["LIFE_DECAY"],
                params["MAX_NT_DISPERSION_STRENGTH_SCALE"],
                params["MAX_THRESHOLD"],
                params["RANDOM_FLUCTUATION"],
                params["NEURON_INIT_INTERVAL"],
                params["MIN_AP_DEN_INIT_INTERVAL"],
                dna_stack,
                fitness_decay=params["FITNESS_DECAY"])

    rectifyDNA!(nn.dna_stack, nn)
    populate_network!(nn, params["INIT_NUM_NEURONS"], params["INIT_MAX_PRIORS"], params["INIT_MAX_POSTERIORS"])
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

function get_dna(x, dna_sample_size)
    d = dna_sample_size # because shorter indexing

    min_max_pairs = [min_max_pair(Flux.Tracker.data(x[i]), Flux.Tracker.data(x[i+1])) for i in 1:2:length(x)]

    nt_init_strs = [min_max_pairs[i] for i in 1:d]
    nt_disp_regs = [InitializationPossition(min_max_pairs[i], min_max_pairs[i+1], min_max_pairs[i+2]) for i in d+1:3:(4*d)]
    nt_disp_strs = [min_max_pairs[i] for i in (d*3)+1:(d*4)]
    nt_retain_ps = [min_max_pairs[i] for i in (d*4)+1:(d*5)]

    ap_max_l = [min_max_pairs[i] for i in (d*5)+1:(d*6)]
    ap_life = [min_max_pairs[i] for i in (d*6)+1:(d*7)]

    den_max_l = [min_max_pairs[i] for i in (d*7)+1:(d*8)]
    den_life = [min_max_pairs[i] for i in (d*8)+1:(d*9)]

    syn_thr = [min_max_pairs[i] for i in (d*9)+1:(d*10)]
    syn_Qd = [min_max_pairs[i] for i in (d*10)+1:(d*11)]
    syn_life = [min_max_pairs[i] for i in (d*11)+1:(d*12)]

    n_max_pri = [min_max_pairs[i] for i in (d*12)+1:(d*13)]
    n_max_pos = [min_max_pairs[i] for i in (d*13)+1:(d*14)]
    n_life = [min_max_pairs[i] for i in (d*14)+1:(d*15)]
    n_init_r = [min_max_pairs[i] for i in (d*15)+1:(d*16)]
    n_den_init_int = [min_max_pairs[i] for i in (d*16)+1:(d*17)]
    n_ap_init_int = [min_max_pairs[i] for i in (d*17)+1:(d*18)]

    net_size = min_max_pairs[end-2]
    ap_sink_f = min_max_pairs[end-1]
    nrf = min_max_pairs[end]

    dna_stack = DNAStack([], [], [], [], [])
    for i in 1:d
        append!(dna_stack.nt_dna_samples, [NeuroTransmitterDNA(nt_init_strs[i], nt_disp_regs[i], nt_disp_strs[i], nt_retain_ps[i])])
        append!(dna_stack.ap_dna_samples, [AxonPointDNA(ap_max_l[i], ap_life[i])])
        append!(dna_stack.den_dna_samples, [DendriteDNA(den_max_l[i], den_life[i])])
        append!(dna_stack.syn_dna_samples, [SynapsDNA(syn_thr[i], syn_Qd[i], syn_life[i])])
        append!(dna_stack.n_dna_samples, [NeuronDNA(n_max_pri[i], n_max_pos[i], n_life[i], n_init_r[i], n_den_init_int[i], n_ap_init_int[i])])
    end

    return NetworkDNA(net_size, ap_sink_f, nrf), dna_stack
end

# function supervised_train(episodes, iterations, data_input, data_output)
#
#     for e in 1:episodes
#         nets = []
#
#         for n in 1:PARALLEL_NETWORKS
#             rand_z = rand(net_latent_size)
#
#             rand_x = decode(rand_z, OUTPUT_SCALE)
#             net_dna, dna_stack = get_dna(rand_x)
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

function unsupervised_train(net_episodes::Integer, env_episodes::Integer, iterations::Integer, parallel_networks::Integer, env, env_version, init_params::Dict, net_params::Dict)

    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    encoder_model, decoders, model_params = init_network_generating_network(net_params)
    # model_params = [encoder_params, decoder_params]

    best_net_dna = nothing

    for e in 1:net_episodes
        nets = []
        zs = []
        xs = []

        for n in 1:parallel_networks
            rand_z = rand(net_params["LATENT_SIZE"]); append!(zs, rand_z);
            rand_x = decode(rand_z, decoders ,net_params["OUTPUT_SCALE"]); append!(xs, rand_x);

            net_dna, dna_stack = get_dna(rand_x, net_params["DNA_SAMPLE_SIZE"])
            net = initialize(net_dna, dna_stack, init_params)
            I = 1 # for counting iterations

            # training
            for ee in 1:env_episodes
                s = reset!(env)
                for i in 1:iterations
                    den_sinks, ap_sinks = value_step!(net, s)
                    state_step!(net, den_sinks, ap_sinks)
                    clean_network_components!(net)
                    runtime_instantiate_components!(net, I)
                    I += 1

                    out = get_output_nodes(get_all_all_cells(net))
                    a = action_space[argmax(out)]

                    r, s = step!(env, a)

                    net.total_fitness += r

                    if env.done
                        break
                    end
                end
            end

            append!(nets, (rand_z => net.total_fitness))
        end

        best_net_dna = collect(keys(nets))[argmax(collect(values(nets)))]

        z_rec_loss(x, z) = Flux.mse(encoder_model(x), zs)
        x_rec_loss(x, rx) = Flux.mse(decode(encoder_model(x), net_params["OUTPUT_SCALE"]), rx)

        # only train on reconstruction if above min loss

        z_rec_train_set = [(xs[i], zs[i]) for i in eachindex(xs, zs)]
        x_rec_train_set = [(xs[i], xs[i]) for i in eachindex(xs)]
        better_x_train_set = [(xs[i], best_net_dna) for i in eachindex(xs)]

        if x_rec_loss(rand(length(best_net_dna)) .* net_params["OUTPUT_SCALE"], rand(length(best_net_dna)) .* net_params["OUTPUT_SCALE"]) > net_params["MIN_RECONSTRUCTION_LOSS"]
            Flux.train!(z_rec_loss, model_params[1], z_rec_train_set, Flux.SGD)
            Flux.train!(x_rec_loss, [model_params[1]..., model_params[2]...], x_rec_train_set, Flux.SGD)
        end

        Flux.train!(x_rec_loss, model_params[2], better_x_train_set, Flux.SGD)
    end

    return best_net_dna
end




# order:
# dna_stack
#   nt_dna_samples
#       init_strength,
#       dispersion_region,
#       dispersion_strength_scale,
#       retain_percentage
#   ap_dna_samples
#       max_length
#       lifeTime
#   den_dna_samples
#       max_length
#       lifeTime
#   syn_dna_samples
#       THR
#       QDecay
#       lifeTime
#   n_dna_samples
#       max_num_priors
#       max_num_posteriors
#       lifeTime
#       dna_and_ap_init_range
#       den_init_interval
#       ap_init_interval
#   nn_dna
#       networkSize
#       ap_sink_force
#       neuron_repel_force



# sadly useless
# function collect_dna(NN::Network, nn_dna::NetworkDNA)
#     collection = []
#     n_samples = NN.dna_stack.n_dna_samples
#
#     for nts in NN.dna_stack.nt_dna_samples
#         append!(collection, nts.init_strength.min)
#         append!(collection, nts.init_strength.max)
#         append!(collection, nts.dispersion_region.x.min)
#         append!(collection, nts.dispersion_region.x.max)
#         append!(collection, nts.dispersion_region.y.min)
#         append!(collection, nts.dispersion_region.y.max)
#         append!(collection, nts.dispersion_region.z.min)
#         append!(collection, nts.dispersion_region.z.max)
#         append!(collection, nts.dispersion_strength_scale.min)
#         append!(collection, nts.dispersion_strength_scale.max)
#         append!(collection, nts.retain_percentage.min)
#         append!(collection, nts.retain_percentage.max)
#     end
#     for aps in NN.dna_stack.ap_dna_samples
#         append!(collection, aps.max_length.min)
#         append!(collection, aps.max_length.max)
#         append!(collection, aps.lifeTime.min)
#         append!(collection, aps.lifeTime.max)
#     end
#     for dens in NN.dna_stack.den_dna_samples
#         append!(collection, dens.max_length.min)
#         append!(collection, dens.max_length.max)
#         append!(collection, dens.lifeTime.min)
#         append!(collection, dens.lifeTime.max)
#     end
#     for syns in NN.dna_stack.syn_dna_samples
#         append!(collection, syns.THR.min)
#         append!(collection, syns.THR.max)
#         append!(collection, syns.QDecay.min)
#         append!(collection, syns.QDecay.max)
#         append!(collection, syns.lifeTime.min)
#         append!(collection, syns.lifeTime.max)
#     end
#     for ns in NN.dna_stack.n_dna_samples
#         append!(collection, ns.max_num_priors.min)
#         append!(collection, ns.max_num_priors.max)
#         append!(collection, ns.max_num_posteriors.min)
#         append!(collection, ns.max_num_posteriors.max)
#         append!(collection, ns.lifeTime.min)
#         append!(collection, ns.lifeTime.max)
#         append!(collection, ns.dna_and_ap_init_range.min)
#         append!(collection, ns.dna_and_ap_init_range.max)
#         append!(collection, ns.den_init_interval.min)
#         append!(collection, ns.den_init_interval.max)
#         append!(collection, ns.ap_init_interval.min)
#         append!(collection, ns.ap_init_interval.max)
#     end
#
#     append!(collection, nn_dna.networkSize.min)
#     append!(collection, nn_dna.networkSize.max)
#     append!(collection, nn_dna.ap_sink_force.min)
#     append!(collection, nn_dna.ap_sink_force.max)
#     append!(collection, nn_dna.neuron_repel_force.min)
#     append!(collection, nn_dna.neuron_repel_force.max)
#     return collection
# end
