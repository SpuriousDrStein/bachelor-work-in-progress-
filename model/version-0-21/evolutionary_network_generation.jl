import Flux
import OpenAIGym


function init_network_generating_network(params::Dict)
    nt_sample_output_size = params["DNA_SAMPLE_SIZE"] * (2+(2*3)+2+2) # 2 * 3 * 2 + (2 * 3) # + (2 * 3) for 3 min-max pairs per possition in init_pos
    ap_sample_output_size = params["DNA_SAMPLE_SIZE"] * (2+2) # dna samples for 2 features * 2 for min and max values
    den_sample_output_size = params["DNA_SAMPLE_SIZE"] * (2+2)
    syn_sample_output_size = params["DNA_SAMPLE_SIZE"] * (2+2+2)
    n_sample_output_size = params["DNA_SAMPLE_SIZE"] * (2+2+2+2+2+2)
    net_sample_output_size = 2+2+2 # net_size, ap_sink_force and neuron_repel_force


    net_input_size = ap_sample_output_size + den_sample_output_size + nt_sample_output_size + syn_sample_output_size + n_sample_output_size + net_sample_output_size
    net_latent_size = params["LATENT_SIZE"]
    latent_activation = params["LATENT_ACTIVATION"]

    encoder_model = Flux.Chain(
        Flux.Dense(net_input_size, params["ENCODER_HIDDENS"][1]),
        [Flux.Dense(params["ENCODER_HIDDENS"][i-1], params["ENCODER_HIDDENS"][i], Flux.relu) for i in 2:length(params["ENCODER_HIDDENS"])]...,
        Flux.Dense(params["ENCODER_HIDDENS"][end], net_latent_size, latent_activation))

    nt_model = Flux.Chain(
        Flux.Dense(net_latent_size, params["DECODER_HIDDENS"][1], Flux.leakyrelu),
        [Flux.Dense(params["DECODER_HIDDENS"][i-1], params["DECODER_HIDDENS"][i], Flux.leakyrelu) for i in 2:length(params["DECODER_HIDDENS"])]...,
        Flux.Dense(params["DECODER_HIDDENS"][end], nt_sample_output_size, Flux.leakyrelu))

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
    encoder_params = [Flux.params(encoder_model)...]

    return encoder_model, [nt_model, ap_model, den_model, syn_model, neuron_model, net_model], (encoder_params, decoder_params)
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

    ns = nn.size / 2.

    input_nodes = [AllCell(InputNode(Possition(ns + rand(Uniform(-0.5, 1))*ns, ns + rand(Uniform(-0.5, 1))*ns, ns + rand(Uniform(-0.5, 1))*ns), 0.)) for i in 1:params["DATA_INPUT_SIZE"]]
    out_nodes = [AllCell(OutputNode(Possition(-ns + rand(Uniform(-1, 0.5))*ns, -ns + rand(Uniform(-1, 0.5))*ns, -ns + rand(Uniform(-1, 0.5))*ns), 0.)) for i in 1:params["DATA_OUTPUT_SIZE"]]

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

function get_dna(x, dna_sample_size)
    d = dna_sample_size # because shorter indexing

    min_max_pairs = [min_max_pair(Flux.Tracker.data(x[i]), Flux.Tracker.data(x[i+1])) for i in 1:2:length(x)]

    nt_init_strs = [min_max_pairs[i] for i in 1:d]
    nt_disp_regs = [InitializationPossition(min_max_pairs[i], min_max_pairs[i+1], min_max_pairs[i+2]) for i in d+1:3:(4*d)]
    nt_disp_strs = [min_max_pairs[i] for i in (d*4)+1:(d*5)]
    nt_retain_ps = [min_max_pairs[i] for i in (d*5)+1:(d*6)]

    ap_max_l = [min_max_pairs[i] for i in (d*6)+1:(d*7)]
    ap_life = [min_max_pairs[i] for i in (d*7)+1:(d*8)]

    den_max_l = [min_max_pairs[i] for i in (d*8)+1:(d*9)]
    den_life = [min_max_pairs[i] for i in (d*9)+1:(d*10)]

    syn_thr = [min_max_pairs[i] for i in (d*10)+1:(d*11)]
    syn_Qd = [min_max_pairs[i] for i in (d*11)+1:(d*12)]
    syn_life = [min_max_pairs[i] for i in (d*12)+1:(d*13)]


    n_max_pri = [min_max_pairs[i] for i in (d*13)+1:(d*14)]
    n_max_pos = [min_max_pairs[i] for i in (d*14)+1:(d*15)]
    n_life = [min_max_pairs[i] for i in (d*15)+1:(d*16)]
    n_init_r = [min_max_pairs[i] for i in (d*16)+1:(d*17)]


    n_den_init_int = [min_max_pairs[i] for i in (d*17)+1:(d*18)]
    n_ap_init_int = [min_max_pairs[i] for i in (d*18)+1:(d*19)]

    net_size = min_max_pairs[d*19+1]
    ap_sink_f = min_max_pairs[d*19+2]
    nrf = min_max_pairs[d*19+3]

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

function unsupervised_train(net_episodes::Integer, env_episodes::Integer, iterations::Integer, parallel_networks::Integer, env, env_version, params::Dict)

    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    encoder_model, decoders, model_params = init_network_generating_network(params)
    # model_params = [encoder_params, decoder_params]

    best_net_dna = nothing

    for e in 1:net_episodes
        nets = []
        zs = []
        xs = []
        println("episode: $e")

        for n in 1:parallel_networks
            rand_z = rand(params["LATENT_SIZE"])
            rand_x = decode(rand_z, decoders ,params["OUTPUT_SCALE"])
            append!(zs, [rand_z]);
            append!(xs, [Flux.Tracker.data(rand_x)]);

            net_dna, dna_stack = get_dna(rand_x, params["DNA_SAMPLE_SIZE"])
            net = initialize(net_dna, dna_stack, params)
            I = 1 # for counting iterations

            # training
            for ee in 1:env_episodes
                s = OpenAIGym.reset!(env)

                for i in 1:iterations
                    den_sinks, ap_sinks = value_step!(net, Array(s) .+ .5)
                    state_step!(net, den_sinks, ap_sinks)
                    clean_network_components!(net)
                    runtime_instantiate_components!(net, I)
                    I += 1

                    out = [on.value for on in get_output_nodes(net)]
                    a = action_space[argmax(out)]

                    r, s = OpenAIGym.step!(env, a)

                    net.total_fitness += r


                    # if env.done
                    #     break
                    # end
                end
            end

            println("net $n fitness: $(net.total_fitness)")
            append!(nets, [(copy(rand_x) => net.total_fitness)])
        end

        best_net_dna = collect(keys(nets))[argmax(collect(values(nets)))]

        # println(encoder_model)

        z_rec_loss(x, z) = begin
            pr = encoder_model(x)
            Flux.mse(pr, z)
        end
        x_rec_loss(x, rx) = Flux.mse(decode(encoder_model(x), decoders, params["OUTPUT_SCALE"]), rx')

        z_rec_train_set = [(Flux.Tracker.data(xs[i]), Flux.Tracker.data(zs[i])) for i in eachindex(xs, zs)]
        x_rec_train_set = [(Flux.Tracker.data(xs[i]), Flux.Tracker.data(xs[i])) for i in eachindex(xs)]
        better_x_train_set = [(Flux.Tracker.data(xs[i]), best_net_dna) for i in eachindex(xs)]

        println(xs[1])
        # println(x_rec_train_set[1])
        # println(better_x_train_set[1])

        # only train on reconstruction if above min loss
        test_rec_x = rand(length(xs[1])) .* params["OUTPUT_SCALE"]
        if x_rec_loss(test_rec_x, test_rec_x)  > params["MIN_RECONSTRUCTION_LOSS"]
            Flux.train!(z_rec_loss, model_params[1], z_rec_train_set,  Flux.Descent(params["LEARNING_RATE"]))
            Flux.train!(x_rec_loss, [model_params[1]..., model_params[2]...], x_rec_train_set, Flux.Descent(params["LEARNING_RATE"]))
        end

        Flux.train!(x_rec_loss, model_params[2], better_x_train_set, Flux.Descent(params["LEARNING_RATE"]))
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
