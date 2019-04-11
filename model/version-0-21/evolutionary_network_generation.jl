import Flux
import OpenAIGym
import Random

function init_network_generating_network(params::Dict)
    nt_sample_output_size = params["DNA_SAMPLE_SIZE"] * 2
    ap_sample_output_size = params["DNA_SAMPLE_SIZE"] * 2
    den_sample_output_size = params["DNA_SAMPLE_SIZE"] * 2
    syn_sample_output_size = params["DNA_SAMPLE_SIZE"] * 3
    n_sample_output_size = params["DNA_SAMPLE_SIZE"] * 6
    net_sample_output_size = 2

    net_input_size = ap_sample_output_size + den_sample_output_size + nt_sample_output_size + syn_sample_output_size + n_sample_output_size + net_sample_output_size
    net_latent_size = params["LATENT_SIZE"]
    latent_activation = params["LATENT_ACTIVATION"]
    encoder_hiddens = params["ENCODER_HIDDENS"]
    decoder_hiddens = params["DECODER_HIDDENS"]


    encoder_model = Flux.Chain(
        Flux.Dense(net_input_size, encoder_hiddens[1]),
        [Flux.Dense(encoder_hiddens[i-1], encoder_hiddens[i], Flux.relu) for i in 2:length(encoder_hiddens)]...,
        Flux.Dense(encoder_hiddens[end], net_latent_size, latent_activation))

    nt_model = Flux.Chain(
        Flux.Dense(net_latent_size, decoder_hiddens[1], Flux.leakyrelu),
        [Flux.Dense(decoder_hiddens[i-1], decoder_hiddens[i], Flux.leakyrelu) for i in 2:length(decoder_hiddens)]...,
        Flux.Dense(decoder_hiddens[end], nt_sample_output_size, Flux.leakyrelu))

    ap_model = Flux.Chain(
        Flux.Dense(net_latent_size, decoder_hiddens[1], Flux.relu),
        [Flux.Dense(decoder_hiddens[i-1], decoder_hiddens[i], Flux.relu) for i in 2:length(decoder_hiddens)]...,
        Flux.Dense(decoder_hiddens[end], ap_sample_output_size, Flux.relu))

    den_model = Flux.Chain(
        Flux.Dense(net_latent_size, decoder_hiddens[1], Flux.relu),
        [Flux.Dense(decoder_hiddens[i-1], decoder_hiddens[i], Flux.relu) for i in 2:length(decoder_hiddens)]...,
        Flux.Dense(decoder_hiddens[end], den_sample_output_size, Flux.relu))

    syn_model = Flux.Chain(
        Flux.Dense(net_latent_size, decoder_hiddens[1], Flux.relu),
        [Flux.Dense(decoder_hiddens[i-1], decoder_hiddens[i], Flux.relu) for i in 2:length(decoder_hiddens)]...,
        Flux.Dense(decoder_hiddens[end], syn_sample_output_size, Flux.relu))

    neuron_model = Flux.Chain(
        Flux.Dense(net_latent_size, decoder_hiddens[1], Flux.relu),
        [Flux.Dense(decoder_hiddens[i-1], decoder_hiddens[i], Flux.relu) for i in 2:length(decoder_hiddens)]...,
        Flux.Dense(decoder_hiddens[end], n_sample_output_size, Flux.relu))

    net_model = Flux.Chain(
        Flux.Dense(net_latent_size, decoder_hiddens[1], Flux.relu),
        [Flux.Dense(decoder_hiddens[i-1], decoder_hiddens[i], Flux.relu) for i in 2:length(decoder_hiddens)]...,
        Flux.Dense(decoder_hiddens[end], net_sample_output_size, Flux.relu))

    decoder_params = [Flux.params(net_model)..., Flux.params(neuron_model)..., Flux.params(syn_model)..., Flux.params(den_model)..., Flux.params(ap_model)..., Flux.params(nt_model)...]
    encoder_params = [Flux.params(encoder_model)...]

    return encoder_model, [nt_model, ap_model, den_model, syn_model, neuron_model, net_model], [encoder_params, decoder_params]
end


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
                params["FITNESS_DECAY"],
                params["NEURON_INIT_INTERVAL"],
                params["MIN_AP_DEN_INIT_INTERVAL"],
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

    means = [Flux.Tracker.data(x[i]) for i in 1:length(x)]

    nt_init_strs = [means[i] for i in 1:d]
    nt_retain_ps = [means[i] for i in (d)+1:(d*2)]

    ap_max_l = [means[i] for i in (d*2)+1:(d*3)]
    ap_life = [means[i] for i in (d*3)+1:(d*4)]

    den_max_l = [means[i] for i in (d*4)+1:(d*5)]
    den_life = [means[i] for i in (d*5)+1:(d*6)]

    syn_thr = [means[i] for i in (d*6)+1:(d*7)]
    syn_Qd = [means[i] for i in (d*7)+1:(d*8)]
    syn_life = [means[i] for i in (d*8)+1:(d*9)]

    n_max_pri = [means[i] for i in (d*9)+1:(d*10)]
    n_max_pos = [means[i] for i in (d*10)+1:(d*11)]
    n_life = [means[i] for i in (d*11)+1:(d*12)]
    n_init_r = [means[i] for i in (d*12)+1:(d*13)]
    n_den_init_int = [means[i] for i in (d*13)+1:(d*14)]
    n_ap_init_int = [means[i] for i in (d*14)+1:(d*15)]

    ap_sink_f = means[d*15+1]
    nrf = means[d*15+2]

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

    encoder_model, decoders, model_params = init_network_generating_network(params)
    z_rec_loss(x, z) = Flux.mse(encoder_model(x), z')
    x_rec_loss(x, rx) = Flux.mse(decode(encoder_model(x), decoders, params["OUTPUT_SCALE"]), rx')

    total_best_net = [-9999999, Flux.Tracker.data.(decode(rand(params["LATENT_SIZE"]), decoders, params["OUTPUT_SCALE"]))]
    metrics = Dict([("rec_loss" => []), ("best_net_fitness" => []), [("net_$(n)_final_neurons" => []) for n in 1:parallel_networks]...])

    for e in 1:net_episodes
        nets = []
        xs = []
        current_best_net = [-99999999, []]
        println("episode: $e")

        for n in 1:parallel_networks
            # every second iteration
            # either: try reconstructing best net
            # or: take random set of parameters
            if e % 2 == 0
                z = Flux.Tracker.data.(encoder_model(total_best_net[2]))
                x = decode(z ,decoders ,params["OUTPUT_SCALE"])
                append!(xs, [Flux.Tracker.data.(x)]);

                net_dna, dna_stack = get_dna(x, params)
                net = initialize(net_dna, dna_stack, params)
            else
                net_dna, dna_stack, x = get_random_set(params)
                append!(xs, [Flux.Tracker.data.(x)]);
                net = initialize(net_dna, dna_stack, params)
            end


            I = 1 # for counting iterations
            # training
            for ee in 1:env_episodes
                s = OpenAIGym.reset!(env)

                # reset_network_components!(net)

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

                    if env.done
                        break
                    end
                end
            end

            append!(metrics["net_$(n)_final_neurons"], [length(get_all_neurons(net))])
            append!(nets, [(net.total_fitness => copy(Flux.Tracker.data.(x)))])
        end

        current_best_net .= [sort(nets)[end][1], [n[2] for n in sort(nets)][end]]
        println("current_best_net_reward: $(current_best_net[1]) \t total_best_net_reward: $(total_best_net[1])")
        append!(metrics["best_net_fitness"], current_best_net[1]) # [sort(nets)[end][1]])

        if current_best_net[1] > total_best_net[1]
            total_best_net = current_best_net
        end

        x_rec_train_set = [(Flux.Tracker.data.(xs[i]), Flux.Tracker.data.(xs[i])) for i in eachindex(xs)]
        best_current_x_train_set = [(Flux.Tracker.data.(xs[i]), current_best_net[2]) for i in eachindex(xs)]
        best_total_x_train_set = [(Flux.Tracker.data.(xs[i]), total_best_net[2]) for i in eachindex(xs)]


        for t in Random.shuffle(best_current_x_train_set)
            if !isnan(x_rec_loss(t...)) && !isinf(x_rec_loss(t...))
                # println("trained on local best")
                Flux.train!(x_rec_loss, model_params[2], [t], Flux.Descent(params["LEARNING_RATE"]))
            end
        end
        for t in Random.shuffle(best_total_x_train_set)
            if !isnan(x_rec_loss(t...)) && !isinf(x_rec_loss(t...))
                Flux.train!(x_rec_loss, model_params[2], [t], Flux.Descent(params["LEARNING_RATE"]))
            end
        end

        # only train on reconstruction if above min loss
        test_rec_x = rand(length(xs[1])) .* params["OUTPUT_SCALE"]
        if x_rec_loss(test_rec_x, test_rec_x)  > params["MIN_RECONSTRUCTION_LOSS"]
            for t in Random.shuffle(x_rec_train_set)
                if !isnan(x_rec_loss(t...))  && !isinf(x_rec_loss(t...))
                    Flux.train!(x_rec_loss, [model_params[1]..., model_params[2]...], [t], Flux.Descent(params["LEARNING_RATE"]))
                    Flux.train!(x_rec_loss, [model_params[1]..., model_params[2]...], [t], Flux.Descent(params["LEARNING_RATE"]))
                end
            end
        end
        println("reconstruction loss: ", Flux.Tracker.data(x_rec_loss(test_rec_x, test_rec_x)))
        append!(metrics["rec_loss"], [Flux.Tracker.data(x_rec_loss(test_rec_x, test_rec_x))])
    end

    return total_best_net, metrics
end


function unsupervised_testing(env_episodes::Integer, iterations::Integer, net_dna, env, env_version, params::Dict)
    env = OpenAIGym.GymEnv(env, env_version)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')
    I = 1

    net_dna, stack = get_dna(net_dna, params["DNA_SAMPLE_SIZE"])
    net = initialize(net_dna, stack, init_params)

    for e in 1:net_episodes
        println("episode: $e")

        for ee in 1:env_episodes
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

                _, s = OpenAIGym.step!(env, a)

                # net.total_fitness += r

                OpenAIGym.render(env)

                # if env.done
                #     break
                # end
            end
        end
    end
end
