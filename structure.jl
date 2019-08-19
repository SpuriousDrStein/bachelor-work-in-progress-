using Flux

# STRUCTURE
mutable struct edge
    VAL::AbstractFloat

    thr::AbstractFloat
    res_coef::AbstractFloat # recovery speed
    freq_coef::AbstractFloat # frequency
end
mutable struct node
    VAL::AbstractFloat

    thr::AbstractFloat
    res_coef::AbstractFloat
    freq_coef::AbstractFloat
end

conectum = Array{edge, 2}
mask = Array{Bool, 2}
neuron_layer = Array{node, 1}

mutable struct network
    neuron_layers::Array{neuron_layer, 1}
    conectums::Array{conectum, 1}
    masks::Array{mask, 1}
end

# UTILITY FUNCTIONS
function Base.getindex(n::node, i::Integer); return n[i]; end
raw(e::edge) = [copy(e.thr), copy(e.res_coef), copy(e.freq_coef)]
raw(n::node) = [copy(n.thr), copy(n.res_coef), copy(n.freq_coef)]
function Base.copy(x::neuron_layer); return collect(Iterators.flatten([raw(l) for l in x])); end
function Base.copy(x::conectum); return collect(Iterators.flatten([raw(l) for l in x])); end

# OTHER ///UC (under construction)
function update!(nl::neuron_layer, x::AbstractArray, d_ut)
    out = [0. for _ in 1:length(nl)]
    for ni in eachindex(nl)
        nl[ni].VAL += x[ni]
        nl[ni].VAL += d_ut * nl[ni].res_coef
        nl[ni].VAL += d_ut * cos(nl[ni].freq_coef)

        nl[ni].res_coef += d_ut * (nl[ni].VAL - nl[ni].thr) # negative if not fiering -> making the resistance coeff negative -> increase probabilty of fiering
        nl[ni].freq_coef = (nl[ni].freq_coef + d_ut) * (nl[ni].freq_coef < 2π)

        out[ni] = nl[ni].VAL * (nl[ni].VAL >= nl[ni].thr)
    end
    return out
end
function update!(con::conectum, m::mask, x::AbstractArray, d_ut)
    out = [0. for _ in axes(con, 2)]
    for xi in eachindex(x)
        for (ai, syn) in enumerate(con[xi, :])
            if m[xi, ai] # if is true in mask
                syn.VAL += x[xi]
                syn.VAL += d_ut * syn.res_coef
                syn.VAL += d_ut * cos(syn.freq_coef)

                syn.res_coef += d_ut * (syn.VAL - syn.thr)
                syn.freq_coef = (syn.freq_coef + d_ut) * (syn.freq_coef < 2π)

                out[ai] += copy(syn.VAL) * (syn.VAL >= syn.thr)
            end
        end
    end
    return out
end

function ward!(net::network, x::AbstractArray, d_ut)
    o = update!(net.neuron_layers[1], x, d_ut)
    o = update!(net.conectums[1], net.masks[1], o, d_ut)
    for i in 2:length(net.neuron_layers)
        o = update!(net.neuron_layers[i], o, d_ut)
        o = update!(net.conectums[i], net.masks[i], o, d_ut)
    end
    return o
end

function construct_network(params::AbstractVector, hidden_sizes::AbstractVector; number_of_edge_params=4, number_of_node_params=4)::network
    ret_net = network([], [], [])
    it = 1

    for i in 1:length(hidden_sizes)-1
        neuron_param_ind   = it + hidden_sizes[i] * number_of_node_params
        conectum_param_ind = neuron_param_ind + hidden_sizes[i] * hidden_sizes[i+1] * number_of_edge_params
        mask_param_ind     = conectum_param_ind + hidden_sizes[i] * hidden_sizes[i+1]

        println("lay = $i")
        println("it  = $it")
        println("npi = $neuron_param_ind")
        println("cpi = $conectum_param_ind")
        println("mpi = $mask_param_ind")

        println("\n $(params[it:it+number_of_node_params-1]) \n")

        append!(ret_net.neuron_layers, [neuron_layer([node(params[j:j+number_of_node_params-1]...) for j in it:number_of_node_params:neuron_param_ind-1])])
        append!(ret_net.conectums, [conectum(reshape([edge(params[j:j+number_of_edge_params-1]...) for j in neuron_param_ind:number_of_edge_params:conectum_param_ind-1], (hidden_sizes[i], hidden_sizes[i+1])))])
        append!(ret_net.masks, [mask(reshape(Bool.(clamp.(round.(params[conectum_param_ind:mask_param_ind-1]), 0, 1)), (hidden_sizes[i], hidden_sizes[i+1])))])

        it = mask_param_ind
    end

    # INFO: order - neurons, conectum, mask, neurons, conectum, mask, ..., mask, neurons
    return ret_net
end

function deconstruct_network(net::network)
    ret_par = []
    for l in eachindex(net.conectums, net.masks)
        append!(ret_par, copy(net.neuron_layers[l]))
        append!(ret_par, copy(net.conectums[l]))
        append!(ret_par, copy(net.masks[l]))
    end
    append!(ret_par, copy(net.neuron_layers[end]))

    return collect(Iterators.flatten(ret_par))
end


# TEST
X = rand(15)
hidden_sizes = [15, 10, 5, 2, 5, 10, 15]
no_node_params = 4
no_edge_params = 4
no_params_per_lay = [hidden_sizes[i] * no_node_params + hidden_sizes[i] * hidden_sizes[i+1] * no_edge_params + hidden_sizes[i] * hidden_sizes[i+1] for i in 1:length(hidden_sizes)-1]
d_UT = 0.01 |> AbstractFloat # update time
d_RT = 0.5 |> AbstractFloat # reaction time

parrs = rand(sum(no_params_per_lay))
net = construct_network(parrs, hidden_sizes)
feed_forward!(net, X, d_UT) |> show

pars = deconstruct_network(net)
pars .|> round
construct_network(pars, hidden_sizes)


training_episodes = 50
parallel_agents = 5
agent_runtime = 30


function train!(initial_parameter_collection, hidden_sizes, env_name, env_version, training_episodes, parallel_agents, agent_runtime; inverse_percentile=70)

    env = OpenAIGym.make_env(env_name, env_version)
    networks = [construct_network(ip, hidden_sizes) for ip in initial_parameter_collection]

    for e in 1:training_episodes

        net_buffer = []

        for k in 1:length(networks)

            for e in 1:runtime_episodes

                s = reset!(env)
                rr = 0 # reward record

                for t in d_UT:d_UT:agent_runtime

                    o = feed_forward!(agents[k], s |> Array)

                    if t % d_RT == 0
                        a = env.actionspace(argmax(o))
                        step!(env, a)

                        rr += env.reward
                    end
                end

                append!(net_buffer, [(rr, deconstruct_network(agents[k]))])
            end
        end

        percentile_margin = max([r[1] for r in net_buffer]) / 100 * inverse_percentile

        top_nets = collect(Iterators.filter(x->x>percentile_margin, net_buffer))

        for i in eachindex(networks)
            y = deconstruct_network(network[i])

            for tn in top_nets
                y_hat = tn[2]
                # forward pass y-y_hat for each top network
            end

            # reconstruction error
        end
    end
end
