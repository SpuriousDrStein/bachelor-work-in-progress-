# include(".\\..\\..\\global_utility_functions\\activation_functions.jl")
import Distributions.sample, Distributions.Weights, Distributions.Normal
using Random
using OpenAIGym
using Plots
using Flux

FloatN = Float16
IntN = Int16

mutable struct Edge
    Q::FloatN
    R::FloatN
    V::FloatN
    I::FloatN

    THR::FloatN
    α::FloatN # α = resistance decay
    c::FloatN
end
mutable struct Node
    Q::FloatN
    R::FloatN
    V::FloatN
    I::FloatN

    THR::FloatN
    α::FloatN
    c::FloatN
end
EdgeMatrix = Array{Union{Missing, Edge}, 2}
NodeMatrix = Array{Node, 1}
DropoutMask = Array{Bool, 2}
mutable struct Record
    fitness::Float32
    edges::EdgeMatrix
    nodes::NodeMatrix
    mask::DropoutMask
end


function MainReset!(nodes::NodeMatrix, edges::EdgeMatrix)
    for n in nodes
        if n.Q / (n.c * n.R) >= n.THR
            n.Q = 0
        end
    end
    for e in skipmissing(edges)
        if e.Q / (e.c * e.R) >= e.THR
            e.Q = 0
        end
    end
end
function MainUpdate!(X::AbstractArray, nodes::NodeMatrix, edges::EdgeMatrix, params::AbstractDict, out_size::Int)
    # feed input
    for i in 1:length(X);   nodes[i].Q = copy(X[i]);    end

    for n1_i in eachindex(nodes)
        for n2_i in eachindex(nodes)
            syn = edges[n1_i, n2_i]

            if !ismissing(syn)
                if n1_i == n2_i
                    # loopback connection
                elseif n1_i > n2_i
                    # connection from n1 to n2
                    n1 = nodes[n1_i]; n2 = nodes[n2_i]
                    n_act_I = n1.Q / (n1.c * n1.R) * (n1.Q >= n1.THR)

                    syn.V = syn.R * n_act_I
                    syn.I = syn.V / syn.R
                    syn.Q = (syn.c * syn.I) / syn.R
                    syn.R = params["MIN_INIT_R"] + (syn.α * syn.R + (1 - syn.α) * syn.Q)

                    s_act_I = syn.Q / (syn.c * syn.R) * (syn.Q >= syn.THR)

                    n2.V *= n2.α * n2.V + (1 - n2.α) * n2.R * s_act_I
                    n2.I *= n2.α * n2.I + (1 - n2.α) * n2.V / n2.R
                    n2.Q *= n2.α * n2.Q + (1 - n2.α) * (n2.c * n2.I) / n2.R
                    n2.R = params["MIN_INIT_R"] + (n2.α * n2.R + (1 - n2.α) * n2.Q)
                elseif n2_i > n1_i
                    # connection from n2 to n1
                end
            end
        end
    end

    # read output
    out = [copy(nodes[i].Q) for i in length(nodes):-1:length(nodes)-out_size+1]

    MainReset!(nodes, edges)
    return out
end

function ApplyMask(edges::EdgeMatrix, mask::DropoutMask)
    newE = []
    for ei in eachindex(edges, mask)
        if !mask[ei]
            append!(newE, [missing])
        else
            append!(newE, [edges[ei]])
        end
    end
    return EdgeMatrix(reshape(newE, size(edges)))
end
function unravel(r::Record)
    out = []
    for ri in r.edges
        append!(out, [copy(ri.Q), copy(ri.R), copy(ri.V), copy(ri.I), copy(ri.THR), copy(ri.α), copy(ri.c)])
    end
    for ri in r.nodes
        append!(out, [copy(ri.Q), copy(ri.R), copy(ri.V), copy(ri.I), copy(ri.THR), copy(ri.α), copy(ri.c)])
    end
    for ri in r.mask
        append!(out, [copy(ri)])
    end
    return out
end
function ravel(in, num_nodes; init_fitness=0)
    new_edges, new_nodes, new_mask = [], [], []
    num_edges = num_nodes ^ 2

    for i in 1:7:num_edges*7
        append!(new_edges, [Edge(copy(in[i]), copy(in[i+1]), copy(in[i+2]), copy(in[i+3]), copy(in[i+4]), copy(in[i+5]), copy(in[i+6]))])
    end
    for i in num_edges*7+1:7:num_edges*7+num_nodes*7
        append!(new_nodes, [Node(copy(in[i]), copy(in[i+1]), copy(in[i+2]), copy(in[i+3]), copy(in[i+4]), copy(in[i+5]), copy(in[i+6]))])
    end
    for i in num_edges*7+num_nodes*7+1:num_edges*7+num_nodes*7+num_edges
        append!(new_mask, clamp(round(in[i]), 0, 1))
    end

    return Record(init_fitness, reshape(new_edges, (num_nodes, num_nodes)), new_nodes, reshape(new_mask, (num_nodes, num_nodes)))
end


function sample(e::Edge, params::AbstractDict)
    r = clamp(rand(Normal(e.R, params["STDV"])), params["MIN_INIT_R"], params["MAX_INIT_R"])
    v = clamp(rand(Normal(e.V, params["STDV"])), params["MIN_INIT_V"], params["MAX_INIT_V"])
    thr = clamp(rand(Normal(e.THR, params["STDV"])), params["MIN_THR"], params["MAX_THR"])
    al = clamp(rand(Normal(e.α, params["STDV"])), 0.1, 0.9)
    c = clamp(rand(Normal(e.c, params["STDV"])), params["MIN_C"], params["MAX_C"])
    return Edge(FloatN(0.), r, v, FloatN(0.), thr, al, c)
end
function sample(n::Node, params::AbstractDict)
    r = clamp(rand(Normal(n.R, params["STDV"])), params["MIN_INIT_R"], params["MAX_INIT_R"])
    v = clamp(rand(Normal(n.V, params["STDV"])), params["MIN_INIT_V"], params["MAX_INIT_V"])
    thr = clamp(rand(Normal(n.THR, params["STDV"])), params["MIN_THR"], params["MAX_THR"])
    al = clamp(rand(Normal(n.α, params["STDV"])), 0.1, 0.9)
    c = clamp(rand(Normal(n.c, params["STDV"])), params["MIN_C"], params["MAX_C"])
    return typeof(n)(0., r, v, 0., thr, al, c)
end
function sample(M::EdgeMatrix, params::AbstractDict)
    newM = []
    for m in M
        append!(newM, [sample(m, params)])
    end
    return EdgeMatrix(reshape(newM, size(M)))
end
function sample(M::NodeMatrix, params::AbstractDict)
    newM = []
    for m in M
        append!(newM, [sample(m, params)])
    end
    return NodeMatrix(newM)
end
function sample(M::DropoutMask, params::AbstractDict)
    newM = []
    for m in M
        if rand() < params["MASK_MUTATION_PROBABILITY"]
            append!(newM, [true])
        else
            append!(newM, [false])
        end
    end
    return DropoutMask(reshape(newM, size(M)))
end
function sample(records::Array{Record}, params::AbstractDict)
    dist = softmax([r.fitness / mean([r2.fitness for r2 in records]) for r in records])
    selected_record = sample(Random.GLOBAL_RNG, records, Weights(dist))
    return sample(selected_record.nodes, params), sample(selected_record.edges, params), sample(selected_record.mask, params)
end
function sample(records, encoder_model, decoder_model)
    new_records = []
    for r in records
        raw_r = unravel(r)
        new_raw_r = Flux.Tracker.data(decoder_model(encoder_model(raw_r)))
        new_r = ravel(new_raw_r, length(r.nodes))
        append!(new_records, [new_r])
    end
    return new_records
end
function Train!(env, records::Array{Record}, train_episodes::Int, env_episodes::Int, env_iterations::Int, params::AbstractDict, render_::Bool)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')

    survey = Dict("total_current_reward" => [])

    for te in 1:train_episodes
        nodes, raw_edges, mask = sample(records, params)
        edges = ApplyMask(raw_edges, mask)
        total_current_reward = 0

        for ee in 1:env_episodes
            s = reset!(env)

            for ei in 1:env_iterations
                s = Array(s)
                a = MainUpdate!(s, nodes, edges, params)

                # println("out = $a")

                if render_
                    render(env)
                end

                a = action_space[argmax(a)]
                r, s = step!(env, argmax(a))

                total_current_reward += (r * ei)

                # println(["$(n.V) " for n in nodes]...)

                if env.done
                    break
                end
            end
        end



        append!(survey["total_current_reward"], [total_current_reward])
        println("iteration = $te :: reward = $total_current_reward")

        for rec_i in eachindex(records)
            if total_current_reward > records[rec_i].fitness
                records[rec_i] = Record(copy(total_current_reward), copy(raw_edges), copy(nodes), copy(mask))
                break
            end
        end
    end
    OpenAIGym.close(env)
    return records, survey
end
function Train!(env, records::Array{Record}, encoder_model, decoder_model, params::AbstractDict, train_episodes::Integer, env_episodes::Integer, env_iterations::Integer, out_size::Integer, render_::Bool; lr=0.33)
    action_index = [i for i in 1:length(env.actions)]
    action_space = one(action_index * action_index')
    survey = Dict("avg_env_reward" => [],
                    "avg_rec_loss" => [],
                    "avg_const_loss" => [])

    construction_loss(x, y) = Flux.mse(decoder_model(encoder_model(x)), y)


    for te in 1:train_episodes
        records = sample(records, encoder_model, decoder_model)
        best_record = records[1]

        for r in records
            nodes, raw_edges, mask = r.nodes, r.edges, r.mask
            edges = ApplyMask(raw_edges, mask)
            total_current_reward = 0

            for ee in 1:env_episodes
                s = reset!(env)

                for ei in 1:env_iterations
                    s = Array(s)
                    a = MainUpdate!(s, nodes, edges, params, out_size)

                    # println("out = $a")

                    if render_
                        render(env)
                    end

                    a = action_space[argmax(a)]
                    reward, s = step!(env, argmax(a))

                    total_current_reward += (reward * ei)

                    # println(["$(n.V) " for n in nodes]...)

                    if env.done
                        break
                    end
                end
            end

            if r.fitness > best_record.fitness
                best_record = r
            end
        end


        avg_rec_loss = 0
        avg_const_loss = 0
        for r in records
            ur = Flux.Tracker.data(unravel(r))

            # reconstruct
            Flux.train!(construction_loss, [Flux.params(encoder_model)..., Flux.params(decoder_model)...], [(ur, ur)], Flux.SGD([], lr))
            # construct better
            Flux.train!(construction_loss, [Flux.params(encoder_model)..., Flux.params(decoder_model)...], [(ur, Flux.Tracker.data(unravel(best_record)))], Flux.SGD([], lr))

            avg_rec_loss += construction_loss(r, r)
            avg_const_loss += construction_loss(r, best_record)
        end
        println("iteration = $te :: avg. reward = $total_current_reward")

        append!(survey["avg_env_reward"], [total_current_reward / length(records)])
        append!(survey["avg_rec_loss"], [avg_rec_loss / length(records)])
        append!(survey["avg_const_loss"], [avg_const_loss / length(records)])
    end

    OpenAIGym.close(env)
    return records, survey
end



pars = Dict(
    "MIN_INIT_R" => 0.01,
    "MAX_INIT_R" => 10,
    "MIN_INIT_V" => 0.1,
    "MAX_INIT_V" => 10,
    "MIN_THR" => 0.5,
    "MAX_THR" => 10,
    "MIN_C" => 0.5,
    "MAX_C" => 1.1,
    "STDV" => 0.1,
    "MASK_MUTATION_PROBABILITY" => 0.1)

environment = GymEnv(:CartPole, :v1)
net_size = 3
input_size = length(environment.state)
output_size = length(environment.actions)
total_width = net_size + input_size + output_size
total_size = 7 * total_width + 7 * total_width^2 + total_width^2 # 7 parameters * ( em size (2 per node) + nm size (1) ) + 1 parameter * dm size (2 per node)
encoder_hiddens = [200, 100, 50, 30]
hidden_size = 20
decoder_hiddens = [30, 50, 100, 200]
parallel_networks = 3

train_episodes = 500
env_episodes = 30
env_iterations = 500



em = EdgeMatrix(fill(Edge(rand(), rand(), rand(), rand(), rand(), rand(), rand()), (total_width, total_width)))
nm = NodeMatrix([Node(rand(), rand(), rand(), rand(), rand(), rand(), rand()) for _ in 1:net_size+input_size+output_size])
dm = DropoutMask(reshape([rand()>0.5 for _ in 1:total_width*total_width], (total_width, total_width)))
records = [Record(0, copy(em), copy(nm), copy(dm)) for _ in 1:parallel_networks]

encoder_model = Flux.Chain(
    Dense(total_size, encoder_hiddens[1], relu),
    [Dense(encoder_hiddens[i], encoder_hiddens[i+1], relu) for i in 1:length(encoder_hiddens)-1]...,
    Dense(encoder_hiddens[end], hidden_size, sigmoid))
decoder_model = Flux.Chain(
    Dense(hidden_size, decoder_hiddens[1], relu),
    [Dense(decoder_hiddens[i], decoder_hiddens[i+1], relu) for i in 1:length(decoder_hiddens)-1]...,
    Dense(decoder_hiddens[end], total_size, relu))

encoder_model(unravel(records[1])) |> decoder_model

records, observations = Train!(environment, records, encoder_model, decoder_model, pars, train_episodes, env_episodes, env_iterations, output_size, false)



plot(observations["total_current_reward"])

# # Q::FloatN
# # R::FloatN
# # V::FloatN
# # I::FloatN
# # THR::FloatN
# # α::FloatN
# # c::FloatN
# records
