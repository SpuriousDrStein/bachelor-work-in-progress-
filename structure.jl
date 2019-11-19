# IDEA: E ≤ V * 3 - 6
# where E = number of edges
# V = number of vertices

# main components are:
#    a layer -> set of edges
#    a direct connection -> connection from one discrete layer to the next
#    a residual connection -> connection from one layer to a residual graph


FloatN = Float32

mutable struct Vertex
    w::FloatN
end
mutable struct Edge
    value::FloatN
    b::FloatN
    thr::FloatN
end
mutable struct ResidualGraph
    inital_edges::Vector{Edge}
    edges::Vector{Edge}
    internal_vertecies::Array{Union{Vertex, Missing}, 2}
    external_vertecies::Array{Array{Union{Vertex, Missing}, 2}} # for each layer individually
    L::Integer
end
mutable struct Layer
    edges::Vector{Edge}
    direct_verticies::Array{Union{Vertex, Missing}, 2}
    residual_verticies::Array{Union{Vertex, Missing}, 2}
    L::Integer
end

random_vertex(init_dropout=0) = rand() >= init_dropout ? Vertex(rand()) : missing
random_edge(thr_mag=2) = Edge(0, rand() * thr_mag)
random_res_graph(internal_size, external_sizes) = ResidualGraph([random_edge() for _ in 1:internal_size], [random_vertex() for _ in 1:internal_size, _ in 1:internal_size], [[random_vertex() for _ in 1:internal_size, _ in 1:es_i] for es_i in external_sizes], internal_size)
random_layer(input_size, output_size, res_connections) = Layer([random_edge() for _ in 1:input_size], [random_vertex() for _ in 1:input_size, _ in 1:output_size], [random_vertex() for _ in 1:input_size, _ in 1:res_connections], input_size)




function update_edges!(l::Layer, x)
    for i in eachindex(l.edges, x)
        l.edges[i].value = x[i] + l.edges[i].b
    end
end
function update_edges!(rg::ResidualGraph, x)
    for i in eachindex(rg.edges, x)
        rg.edges[i].value = x[i] + rg.edges[i].b
    end
end

import Base.*
Base.*(e::Vector{Edge}, v::Array{Union{Vertex, Missing}, 2}) = [sum([e[i].value * v[i, j].w for i in skipmissing(v[:, j])]) for j in 1:length(v[1, :])]
Base.*(x::Vector, v::Array{Union{Vertex, Missing}, 2}) = [sum([x[i] * v[i, j].w for i in skipmissing(v[:, j])]) for j in 1:length(v[1, :])]

function forward!(layers::Array{Layer}, graph::ResidualGraph, x)

    # TODO GraphStateBuffer = []

    # 0. set edges initial state
    graph.edges = graph.initial_edges
    # TODO GraphStateBuffer[1] = graph.edges

    for i in eachindex(layers)
        # 1. get residual information for layer i from the graph
        layer_suplement = graph.edges * graph.external_vertices[i]
        # TODO V3-4: THIS CAN ALSO COLLECT RESIDUAL INFORMATION FOR EVERY LAYER AT THIS POINT

        # 2. update layer edges from input and residual information
        update_edges!(layers[i], x .+ layer_suplement)

        # 3. calculate transmition of info from layer to res graph
        res_in = layers[i].edges * layers[i].residual_vertices

        # 4. calculate new internal graph state
        update_edges!(graph, res_in * graph.internal_vertices)
        # TODO GraphStateBuffer[i+1] = graph.edges

        # 5. overwrite x
        x = layers[i].edges * layers[i].direct_verticies
    end

    #=
        (dE) d_x[L+1] / d_x[L] =
            (dE) -> direct_verticies[L] + (dE) -> external_verticies[L] -> internal_vertecies[@ i=L-1 or i=init] -> residual_vertecies[L]

        (dE) d_x[L+1] / d_w[L] =
            (dE) -> x[L]

        (dE) d_x[L+1] / d_b[L] =
            (dE) -> 1

        (dE) d_x[L+1] / d_residual_verticies[L] =
            (dE) -> external_vertices[L] -> internal_vertices ->(outer) x[L]

        (dE) d_x[L+1] / d_internal_verticies[L] =
            (dE) -> external_vertices[L] ->(outer) (x[L] * residual_vertices[L])


    =#


    return x
end











using OpenAIGym


env = GymEnv(:LunarLander, :v2)





# edges layer 1 = 3
# edges layer 2 = 4
# edges layer 3 = 2
# edges layer 4 = 5
# edges residual graph = 6

# direct connections l1 -> l2:
#   [ _ _ _ _ ]
#   [ _ _ _ _ ]
#   [ _ _ _ _ ]
# in a [from, to] matrix of verticies

# residual connections l1 -> ResidualGrap:
#   [ _ _ _ _ _ _ ]
#   [ _ _ _ _ _ _ ]
#   [ _ _ _ _ _ _ ]
# in a [from, to] matrix of vertices for each subsequent layer

# residual connections ResidualGraph -> [l1, l2, l3, l4]:
#   [ _ _ _, _ _ _ _, _ _ , _ _ _ _ _ ]
#   [ _ _ _, _ _ _ _, _ _ , _ _ _ _ _ ]
#   [ _ _ _, _ _ _ _, _ _ , _ _ _ _ _ ]
#   [ _ _ _, _ _ _ _, _ _ , _ _ _ _ _ ]
#   [ _ _ _, _ _ _ _, _ _ , _ _ _ _ _ ]
#   [ _ _ _, _ _ _ _, _ _ , _ _ _ _ _ ]
# in a [from, to] matrix of vertices for each subsequent layer

# residual connections ResidualGraph -> ResidualGrap:
#   [ _ _ _ _ _ _ ]
#   [ _ _ _ _ _ _ ]
#   [ _ _ _ _ _ _ ]
#   [ _ _ _ _ _ _ ]
#   [ _ _ _ _ _ _ ]
#   [ _ _ _ _ _ _ ]
# in a [from, to] matrix of vertices for each subsequent layer
