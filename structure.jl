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
    edges::Vector{Edge}
    internal_vertecies::Array{Vertex, 2}
    external_vertecies::Array{Vertex, 2}
    L::Integer
end
mutable struct Layer
    edges::Vector{Edge}
    direct_verticies::Array{Union{Vertex, Missing}, 2}
    residual_verticies::Array{Union{Vertex, Missing}, 2}
    L::Integer
end

random_vertex(init_dropout=0.5) = rand() >= (1-init_dropout) ? Vertex(rand()) : missing
random_edge(thr_mag=2) = Edge(0, rand()*thr_mag)
random_res_graph(internal_size, external_size) = ResidualGraph([random_edge() for _ in 1:internal_size], [random_vertex() for _ in 1:internal_size, _ in 1:internal_size], [random_vertex() for _ in 1:internal_size, _ in 1:external_size], internal_size)
random_layer(input_size, output_size, res_connections) = Layer([random_edge() for _ in 1:input_size], [random_vertex() for _ in 1:input_size, _ in 1:output_size], [random_vertex() for _ in 1:input_size, _ in 1:res_connections], input_size)


function update_value!(l::Layer, x)
    for i in eachindex(l.edges, x)
        l.edges[i].value = x[i] + l.edges[i].b
    end
end
function update_value!(rg::ResidualGraph, x)
    for i in eachindex(rg.edges, x)
        rg.edges[i].value = x[i] + rg.edges[i].b
    end
end
function forward(l::Layer)
    direct_out = zeros(axes(l.direct_vertices, 2))
    res_out = zeros(axes(l.residual_vertices, 2))

    for i in axes(l.direct_vertices, 1)
        for j in axes(l.direct_vertecies, 2)
            direct_out[j] += (l.edges[i].value * l.direct_verticies[i,j].w) / l.L
        end

        for k in axes(l.residual_verticies, 2)
            res_out[k] += (l.edges[i].value * l.residual_verticies[i,k].w) / l.L
        end
    end
    return direct_out, res_out
end
function forward(rg::ResidualGraph)
    internal_out = zeros(axes(rg.internal_verticies, 2))
    external_out = zeros(axes(rg.external_verticies, 2))

    for i in axes(rg.internal_verticies, 1)
        for j in axes(rg.internal_verticies, 2)
            internal_out[j] += (rg.edges[i].value * rg.internal_verticies[i,j].w) / rg.L
        end

        for k in axes(rg.external_verticies, 2)
            external_out[k] += (rg.edges[i].value * rg.external_vertices[i,k].w) / rg.L
        end
    end
    return direct_out, res_out
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
