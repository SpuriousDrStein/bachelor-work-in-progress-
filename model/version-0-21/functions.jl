using Distributions
using Plots
using OpenAIGym
import Distributions
using Random
using StatsBase

include(".\\..\\..\\global_utility_functions\\activation_functions.jl")
include("utility_functions.jl")
include("network_operation_functions.jl")
include("evolutionary_network_generation.jl")



# MULTIPLE DISPATCH
import Base.+, Base.-, Base./, Base.^, Base.*, Base.==, Base.!=
import Base.copy, Base.abs
import StatsBase.sample
import LinearAlgebra.normalize
function +(p1::Position, p2::Position); Position(([p1.x, p1.y] .+ [p2.x, p2.y])...); end
function -(p1::Position, p2::Position); Position(([p1.x, p1.y] .- [p2.x, p2.y])...); end
function /(p::Position, n::Number); Position([p.x, p.y] ./ n); end
function ^(p::Position, pow::Integer); [p.x, p.y] .^ pow; end
function *(p::Position, n::Number); Position(([p.x, p.y] .* n)...); end
function ==(p1::Position, p2::Position); all([p1.x == p2.x, p1.y == p2.y]); end
function !=(p1::Position, p2::Position); all([p1.x != p2.x, p1.y != p2.y]); end
function vec_mean(p1::Position, p2::Position); Position((p1 + p2) ./ 2.); end
function copy(p::Position); Position(copy(p.x), copy(p.y)); end
function abs(p::Position); Position([abs(p.x), abs(p.y)]...); end
function sample(mean::FloatN, global_stdv::FloatN); rand(Normal(mean, global_stdv)); end
function normalize(p::Position); [p.x, p.y] ./ vector_length(p); end
function normalize(v::Vector); v ./ vector_length(v); end



# VISUALIZATION
function display_timestep(positions, connections, params, episode, iteration) # [np, app, denp, synp, inp, outp], cons
    l = @layout [a b{0.2w}]
    p1 = scatter(title="episode $episode; iteration $iteration", leg=false, xlabel="position x", ylabel="position y", xlims=(-params["NETWORK_SIZE"], params["NETWORK_SIZE"]), ylim=(-params["NETWORK_SIZE"], params["NETWORK_SIZE"]))
    scatter!([ac[1].x for ac in positions[1]], [ac[1].y for ac in positions[1]], c="blue", markersize=[3+ac[2] for ac in positions[1]])
    scatter!([ac[1].x for ac in positions[2]], [ac[1].y for ac in positions[2]], c="yellow", markersize=[3 for ac in positions[2]])
    scatter!([ac[1].x for ac in positions[3]], [ac[1].y for ac in positions[3]], c="green", markersize=[3 for ac in positions[3]])
    scatter!([ac[1].x for ac in positions[4]], [ac[1].y for ac in positions[4]], c="red", markersize=[3+ac[2] for ac in positions[4]])
    scatter!([ac[1].x for ac in positions[5]], [ac[1].y for ac in positions[5]], c="magenta", markersize=[3+ac[2] for ac in positions[5]])
    scatter!([ac[1].x for ac in positions[6]], [ac[1].y for ac in positions[6]], c="black", markersize=[3+ac[2] for ac in positions[6]])

    for c in connections
        plot!([c[1].x, c[2].x], [c[1].y, c[2].y], c="red", linewidth=0.1, linealpha=0.6, l=:arrow)
    end

    for l in 1:length(params["LAYERS"])
        plot!([cos.(-π:0.001:π).*(params["NETWORK_SIZE"]/length(params["LAYERS"])*l)], [sin.(-π:0.001:π).*(params["NETWORK_SIZE"]/length(params["LAYERS"])*l)], linealpha=0.3)
    end

    p2 = scatter(grid=false, showaxis=false, xlims=(0,0))
    scatter!([0], label="neurons", c="blue")
    scatter!([0], label="axon points", c="yellow")
    scatter!([0], label="dendrites", c="green")
    scatter!([0], label="synapses", c="red")
    scatter!([0], label="input nodes", c="magenta")
    scatter!([0], label="output nodes", c="black")

    display(plot(p1, p2, layout=l))
end
