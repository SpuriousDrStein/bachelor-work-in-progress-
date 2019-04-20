using Distributions
using Plots
# Plots.pyplot()


include("utility_functions.jl")
include("network_operation_functions.jl")
include("evolutionary_network_generation.jl")


# BASE OVERLOAD
import Base.+, Base.-, Base./, Base.^, Base.*, Base.==, Base.!=
import Base.copy, Base.abs
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


function display_timestep(positions, connections, params, episode, iteration)
    l = @layout [a b{0.2w}]
    p1 = scatter(title="episode $episode || iteration $iteration", leg=false, xlabel="position x", ylabel="position y", xlims=(-params["NETWORK_SIZE"], params["NETWORK_SIZE"]), ylim=(-params["NETWORK_SIZE"], params["NETWORK_SIZE"]))
    scatter!([ac.x for ac in positions[1]], [ac.y for ac in positions[1]], c="blue")
    scatter!([ac.x for ac in positions[2]], [ac.y for ac in positions[2]], c="yellow")
    scatter!([ac.x for ac in positions[3]], [ac.y for ac in positions[3]], c="green")
    scatter!([ac.x for ac in positions[4]], [ac.y for ac in positions[4]], c="red")
    scatter!([ac.x for ac in positions[5]], [ac.y for ac in positions[5]], c="magenta")
    scatter!([ac.x for ac in positions[6]], [ac.y for ac in positions[6]], c="black")
    for c in connections
        plot!([c[1].x, c[2].x], [c[1].y, c[2].y], c="red", linewidth=0.1, linealpha=0.6,  l=:arrow)
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
