using Distributions
using Plots

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


function display_network(metrics, net, episodes)

    for e in episodes[1]:episodes[2] # np, app, denp, synp, inp, outp
        for (i, poss) in enumerate(metrics["net_$(net)_position_episode_$(e)"])
            l = @layout [a b{0.2w}]
            p1 = Plots.scatter(title="net $(net); episode $e; iteration $i", leg=false, xlabel="position x", ylabel="position y", xlims=(-init_params["NETWORK_SIZE"], init_params["NETWORK_SIZE"]), ylim=(-init_params["NETWORK_SIZE"], init_params["NETWORK_SIZE"]))
            Plots.scatter!([ac.x for ac in poss[1]], [ac.y for ac in poss[1]], c="blue")
            Plots.scatter!([ac.x for ac in poss[2]], [ac.y for ac in poss[2]], c="yellow")
            Plots.scatter!([ac.x for ac in poss[3]], [ac.y for ac in poss[3]], c="green")
            Plots.scatter!([ac.x for ac in poss[4]], [ac.y for ac in poss[4]], c="red")
            Plots.scatter!([ac.x for ac in poss[5]], [ac.y for ac in poss[5]], c="magenta")
            Plots.scatter!([ac.x for ac in poss[6]], [ac.y for ac in poss[6]], c="black")

            p2 = Plots.scatter(grid=false, showaxis=false, xlims=(0,0))
            Plots.scatter!([0], label="neurons", c="blue")
            Plots.scatter!([0], label="axon points", c="yellow")
            Plots.scatter!([0], label="dendrites", c="green")
            Plots.scatter!([0], label="synapses", c="red")
            Plots.scatter!([0], label="input nodes", c="magenta")
            Plots.scatter!([0], label="output nodes", c="black")

            Plots.display(Plots.plot(p1, p2, layout=l))
            # sleep(0.05)
        end
    end
end
