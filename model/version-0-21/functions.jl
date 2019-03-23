include("utility_functions.jl")
include("network_operation_functions.jl")
include("dna_generator_functions.jl")

# BASE OVERLOAD
import Base.+, Base.-, Base./
import Base.convert
function +(poss::Possition, f::Force); [poss.x, poss.y, poss.z] .+ ([f.x, f.y, f.z] .* f.strength); end
function +(p1::Possition, p2::Possition); [p1.x, p1.y, p1.z] .+ [p2.x, p2.y, p2.z]; end
function -(poss::Possition, f::Force); [poss.x, poss.y, poss.z] .- ([f.x, f.y, f.z] .* f.strength); end
function /(poss::Possition, n::Number); [poss.x, poss.y, poss.z] ./ n; end
function mean(p1::Possition, p2::Possition); (p1 + p2) / 2; end


# CONTROL FUNCTIONS
function addDendrite!(N::Neuron, denDNA::DendriteDNA)
    if any(ismissing.(N.priors))
        for i in eachindex(N.priors)
            if ismissing(N.priors[i])
                N.priors[i] = AllCell(unfold(denDNA))
                return nothing
            end
        end
    end
    println("no den added to neuron: $(N.id)")
    return nothing
end
function addAxonPoint!(N::Neuron, dna::AxonPointDNA)
    # add axon point and add N.NT reference to it
end



function rectifyInRangeInitialization!(pos::Possition, network_range::FloatN, min_distance_between_components::FloatN)
end
