using Distributions

include("utility_functions.jl")
include("network_operation_functions.jl")
include("evolutionary_network_generation.jl")


# BASE OVERLOAD
import Base.+, Base.-, Base./, Base.^, Base.*
function +(p1::Position, p2::Position); Position(([p1.x, p1.y, p1.z] .+ [p2.x, p2.y, p2.z])...); end
function /(poss::Position, n::Number); Position([poss.x, poss.y, poss.z] ./ n); end
function ^(p::Position, pow::Integer); [p.x,p.y,p.z] .^ pow; end
function vec_mean(p1::Position, p2::Position); Position((p1 + p2) ./ 2.); end
function *(p::Position, n::Number); Position(([p.x, p.y, p.z] .* n)...); end
