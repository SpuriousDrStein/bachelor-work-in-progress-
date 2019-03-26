using Distributions

include("utility_functions.jl")
include("network_operation_functions.jl")


# BASE OVERLOAD
import Base.+, Base.-, Base./
import Base.convert
function +(poss::Possition, f::Force); [poss.x, poss.y, poss.z] .+ ([f.x, f.y, f.z] .* f.strength); end
function +(p1::Possition, p2::Possition); [p1.x, p1.y, p1.z] .+ [p2.x, p2.y, p2.z]; end
function -(poss::Possition, f::Force); [poss.x, poss.y, poss.z] .- ([f.x, f.y, f.z] .* f.strength); end
function /(poss::Possition, n::Number); [poss.x, poss.y, poss.z] ./ n; end
function mean(p1::Possition, p2::Possition); (p1 + p2) ./ 2.; end
