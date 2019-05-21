import Pkg

# used to install all neccesary packages

# adding required packages
Pkg.add("Distributions")
Pkg.add("Plots")
Pkg.add("Random")
Pkg.add("StatsBase")

# adding OpenAIGym
Pkg.add("PyCall")
withenv("PYTHON" => "") do
   Pkg.build("PyCall")
end
Pkg.add("https://github.com/JuliaML/OpenAIGym.jl.git")
