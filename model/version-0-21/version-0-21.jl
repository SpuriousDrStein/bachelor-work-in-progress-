# version 2

# specifications:
# 1. dendrite and axonPoint max is bound by the network size / 2


include("structure.jl")
include("functions.jl")

# TESTING
pos1 = get_random_init_possition(100)
length1 = min_max_pair(5, 60)
m0_5 = min_max_pair(0.1, 0.999)
m1 = min_max_pair(0.5,1.5)
life1 = min_max_pair(100, 200)
num_pri_post = min_max_pair(1, 5)

t_nt = NeuroTransmitterDNA(m1, pos1, m1 ,m0_5)
a_dna = AxonPointDNA(length1, life1, pos1)
d_dna = DendriteDNA(length1, life1, pos1)
s_dna = SynapsDNA(m0_5, m0_5, life1, t_nt)
n_dna = NeuronDNA(pos1, num_pri_post, num_pri_post, life1)
dna_stack = DNAStack([t_nt], [a_dna], [d_dna], [s_dna], [n_dna])

NN_dna = NetworkDNA(min_max_pair(100, 200),
                    min_max_pair(3000, 5000),
                    min_max_pair(2000,3000),
                    min_max_pair(2000,3000),
                    min_max_pair(2000,3000),
                    m1, m1)
rectifyDNA!(NN_dna)

NN = unfold(NN_dna, FloatN(0.5), FloatN(0.99), FloatN(2), FloatN(10), dna_stack)

rectifyDNA!(NN.dna_stack, NN)
populate_network!(NN, 10, 3, 3)
input_node = AllCell(InputNode(Possition(-5,-5,-5), 0.))
out_node = AllCell(OutputNode(Possition(5,5,5), 0.))
append!(NN.IO_components, [input_node, out_node])


println(["$(s)\n" for s in get_synapses(get_all_all_cells(NN))]...)
println(["$s\n" for s in get_dendrites(get_all_all_cells(NN))]...)
println(["$s\n" for s in get_axon_points(get_all_all_cells(NN))]...)

function test(NN)
    for n in get_all_neurons(NN)
        in_cells = get_input_nodes(get_prior_all_cells(n))
        if in_cells != []
            println(n)
            return true
        end
    end
end

for i in 1:1000
    den_sinks, ap_sinks = value_step!(NN, [1.])
    state_step!(NN, den_sinks, ap_sinks)

    # println("num_components = ", length(NN.components))
    # println("num_dens       = ", length(get_dendrites(get_all_all_cells(NN))))
    # println("num_aps        = ", length(get_axon_points(get_all_all_cells(NN))))
    # println("num_synapses   = ", length(get_synapses(get_all_all_cells(NN))))
    # println("num_neurons   = ", length(get_all_neurons(NN)))

    test(NN)
    # println(["$(s.Q) :: " for s in get_synapses(get_all_all_cells(NN))])
    # println([distance(get_input_nodes(NN)[1].possition, d.possition) for d in get_dendrites(get_all_all_cells(NN))])
    # println("----------")

    # println(NN.components[1])
    clean_network_components!(NN)

    runtime_instantiate_components!(NN, i)
end









import Plots
all_all_cells = get_all_all_cells(NN)
all_neurons = get_all_neurons(NN)
Plots.plot([])
