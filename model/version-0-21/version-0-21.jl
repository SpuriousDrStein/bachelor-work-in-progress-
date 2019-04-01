# version 2

# specifications:
# 1. dendrite and axonPoint max is bound by the network size / 2


include("structure.jl")
include("functions.jl")

# HPs
min_fuse_distance = FloatN(0.1)
init_life_decay = FloatN(1.)
dendrite_instantiation_threshold = 3
axon_point_instantiation_threshold = 4
fitness_decay = 0.7
sample_array_length = 4
init_num_neurons = 10
init_max_priors = 3
init_max_posteriors = 3

# TESTING
pos1 = get_random_init_sub_possition(Possition(0,0,0), 4, 3)
length1 = min_max_pair(5, 60)
m0_5 = min_max_pair(0.1, 0.999)
m1 = min_max_pair(0.5,1.5)
life1 = min_max_pair(200, 2000)
num_pri_post = min_max_pair(1, 5)

t_nt = NeuroTransmitterDNA(m1, pos1, m1 ,m0_5)
a_dna = AxonPointDNA(length1, life1, pos1)
d_dna = DendriteDNA(length1, life1, pos1)
s_dna = SynapsDNA(m0_5, m0_5, life1, t_nt)
n_dna = NeuronDNA(pos1, num_pri_post, num_pri_post, life1)
net_min_max = [min_max_pair(3000,5000), min_max_pair(2000,3000), min_max_pair(1000,1500),min_max_pair(1000,1500)]
dna_stack = DNAStack([t_nt], [a_dna], [d_dna], [s_dna], [n_dna])
NN_dna = NetworkDNA(min_max_pair(100, 200), net_min_max..., m1, m1)

NN = unfold(NN_dna, min_fuse_distance, init_life_decay, dna_stack, [])
populate_network!(NN, init_num_neurons, init_max_priors, init_max_posteriors)
input_node = AllCell(InputNode(Possition(-5,-5,-5), 0.))
out_node = AllCell(OutputNode(Possition(5,5,5), 0.))
append!(NN.components, [input_node, out_node])


println(["$(s)\n" for s in get_synapses(get_all_all_cells(NN))]...)
println(["$s\n" for s in get_dendrites(get_all_all_cells(NN))]...)
println(["$s\n" for s in get_axon_points(get_all_all_cells(NN))]...)
println(["$s\n" for s in get_input_nodes(get_all_all_cells(NN))]...)
println(["$s\n" for s in get_output_nodes(get_all_all_cells(NN))]...)



for i in 1:100
    den_sinks, ap_sinks = value_step!(NN, [1.])
    state_step!(NN, den_sinks, ap_sinks)
    clean_network_components!(NN)
end







import Plots
all_all_cells = get_all_all_cells(NN)
all_neurons = get_all_neurons(NN)
Plots.plot([])
