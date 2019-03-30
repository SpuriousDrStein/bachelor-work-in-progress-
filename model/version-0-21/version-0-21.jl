# version 2-_ are:
# exploring alternative solutions for threshold learning
# initial solutions for structural adaptation

include("structure.jl")
include("functions.jl")

# HPs
min_fuse_distance = FloatN(0.1)
init_life_decay = 1
dendrite_instantiation_threshold = 3
axon_point_instantiation_threshold = 4
fitness_decay = 0.7
sample_array_length = 4


# TESTING
pos1 = get_random_init_sub_possition(Possition(0,0,0), FloatN(10.), FloatN(2.))
length1 = m_v_pair(40, 20)
m0_5 = m_v_pair(0.5,0.5)
m1 = m_v_pair(1,0.5)
life1 = min_max_pair(200, 2000)
num_pri_post = min_max_pair(1, 5)

t_nt = NeuroTransmitterDNA(m1, pos1, m1 ,m0_5)
a_dna = AxonPointDNA(length1, life1, pos1)
d_dna = DendriteDNA(length1, life1, pos1)
s_dna = SynapsDNA(m0_5, m0_5, life1, t_nt)
n_dna = NeuronDNA(pos1, num_pri_post, num_pri_post, life1)
net_min_max = [min_max_pair(3000,5000), min_max_pair(2000,3000), min_max_pair(1000,1500),min_max_pair(1000,1500)]
dna_stack = DNAStack([t_nt], [a_dna], [d_dna], [s_dna], [n_dna])
NN_dna = NetworkDNA(m_v_pair(200., 10), net_min_max..., m1, m1)

NN = unfold(NN_dna, min_fuse_distance, init_life_decay, dna_stack, network_components)



N1 = unfold(n_dna, copy(N_id))
N2 = unfold(n_dna, copy(N_id)+1)
add_dendrite!(N1, d_dna)
add_axon_point!(N1, a_dna)
add_dendrite!(N2, d_dna)
add_axon_point!(N2, a_dna)
input_node = AllCell(InputNode(Possition(-5,-5,-5), 0.))
out_node = AllCell(OutputNode(Possition(5,5,5), 0.))
network_components = [N1, N1.priors..., N1.posteriors..., N2, N2.priors..., N2.posteriors..., input_node, out_node]

append!(NN.components, network_components)


for i in 1:100
    den_sinks, ap_sinks = value_step!(NN, [1.])
    state_step!(NN, den_sinks, ap_sinks)
    println([d.possition for d in get_dendrites(get_all_all_cells(NN))])
end

println(get_dendrites(get_all_all_cells(NN)))

import Plots
all_all_cells = get_all_all_cells(NN)
all_neurons = get_all_neurons(NN)
Plots.plot([])
