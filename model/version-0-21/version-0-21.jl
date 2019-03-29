# version 2-_ are:
# exploring alternative solutions for threshold learning
# initial solutions for structural adaptation

include("structure.jl")
include("functions.jl")

# HPs
N_id = 0
S_id = 0
min_fuse_distance = FloatN(0.1)
init_life_decay = 1



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
NN_dna = NetworkDNA(m_v_pair(200., 10), net_min_max..., m1, m1)

N1 = unfold(n_dna, copy(N_id))
N2 = unfold(n_dna, copy(N_id)+1)


addDendrite!(N1, d_dna)
addAxonPoint!(N1, a_dna)
addDendrite!(N2, d_dna)
addAxonPoint!(N2, a_dna)
input_node = InputNode(Possition(-10,-10,-10), 0.)
out_node = OutputNode(Possition(10,10,10), 0.)

network_components = [N1, N1.priors..., N1.posteriors..., N2, N2.priors..., N2.posteriors..., input_node, out_node]

NN = unfold(NN_dna, min_fuse_distance, init_life_decay, network_components)

den_sinks, ap_sinks = value_step!(NN, [1.])

state_step!(NN, den_sinks, ap_sinks)
println([d.possition for d in get_dendrites(get_all_all_cells(NN))])

println(get_dendrites(get_all_all_cells(NN)))

import Plots
all_all_cells = get_all_all_cells(NN)
all_neurons = get_all_neurons(NN)
Plots.plot([])
