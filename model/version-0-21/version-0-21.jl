# version 2-_ are:
# exploring alternative solutions for threshold learning
# initial solutions for structural adaptation

include("structure.jl")
include("functions.jl")

N_id = 0
S_id = 0

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

unfold(t_nt)

N1 = unfold(n_dna, copy(N_id))
N2 = unfold(n_dna, copy(N_id)+1)


addDendrite!(N1, d_dna)
addAxonPoint!(N1, a_dna)
addDendrite!(N2, d_dna)
addAxonPoint!(N2, a_dna)
input_node = InputNode(Possition(-10,-10,-10), 0.)
out_node = OutputNode(Possition(10,10,10), 0.)


NN = Network([N1, N1.priors..., N1.posteriors..., N2, N2.priors..., N2.posteriors..., input_node, out_node])

propergate!(N1, accf_sum)
