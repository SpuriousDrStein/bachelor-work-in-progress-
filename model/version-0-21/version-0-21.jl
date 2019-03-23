# version 2-_ are:
# exploring alternative solutions for threshold learning
# initial solutions for structural adaptation

include("structure.jl")
include("functions.jl")

# TESTING
pos1 = get_random_init_possition(FloatN(0.), FloatN(5.))
length1 = m_v_pair(40, 20)
dec1 = m_v_pair(0.5,0.5)
thr1 = m_v_pair(1,0.5)
life1 = min_max_pair(200, 2000)
num_pri_post = min_max_pair(1, 5)

t_nt = NeuroTransmitter(1)
a_dna = AxonPointDNA(pos1, length1, life1)
d_dna = DendriteDNA(pos1, length1, life1)
s_dna = SynapsDNA(dec1, thr1, life1)
n_dna = NeuronDNA(pos1, life1, num_pri_post, num_pri_post)

N1 = unfold(n_dna, t_nt, FloatN(1.)); println(N1.priors)

addDendrite!(N1, d_dna)

for d in eachindex(N1.priors) # SynapsDNA, Possition, NeuroTransmitter, life_decay::FloatN
    N1.priors[d] = AllCell(unfold(s_dna, N1.priors[d].cell.possition, t_nt, 0))
end

N1.priors[1].cell


propergate!(N1, accf_sum)
