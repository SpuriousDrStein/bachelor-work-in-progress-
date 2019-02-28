
#=
function activational_Neuron(lowest_V, lowest_R, lowsest_I, V_threashhold, activation_strength, influenced_by)
    R = lowest_R
    V = lowest_V
    I = lowest_I
    # all three should not be 0

    function update()
        # t_V = R*I
        # t_I = V/R
        t_R = V/I

        update V -> i/o activate
        i -> update R (large) so that ΔV/Δt inhibits any imediately conseccutive current
        o -> update R (small) so that ΔV/Δt stays positive

        # prior neurons and voltage decay change voltage
        V = f1(t_V, influenced_by.V)

        # higher current -> higher resistance
        R = V/I

        # activate changes current
        I = V < V_threashhold ? t_I : activation_strength
    end
end
=#

# t1 = @task neuron1(x1)

# this version of multithreading entails that
# the x at a given time t (structuraly)
# is not fixed for t
# i.e.
# if somewhere parallel to neuron1 (lets say neuron2)
# the same signal (lets say from neuron0) is processed
# neuron1 & neuron2 both task the activation given x since they are in the same layer

# a neurological model that could work here:
# 1. neuron0 fires
# 2. neuron1 and 2 are both only infering the state of neuron0 not the actual output of the function that propergates x
#

# actually: in big O not. :
# how expensive is the state update for the entire network given:
# nn = number of neurons
# it = the average number of inferences to determin the state of one neuron

# O(nn*it)

# this notation denotes the update rate (ct) of the network
# the input rate (st) is proportional to the update rate
# since there has to be at least one update per change of input voltage

# comming back to the multithreading aproach:
# async queue: [update(set_of_neuron) for each region]
#


#=

PROPERGATION OPTIONS

    a signal sent into the netowk seems to be not sufficient if it is just a current or just a potential.
    whatever makes, for example photoreceptors fire, makes them fire a current

    1. Propergating states

        neuron at (xyz) in Layer (L) has to know what state its axons have
        so its axon can infere its state

        I: BACKWARD SPANNING RECURSIVE TREE:

            axon        = ( voltage_at_segment_0 )
            neuron      = ( NeuroL, xyz, dendrites )
            dendrite    = ( DenAxonL, axon, possition_on_axon)

            connection = ( DenAxonL, axon, dendrite )

    2. Propergating signals

        neuron at (xyz) in Layer (L) has build up enough electric charge to release through its axon.
        this is obviously different than (n.axon) infering its state from neuron and just seeing a even larger potential than in t-1.
        i.e. the state inference only describes the components polarity with respect to its prior (axon from neuron, neuron from dendrites, dendrite from axon)
        that means if neuron (n) fiers => n.axon.state becomes more negative but now there is allso a current running thorugh the axon instead of just potential.


        I:
            n.activation_potential = constant

            while in STATE UPDATE:

                foreach L in NueroL[1:end]
                    foreach (x, y) in (1:L.x), (1:L.y)
                        foreach n in L[x,y] n != nothing
                            d_ds = sum([d.voltage for d in neuron.dendrites]) - ns
                            ns = ( ns + d_ds ) -> clamp( 0, n.activation_potential )

                    if ns >= activation_potential
                        n.activate = true

            wile in PROPERGATION UPDATE:

                foreach (x, y) in (1:L.x), (1:L.y)
                    foreach n in L[x,y] n != nothing
                        of n.active

=#


#=
10/17/18

the potential accross the brain is negligible in a simulation that contains only the data processing relevant parts of a NN
the nucleus stores charge
this charge is released through the axon where the current ( load ) of the charge at each dendrite depends on the number of dendirtes connected to it

the state of a nucleus ->           [ Q = CV ]
                                    so as the voltage in a nucleus increases ( given that its dendirtes sence something )
                                    its charge increases liniarly.
                                    if the charge reaches a threashold it fires a puls thorugh its axon.
                                    neurons connected to this axon have a given charge Q relative to Q/C = V = RI
                                    so as I spikes because of the axon, V spikes and subsequently Q spikes.
                                    but thus uses allmost all charges in the nucleus and thus drops the voltage afterwards given by [ V = Q/C ]
                                    C = hyperparam constant given to each neuron

 [sidenote] use Float16 to be more efficient since precision does not matter as much as the systems grow larger

the state of a axon ->

=#

# 3D & 2D FUNCTIONS
function flash(xyz::VID_3D, Q)
    # give dendrites at xyz if any:
    #   Q as value for the structural timestep
end

function flash(xyz::VID_2D)

end


# 3D FUNCTIONS

function activate_3D(neuron::ActNeuron_3D)
    neuron.Q = neuron.minQ
end

function neuron_step_3D(neuron::ActNeuron_3D)
    Res = neuron.priorQ * normalized(neuron.Q) # maybe take Q maybe take normalized Q

    for d in neuron.dendrites
        dQdt = d.Q * d.inhin_dendrite * neuron.Res
        d.Q = 0
    end
    neuron.Q += dQdt

    neuron.priorQ = (neuron.gamma) * neuron.priorQ + (1-neuron.gamma) * dQdt
end

function axon_step_3D(axon::Axon)
    if axon.neuron.Q >= axon.neuron.ActThr
        activate(neuron)

        for i, xyz in enumerate(axon.valid_possitions) # expect to be in order (nearest - farthest)
            flash(xyz, (axon.neuron.Q - axon.neuron.minQ) * axon.insulation)
        end
    end
end


# 3D STRUCTURE

mutable struct Axon
    insulation::Float16 # from 0.000001 to 0.999999
    neuron::ActNeuron_3D
    valid_possitions::AbstractSet{VID_3D}
end

mutable struct VID_3D # vector id
    x::Int32
    y::Int32
    z::Int32
end

mutable struct ActNeurons_3D
    # deffinitive
    VEC::VID_3D          # possition inside layer
    ActThr::Float16      # activation threshold determins: potential needed and potential send
    MinQ::Float16        # charge value to fall back on after activation or decay
    gamma::Float16       # (0.999 - 0.001) where: 0.999 -> priorQ = mostly priorQ & 0.001 -> priorQ = mostly dQdt

    # variational
    Res::Float16         # resistance value characterosed by: dQ/dt * Q       ->      and describing: if total charge high Resistance high; if total charge changes much Res allso becones high High
    Q::Float16
    priorQ::Float16      # some prior charge determined by dQ/dt
end

struct InhDendrite_3D
    VEC::Vid_3D


struct InhNeuron_3D
    VEC::Vid_3D



struct NeuronLayer_3D
    VEC::VID_3D
    range::Tuple{Int16, Int16, Int16} # allso gives number of possible neurons
    ActNeurons::AbstractSet{ActNeuron_3D}
    InhNeurons::AbstractSet{InhNeuron_3D}
end

struct ConnectionLayer_3D
    VEC::VID_3D
    range::Tuple{Int16, Int16, Int16} # allso gives number of possible neurons
    ActNeurons::AbstractSet{ActNeuron_3D}
    InhNeurons::AbstractSet{InhNeuron_3D}
end

# 2D STRUCTURE
struct VID_2D
    x::Int32
    y::Int32
end

mutable struct dendrite2D
    pos:possition2D
end

mutable struct axon2D

end


mutable struct neuron2D
    d::AbstractSet{dendrite2D}
    N::nucleus2D
    A::axon2D
end
