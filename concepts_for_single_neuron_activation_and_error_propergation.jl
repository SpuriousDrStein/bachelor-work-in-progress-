#=
VERSION I
    nuclei layers = 1D
    axon/den layers = 2D
    no axon charge disserpation
    sockets = non dendrite non axon non synaps possitions
    information can only be passed to sucessive layers
    connective layers are reduced to their numeric representation given by preceding NL
    numeric connective layers are squared matricies
    inhibition dendrites can only manifest themselfs in CL
    this example tests the SIMO version of inhibition neurons (single inputs multiple outputs) =-               # to mitigate confusion it is to note that multiple out means outputs of inhibiting dendrites for cl@current_layer
=#

struct Axon
    length::Int16
end

struct Dendrite
    possition::Tuple{Int16, Int16}
end

# struct Synaps
#     possition::Tuple{Int16, Int16}
#     d_nucleus::Nucleus
#     a_nucleus::Nucleus
# end

struct Nucleus
    # variables
    possition::Int16
    Q::Float16
    resistance::Float16

    # temporal update
    Q_decay::Float16
    resistance_decay::Float16

    # activation
    threshold::Float16
    setback_resistance::Float16

    # structur
    axon::Axon
    dens::Array{Dendrite}
end


struct InhNucleus
    # variables
    possition::Int16
    Q::Float16
    resistance::Float16

    # temporal update
    Q_decay::Float16
    resistance_decay::Float16

    # activation
    threshold::Float16
    setback_resistance::Float16

    # structur
    axon::Axon                  # axon in num_CL@layer+1 i.e. in the same B_BLOCK index
    dens::Array{Dendrite}       # dens in num_CL@Layer i.e. in the same F_BLOCK index
end


#CL = Array{Union{nothing, Axon, Dendrite, Synaps}, 2}
NL = Array{Union{nothing, Nucleus, InhNucleus}, 1}
num_CL = Array{Float16, 2}

F_BLOCK = Array{Tuple{num_CL, N}}
B_BLOCK = Array{Tuple{NL, num_CL}}


# idea is to have step return a layer change matrix LCM that can be added onto the existing num_CL
# num_CL_block[current_layer+1] .+= step(num_CL_block[current_layer], NL_block[current_layer+1])
# (sidenote: in case of the input layer the first argument of step would be the receptor field)
# every layer other than the first will now be changed depending on the inhibition values given by NL_block[current_layer+1].InhibNeurons.dends for each InhibNeuron

# MAIN LOOP STRUCTURE FOR ONE (2 CL 2 NL) BLOCK
function forward_pass(block::block)
    cl = block[1][1]                        # get 2D connective layer
    nl = block[1][2]                        # get 1D neural layer

    change_field = INPUT                    # a matrix to capture changes for one external timestep (i.e. input_matrix for first layer and change_by_activation for each hiddel layer)
    cl = change_field                       # because its the input
    inhibit!(cl, nl)                        # inhibit cl based on nl_inhibition_neurons
    increment!(nl, cl)                      # increment nl_activation_neurons based on cl

    change_field = activate!(nl)            # activate neurons where threshold is reached and return change_field for axon activations based on their lengths for cl+1

    for d in block[2:end-1]
        cl = d[1]
        nl = d[2]

        cl += change_field                  #  add activation values from preceding nl
        inhibit!(cl, nl)
        increment!(nl, cl)

        change_field = activate!(nl)
    end


    cl = d[1]
    nl = d[2]

    cl += change_field                  #  add activation values from preceding nl
    inhibit!(cl, nl)
    increment!(nl, cl)

    return activate!(nl)[:,1]           # return 1. row of final activation

    # at this point there could be a loop back matrix that
    # maps a point in the returned field of change to:
    #   (1) - an evaluation metric
    #   (2) - an eqivalent point in the same matrix that is associated with an Inhibition Nucleuses input-'Axon'
end


function increment!(nl::NL, cl::num_CL)
    for n in nl
        if isActNuc(n)
            nl[n.dens]
            n.Q = sum([cl[d.possition...] for d in n.dens]) * n.resistance
        else
    end
end

function inhibit!(cl::num_CL, nl::NL)
    for n in nl
        if isInhNuc(n)
            n.Q = sum([cl[d.possition...] for d in n.dens]) * n.resistance
        else
    end
end

function activate!(nl::NL)
    return_matrix = zeros(length(nl), length(nl))
    for n in nl
        if typeof(n) == Nucleus
            if test(n)                                                        # test charge for reaching the treshold
                # capture activation change for axons reaching into cl+1
                for a in 1:n.axon.length
                    return_matrix[n.possition, a] += n.Q
                end
                # reset cell and increase resistance
                n.Q = 0
                n.resistance = n.setback_resistance
            end
        end
    end
    return_matrix                                                             # return changes in num_CL+1
end


# smaller imutable functions
function test(n::Nucleus)
    n.Q >= n.threshold
end
function isActNuc(n)
    typeof(n) == Nucleus
end
function isInhNuc(n)
    typeof(n) == InhNucleus
end
