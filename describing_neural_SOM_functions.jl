#=

Network:
    NT = network time   -   one NT might occure for each n'th ET = environment time


Network parameters:
    adhesion_distance : for open-sending and open-recieving connections forming a synaps
    neuron_distribution : electrical to chemical neuron ratio
        - also describes distribution over selected neuron subtypes that have to/ will be present in the network (tournament selection for neuron classes)
    input/output_node_possitions

Axon parameters:
    axon_strength : depending on activations alowing for less and less resistance
        - influences axon_growth
    axon_sending_point_density : number of points on axon that can transmit a signal

Neuron parameters:
    max_dendrites
    max_dendrite_distance
    connections_directly_on_body
    NT_type (in the case of electric neurons this is none)
        - strength of expression
        - difusion range

=#


#=

There will be some groupse of established points:
    synapses
    open-reciever
    open-senders
    chem-neurons
        (subgroups)


one collective function could be:
    the distance update:
        mean([os.activated * os.possition * os.strength for os in open_senders])
            - where strength is the relative strength of the os wrt. os.axons.strength
            - and activated is true/false if it has been activated at NT


=#

#=

some general rules:

    1. open-sending connections when fiering, pull open-recieving connections towards them
    2. open-sending and open-recieveing connections that come close to each other bind to form a synaps
    3. synapse strength degrades over time and is renewed more by larger currents passing through it
    4. when a synaps degrades, the axon persists and the dendrite is removed
    5. dendrites share current in an axon (like electro dinamic)
    6. NTs, for chem-neurons either replenish, difuse or disapear
        - replenish is guided by the NT_effected variable


optimization parameters:
    runtime:
        axon:
            max( number of dendrites connected )
        neuron:
            max( connected_dendrites / max_dendrites )
    genom:


=#


#=

alpha_cell:
    charge
    threshold
    pre_connections
    post_connections

    ->
    input_cell:
        <charge
        <post_connection

    output_cell:
        <charge
        <threshold
        <pre_connections
        ~action_function

    el_neuron_cell:
        <charge
        <threshold
        <pre_connections
        <post_connections

    chem_neuron_cell:
        <charge
        <threshold
        <pre_connections
        <post_connections
        ~NT
        ~NT_effected


=#


#=

describing NT

=#
