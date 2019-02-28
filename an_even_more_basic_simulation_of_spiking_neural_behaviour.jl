import CSV
import GZip

# data
data = CSV.read("C:\\Users\\BBM2H16AHO\\Documents\\_DATA\\mnist\\mnist_train.csv")
pictures = convert(Array, data[:, 2:end])
labels = convert(Array, data[:, 1])



mutable struct ActNeuron
    Q::Float16          # 0 <= Q    < actQ
    R::Float16          # 0 <  R    < maxR
    maxR::Float16       # 0 <  maxR < 1
    actQ::Float16       # 0 <  actQ
    RDecay::Float16     # 0 <  RDecay < 1
    QDecay::Float16     # 0 <  QDecay < 1

    filter::AbstractArray{Bool}
end
mutable struct InhNeuron
    Qdischarge::Float16 # 0 <  R    < 1
    Qdecay::Float16     # 0 <  RDecay < 1
end


function step!(ns::Array{Tuple{ActNeuron, InhNeuron}}, data)
    out = []
    for (i, (an, in)) in enumerate(ns)
        an.Q += sum(data .* an.filter) * (1-an.R) * in.Qdischarge
        if an.Q >= an.actQ
            append!(out, an.Q)
            an.Q = 0
            an.R = an.maxR
        else
            append!(out, 0)
            an.R -= an.RDecay
            an.Q -= an.QDecay
            in.Qdischarge += in.Qdecay
            if an.R < 0; an.R = 0; end
            if an.Q < 0; an.Q = 0; end
            if in.Qdischarge > 1; in.Qdischarge = 1; end
        end
    end
    out
end

function instantiate_layer(layer_size::Integer, mask::AbstractArray, actNeuronParams::Tuple, inhNeuronParams::Tuple)
    [(ActNeuron(actNeuronParams..., mask), InhNeuron(inhNeuronParams...)) for i in 1:layer_size]
end

function instantiate_network(layer_sizes::AbstractArray, input_size, actNeuronParams::Tuple, inhNeuronParams::Tuple)
    network = []
    mask = rand(Bool, input_size)
    push!(network, Array(instantiate_layer(layer_sizes[1], mask, actNeuronParams, inhNeuronParams)))
    if length(layer_sizes) >= 2
        for i in 2:length(layer_sizes)
            mask = rand(Bool, layer_sizes[i-1])
            push!(network, Array(instantiate_layer(layer_sizes[i], mask, actNeuronParams, inhNeuronParams)))
        end
    end
    network
end


function instantiate_masks(layer_sizes::AbstractArray)
    network = []
    for ls in layer_sizes
        push!(network, rand(Bool, ls))
    end
    network
end


# network
AlphaNeuronAParams = (0, 0.1, 0.999, 1, 0.05, 0.3)
AlphaNeuronIParmas = (1, 0.1)

# model params
input_size = length(pictures[1,:])
layer_sizes = [4,6,5,3,10]

# train params
train_iters = 1
sample_size = 100

GEN1_NETWORK = instantiate_network(layer_sizes, input_size, AlphaNeuronAParams, AlphaNeuronIParmas)


# loop
for i in 1:train_iters
    rnd_ind = rand(axes(pictures, 1))
    x, y = pictures[rnd_ind, :], labels[rnd_ind]

    println("iteration $i --------")
    tmpdata = x
    for j in eachindex(GEN1_NETWORK)
        tmpdata = step!(GEN1_NETWORK[j], tmpdata)
        if j != 1
            println(GEN1_NETWORK[j][1][1].Q)
            println(GEN1_NETWORK[j][1][1].R)
        end
    end

    println(tmpdata)
end
