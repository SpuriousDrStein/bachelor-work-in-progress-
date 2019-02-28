#include("network_functions.jl")
include("parallel_network_functions.jl")
include("loss_functions.jl")
include("activation_functions.jl")

using Distributions
using MLDatasets

# TESTS

# HP's
T = 10
input_size = 28*28
out_size = 10
iterations = 1
logits = 10

# Data
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()
oh_train_y = zeros(length(train_y), logits)
for r in axes(train_y, 1)
    oh_train_y[r, train_y[r]+1] = 1
end


# network
l1 = create_dense_layer(input_size, 13)
l2 = create_dense_layer(13, 12)
l3 = create_dense_layer(12, 11)
l4 = create_dense_layer(11, 10)

for i in 1:iterations
    ind = rand(1:length(train_y))
    X = reshape(train_x[:, :, ind] * 1, (28*28))
    y = oh_train_y[ind, :]

    frq_update!(l1, X, T) # returns membranes
    frq_update!(l2, l1.act_frequencies, T)
    frq_update!(l3, l2.act_frequencies, T)
    frq_update!(l4, l3.act_frequencies, T)
    println("iteration  : $i")

    prediction = softmax(l4.act_frequencies)
    error = crossentropy(prediction, y)
    println("prediction : $prediction")
    println("label      : $y")
    println("error      : $error")

    d_ce = d_crossentropy(prediction, y)
    d_soft = d_softmax(l4.act_frequencies)
    d_frq_update!(l4, l3.act_frequencies, T)

    # println(l1.act_frequencies, l1.membranes)
    # println(l2.act_frequencies, l2.membranes)
    # println(l3.act_frequencies, l3.membranes)
    # println(l4.act_frequencies, l4.membranes)
end



#@code_llvm loss(X, y)
#@code_typed loss(X, y)

# this is true for values in range(0,1) ???
# test1(x) = -log((ℯ^x[1])/sum(ℯ.^x))
# test2(x) = log(sum(ℯ.^x)) - x[1]
