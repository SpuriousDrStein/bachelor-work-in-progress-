# automatic differentiation etc

import Base: +, /, -, *, convert, promote_rule


function babylonian_sqrt(x; N=10)
    t = (1+x)/2
    for i = 2:N; t = (t + x/t)/2 end
    t
end

# test
babylonian_sqrt(5)
sqrt(5)

# automatic differentiation

struct D <: Number # a function derivative pair
    f::Tuple{Float64, Float64}
end

+(x::D, y::D) = D(x.f .+ y.f)
/(x::D, y::D) = D((x.f[1] / y.f[1], (y.f[1]*x.f[2] - x.f[1]*y.f[2])/y.f[1]^2))
-(x::D, y::D) = D(x.f .- y.f)
*(x::D, y::D) = D((x.f[1] * y.f[1]), (x.f[2]*y.f[1] + x.f[1]*y.f[2]))
convert(::Type{D}, x::Real) = D((x, zero(x)))
promote_rule(::Type{D}, ::Type{<:Number}) = D

# now the babylonian can be written as
x=49
babylonian_sqrt(D((x,1))), (√x, .5/√x)

# @code_native(babylonian_sqrt(D((2,1))))





# functionality outilne for oldschool nn

# errors
struct Dimention_Mismatch <: Exception end

# ground
function ⊚(W,B,a,∇a)
    w = deepcopy(W)
    b = deepcopy(B)

    act = (X) -> a.(w*X+b)
    err = (dpwdE) ->
    begin
        w -= ( dw/dpw ) * dpw/dE
    end
end


# activations
no_ac               = (X) -> X
σ                   = (X) -> 1/(1+e^-X)
∇σ                  = (X) -> (1/(1+e^-X)*(1-(1/(1+e^-X))))
relu                = (X) -> X>=0 ? X : 0
∇relu               = (X) -> X>=0 ? 1 : 0
tanh                = (X) -> tanh(X)
∇tanh               = (d) -> 1-tanh(d)^2
SoftPlus            = (X) -> log(e, 1+e^x)
∇SoftPlus           = (d) -> e^d/(1+e^d)
l_relu(u::Real)     = (X) -> X>0 ? X : u*X
∇l_relu(u::Real)    = (d) -> d>0 ? 1 : u
ELU(a::Real)        = (X) -> X>0 ? X : a*(e^X-1)
∇ELU(a::Real)       = (d) -> d>0 ? 1 : a*(e^d-1)+a


# structure components
DENSE(w, b, act) =
begin
    nlin = ⊚(W,B,a)
    ∇nlin = ∇(W,B,a)

    act = (X) -> nlin(X)
    err = (E) -> ∇nlin(E)
    return Tuple(act, err)
end

RNN(xh_w, xh_b, act_1, hh_w, hh_b, act_2, hy_w, hy_b, act_3) =
begin
    xh = ⊚(xh_w, xh_b, act_1)
    hh = ⊚(hh_w, hh_b, act_2)
    hy = ⊚(hy_w, hy_b, act_3)

    act = (X) -> nlin(X)
    err = (E) -> ∇nlin(E)
    return Tuple(act, err)
end

CONV(K_w, K_b, stride) =
begin
    kf = ⊚(K_w, K_bX)

    act = (X) ->
    begin
        if length(size(K_w)) != length(size(X)) throw(Dimention_Mismatch) end
        sx, sy = size(X)[1], size(X)[2]

        y = AbstractArray
        for i ∈ [1:stride:sx-size(K_w)[1]]; j ∈ [1:stride:sy-size(K_w)[2]]
            y[i,j] = kf([i:i+size(K_w)[1],j:j+size(K_w)[2]])
        end
        y
    end

    err = (E) -> ∇nlin(E)
    return Tuple(act, err)
end

GLU(W,b,a) = # gated linear unit
begin
    nlin_1 = ⊚(W,b,a)
    nlin_2 = ⊚(W,b,no_ac)
    ∇nlin_1 = ∇(W,b,a)
    ∇nlin_2 = ∇(W,b,no_ac)

    act = (X) -> nlin_1(X)*nlin_2(X)
    err = (E) -> nlin_1(E)*∇nlin_1(E)
    return Tuple(act, err)
end


# utility collections

MODELTABLE = Dict()
MODELTABLE[:DENSE] = DENSE
MODELTABLE[:RNN] = RNN
MODELTABLE[:CONV] = CONV
MODELTABLE[:GLU] = GLU





# assembly
#=

init_graph(init_layers, init_levels) =
begin
     LayersToLevels = [((i,j),(x)->x) for i∈[1:it_layers], j∈[1:it_levels]]
     create_layer()
end

fuse(level1, level2, method) = method(@fetch(level1), @fetch(level2))


function runback(cs, f::Function, computed_nodes::Dict) # all input nodes should be in the computde_nodes in the beginning
    for priorNode in cs[1]
        qer = get(computed_nodes, priorNode, false)

        if qer == false
            runback(priorNode, f(cs[2][2]), computed_nodes)
            push!(computde_nodes, (qer[2][1] => qer))
        else
            f(qer[2][2])
        end
        end
end
function assemble_c_graph(output_nodes::Node)
    for n in output_nodes
        runback(n)
    end
end



end

=#







#=
::: STRUCTURE :::

    == protocoll level ==
    LID = Tuple{UInt8}                 | Local ID
    LT = UInt8                              | local time
    TF = Function                           | thread function

    == base level ==
    ⟷ = Array{LID, LID, Function}               | from, to, Activation_Function
    ∘ = Array{LID, Function, Array{Array}}      | Local ID, function, parameter
    ⊡ = Pair{Array{∘}, Array{⟷}}
    ≗ = Pair{LID, ⊡}                            | for cell level
    ⊞ = Function                                | produced from ⊡

    == cell level ==

MULT(a...) =



⊡.build():
    ⊞ = x -> x

    for ⟷ in ⊡[2]
        ∘1 = ⊡.where(x->x[1] == ⟷[1])
        ∘2 = ⊡.where(x->x[1] == ⟷[2])
        if ∘1 || ∘2 null throw error

        ⊞ = x -> ⊞(∘1[2](x, ∘1[3]...))

        if ⟷[3] not NaN
            ⊞ = x -> ⟷[3](⊞(x)) end
    end
    return ⊞, d⊞/dE
end


TIMECELL <: ⊡
    [1]
        ∘(1, xh, xh_b, rescale)
        ∘(2, hh, hh_b, dense)
        ∘(3, hy, hy_b, rescale)
    [2]
        ⟷(1,2,σ)
        ⟷(2,3,relu)
        ⟷(3,1,relu)

LSTMCELL <: ⊡
    [1]
        ∘(1, concat, xh, xh_b)
        TF.split(LID=2)
        ≗(3, TIMECELL)
        ∘(4, mm)
    [2]
        ⟷(1,2)
        ⟷(2,3)
        ⟷(2,4, tanh)
        ⟷(2,4, σ)



    functions:
        dense       (X, w, b){inputsize}
        rescale     (X, w, b){inputsize, outputsize}
        dropout     (X, rate, attentionmask=NaN){}
        convolve    (X, Kw, Kb){inputsize, Ksize, stride}

E2E:
    conv(input, func) : Kw, Kb
    RNN(input, func) : xh, xh_b, hh, hh_b, hy, hy_b


LSTM(intut) : E2E[]


=#


#=
what i need:

Iterators.drop(generator, n)
collect(generator)
Iterators.take(generator, n)
zip(generator...)

                       D
     _____________________________________ time
    |
    |
    |
    | o<>
 N  |oo<>
    | o<>
    |
    |
    |
    |
    |
    |
Input space                         output space
Node = Tuple{Array,Function,Bool=true}








x= Node(Array(rand(10)),+)


function seek(X::Array, node::Node)
    if node.done
        return node.func!(X) end
    if length(node.former)<=0
        return node.func!(X)
        node.done = true
    end
    for n in node.former
        seek(X, n)
    end
end

function transmit(X::Array, node::Node)



end



#collect(Iterators.flatten((1:10,3:4,10:11)))

f = (x for x in 1:10)

fd = zip(f,f,f,f)
collect(fd)
Iterators.rest(f, "hello")



collect(f)


function simple_n(input_::Wb_pair)
    W = input_[1]
    b = input_[2]
    @generated function internal(o)
        return :(o*W.+b)
    end
    return internal(input_)
end

Wb_1 = Wb_pair([5],[10])
simple_n(Wb_1)
=#



# neural soup


#import CSV
#dat = CSV.read("data/arabic_alphabet/TrainImages.csv")


X=rand(2,2,2)
W1=rand(2,2,2)
b1=rand(2,2,2)
a1 = GLU(W1, b1, l_relu)


println(a(X))

# println(size(permutedims(X, [2,1,3])))
# eachindex(X)



#=
differentiation between the:
- one function generator
- multithreaded dispatch

the scales

1. neuron to neuron in one structure    -> one function generator (internal_function)
2. sections reacting on stimuli         -> @spawn(internal_function)


    there is a step in physical development of a brain in which the majority of the neurons are still basic units

    each n-th timestep yields a certain connection that get more expressed as time goes on


    if i want to match the development of the brain onto a recursive model:

        steps to take at each global timestep:

            the brain has to grow
            the brain computes his p(current_state | last_state)
                this entails:
                    fire forward signals to indicate decisions
                    fire backward signals to indicate reward

        i need basic structure:

            structure I:

                nodes have:
                    accumulation of volt over time (by activations of local@t-1)
                    natural decrease of volt over time



=#


# some approach

mutable struct w_b{dim}
    w::Array{AbstractFloat, dim}
    b::Array{AbstractFloat, 1}
end

# types of linear relations are:
# (NxM) -> lin -> (NxG)

abstract type net end
abstract type IO_net <: net end
abstract type IIO_net <: net end
abstract type IOO_net <: net end
abstract type IIOO_net <: net end


a = w_b{3}

function linear(w,b)
    W = w
    B = b
    (x)->schedule(Task(x.*w.+b))

w = rand(100,10)
b = rand(1,10)
f = lin(w,b)

# end 1



module tcwpa

x = Condition()

function give(name)
    wait(x)
    for a in 1:3
        println(string(name," running: ", a))
        sleep(3)
    end
end

a1, a2, a3 = [Task(()->give(i)) for i in 1:3]

schedule(a1)
schedule(a2)
schedule(a3)
notify(x)


function softmax(collection::Array)
    exp.(collection)/sum(exp.(collection))
end


function attention_aggregate3D(w, b, soft_w, soft_b, patch::Array)
    sftm = softmax(patch .* soft_w .+ soft_b)
    println(sum(sftm))
    (patch .* w .+ b).*sftm
end


function conv3D(x_size, kernel, stride)
    if any([k%2==0.0 for k in size(kernel)[1:2]])
        throw(string("kernel size must be odd so that left and right from the pixel is the same distance. kernel size: ",size(kernel)))
    end

    k_size = collect(floor.(Int32, [((size(kernel)[1:2].-1)./2)...,x_size[3]]))

    x_size += [[%(a, stride)-size(kernel)[i] for (i,a) in enumerate(x_size[1:2])]...,0]

    y = fill(1.0, floor.(Int32, Tuple([(a-size(kernel)[i])/stride for (i,a) in enumerate(x_size)])))
    println("expect: ", x_size)

    w = soft_w = rand(size(kernel)...)
    b = soft_b = rand(size(kernel)[1:2]...)

    @task (x) ->
    begin
        for (i,ii) in enumerate(k_size[1]+1:stride:x_size[1]-k_size[1]-1)
            for (j,jj) in enumerate(k_size[2]+1:stride:x_size[2]-k_size[2]-1)
                z = x[ii-k_size[1]:ii+k_size[1],jj-k_size[2]:jj+k_size[2], :]
                println(size(z), size(w), size(b))
                y[i,j] = attention_aggregate3D(w, b, soft_w, soft_b, z)
            end
        end
        return y
    end
end

c1 = conv3D([128,128,3], rand(5,5,3), 6); schedule(c1);


point = Tuple{Int32, Int32, Int32}

struct Neuron
    con::Tuple{point, point}
    w::Float32
    b::Float32
end

struct Layer_Hull
    shape::Tuple{Int32, Int32}
    depth::Int64
end

mutable struct Layer
    shape::Array{Int32, 2}
    depth::Int64
    neurons::Array{Neuron, 2} # replace with >> ::Union{Array{nothing,0}, Array{Neuron, 2}}
end

function neural_connection(from::point, to::point; W=rand(), b=rand())::Neuron
    return Neuron((from, to), W, b)
end

function establish_layer_connection(input_layer::Layer_Hull, output_layer::Layer_Hull)::Layer
    out_L = Layer(output_layer[1], output_layer[2], )
    for o_i ∈ axes(output_layer[1], 1), o_j ∈ axes(output_layer[1], 2)
        for i_i ∈ axes(input_layer[1], 1), i_j ∈ axes(input_layer[1], 2)
            out_L[3][o_i, o_j] = neural_connection((input_layer[2], i_i, i_j), (output_layer[2], o_i, o_j))
        end
    end
    out_L
end


lh1 = Layer_Hull((3,3),1)
establish_layer_connection(, Layer_Hull(collect([5,5]),2))



# thesis
# a brain optimizes itself towards evolutionary efficiency in a given context

# each neurons deffining properties are
# possition as (x,y,z)
# connections to consecutive neurons as [(x,y,z)]
#   (temporal)
# internal voltage          -> V
# internal resistance       -> IR
#   (hp)
# given potential           -> P
# maximum resistance        -> MAXR
# activation threshhold     -> AT
# rate of voltage decay     -> RoVD
# rate of resistance decay  -> RoRD


# ( as the voltage level (given by prior neurons) increases, the IR allso increases to a given maximum: MAXR )
# ( the delay for the IR to regulate after an activation is given by RoRD )
# ( it decays to some minimum resistance, i want to keep the same over the entire network )

#=

further thoughts:

parvalbumin type inhibitory neurons can be described as:
"neurons with (med to low P + low AT + high RoRD) that stimulate a given subsection of neurons,
providing them with a 'platou' from wich activation is easyer"

'disinhibitory' neurons can be described as:
"neurons with either (high P + high AT + low RoRD) or (low P + low AT + high RoRD)
that provide a negative value to the neurons they are connected to making it harder for them to open"

the brains time function seems to be given by current
the voltage denotes a neurons readyness at any particular time
yet the reaching of an activation threshhold only influences the voltage and does not induce a current.

with the last remark thefollowing question rises:
what happens to the current that is not transmitted or stil waits for the activation of the neuron ?
(because it seems that it is only described as the voltage for a given neuron)

high voltage produces high current
current should never be < 1 because then the current regulates the voltage of a given node
it seems as if a fiering neuron first imposes a difference in potential to a given consecutive neuron
if this neuron the fiers at well it generates a resistance.

all laws apply: V (of a neuron) = R (internal resistance given state at t-1) x I (given by hyper param)

the function of the internal resistance:
    IR = somefunc(V, t, RoRD) [ where t is some factor describing the state of the current iteration over all neurons ]

the function of the internal voltage:
    V = somefunc(prior, t, RoVD)


if i fill the fabric between neurons with lightly transmitting material
(i.e. neurons hold a 'lightweight' referenc to the dendrieds of some prior)
that can give a neuron some idea about unestablished priors, whos connection might hold a possitive net value.

=#



end


#=

( a neurons internal state is given by: Volt, Time

    there are two times:
        CT = computational time = as the time a neuron/layer/region takes to compute
        ST = structural time = as the point in time at witch a specivic component should be activated

    { given 3 Regions A,B and C

        CT:
            A = 30ms, 42ms, 10ms
            B = 15ms, 10ms, 12ms
            C = 50ms, 11ms, 32ms

        ST:
            A = 1, 3
            B = 1, 2
            C = 3, 6

        So the order of the system function is C(B(C(A(B(A(x))))))


        start jobs A(X)
    }

    thoughts:

        1. each neuron has a listener that is not susceptible to simultanious inputs at CT (i.e. a queue)
        2. since the forward passing function is assembled backwards, CT is ambiguous for individual layers
            thus making it either complicated or expensive to sync the network on any given complexity.

    answers:

        (2): the CT-avg(CT) for a given node/layer/region could be propergated together with the signal at t
                and at a given complexity level the network just waits that ammount befor continuing
)

P = Pair{Int32, Int32}
N = Union{P, Function}
C = Pair{Point, Point}
L = Array{N, 2}
R = Pair{Array{L}, Array{C}}

=#

module principal_components
    xyz(x,y,z) = x, y, z

    pkg(W,b) = deepcopy(W), deepcopy(b)

    lin(pkg) = (x)->pkg[1]*x+pkg[2]

    c_node(W, b, xyz) = begin
        if typeof(W) != typeof(b)
            throw("no mathing type for W and b")
        end

        Wb = pkg(W, b)

        (lin(Wb), xyz, Wb)
    end

    d_node(xyz) = begin
        ((x)->x, xyz, prior)
    end

    rnn(W, b, xyz) = begin
        if length(W) != 3 || length(b) != 3 || length(xyz) != 3
            throw("no matching size [required 3]")
        end

        n1 = c_node(W[1], b[1], xyz[1])
        n2 = c_node(W[2], b[2], xyz[2])
        n3 = c_node(W[3], b[3], xyz[3])

        Dict(   :f=>(x)->n3[1](n2[1](n1[1](x))),
                :n=>[n1, n2, n3])
    end
end

# rnn1 = rnn([2,3,4], [5,6,7], [1,1,1])
# rnn2 = rnn([20,30,40], [50,60,70], [1,2,1])
# rnn3 = rnn([200,300,400], [500,600,700], [1,3,1])


# x=3.3
# println(@time rnn1[:f](x))
# println(@time rnn2[:f](x))
# println(@time rnn3[:f](x))
