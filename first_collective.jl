
mutable struct wb_pair
    w::AbstractArray{Float32}
    b::AbstractArray{Float32}
end

mutable struct act_wb_pair
    wb::wb_pair
    act::Function
    act_del::Function
end

function lin(a_wb::act_wb_pair)
    (x) -> a_wb.act.(a_wb.wb.w * x .+ a_wb.wb.b)
end

function relu(x)
    if x>0
        x
    else 0
    end
end

function del_relu(x)
    if x>0
        1
    else 0
    end
end

function del_act_lin(a_wb::act_wb_pair)
    w_del(x, dE) = dE * a_wb.act_del(lin(a_wb)) * x
    b_del(x, dE) = dE * a_wb.act_del(lin(a_wb))
    return w_del, b_del
end

function square_dist(pred::AbstractArray, label::AbstractArray)
    return (pred-label).^2
end

function del_square_dist(pred::AbstractArray, label::AbstractArray)
    return (pred-label) .* 2
end

function MSE(pred::AbstractArray, label::AbstractArray) # has to have batch for dim 1
    return (1/size(pred, 1)) * sum((pred-label).^2, dims=1)
end

function del_MSE(pred::AbstractArray, label::AbstractArray)
    return (2/size(pred, 1)) * sum(pred-label, dims=1)
end


# test  =
#       network
wbp = wb_pair(rand(1, 20), rand(1))
awbp = act_wb_pair(wbp, relu, del_relu)


#       fake data
X = rand(20, 3)
y = rand(1,3)

#       NETWORK
l1 = lin(awbp)
y_hat = l1(X)
loss = square_dist(y_hat, y)

#       'BACKPROP'
dEdloss = del_square_dist(y_hat, y)
dlossdl1_w, dlossdl1_b = del_act_lin(awbp)
dlossdl1_w(X, dEdloss)


function FC(wbs::wb_pair...)
    ins = ()
    #
    # b_f = (dEdOut)->begin
    #     dEdWblast = dEdOut
    #     for i in 1:length(wbs)
    #         dEdWblast =  dEdWblast * # dif(wbs[-i])
    #         wbs[-i].w -= dEdWblast * dWblastdW # usually input @ wbs[-1]
    #         wbs[-i].b -= dEdWblast * dWblastdB # usually 1
    #     end
    #     clear(ins)
    # end

    f_f = (x)->lin(x, wbs[1]);
    for i in 2:length(wbs)
        tmp = f_f
        f_f = (x)->lin(tmp(x), wbs[i]);
    end
    f_f
end

x = rand(5,3)
wb1 = wb_pair(rand(4,5), rand(4))
wb2 = wb_pair(rand(3,4), rand(3))
wb3 = wb_pair(rand(1,3), rand(1))


NN = FC(wb1, wb2, wb3)
