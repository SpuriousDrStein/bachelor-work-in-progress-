using LinearAlgebra

p_norm(A::AbstractArray, p) = begin
    sum(abs.(A).^p)^(1/p)
end

p_norm([1,2,3,4,5,6,7], 2)

softmax(A::AbstractArray) = [exp(a)/sum(exp.(A)) for a in A]
softmax(A::AbstractArray, ind::Integer) = exp(A[ind])/sum(exp.(A))
sigmoid(x; base=ℯ) = 1/(1+ℯ^-x)

function d_softmax(A::AbstractArray)
    out = zeros(length(A), length(A))
    for i in eachindex(A)
        for j in eachindex(A)
            out[i, j] = softmax(A, i) * ((i == j) - softmax(A, j))
        end
    end
    out
    #[out[i,i] for i in 1:length(A)] # the diagonal values
end


function d_softmax(A::AbstractArray, ind::Integer)
    out = zeros(length(A))
    for i in eachindex(A)
        out[i] = softmax(A, i) * ((i == ind) - softmax(A, ind))
    end
end


d_sigmoid(x; base=ℯ) = sigmoid(x, base=base)*(1-sigmoid(x, base=base))
