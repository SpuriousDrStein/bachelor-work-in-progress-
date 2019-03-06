
entropy(p::Number; base=ℯ) = -(p * log(base, p))
entropy(p::AbstractArray; base=ℯ) = -sum(p .* log.(base, p))

crossentropy(p::Number, l::Number; base=ℯ) = -(l * log(base, p))
crossentropy(p::AbstractArray, l::AbstractArray; base=ℯ) = -sum(l .* log.(base, p))

KL_divergence(p::Number, l::Number; base=ℯ) = -(p * log(base, l/p))
KL_divergence(p::AbstractArray, l::AbstractArray; base=ℯ) = -sum(p .* log.(base, l./p))

SD(x::Number, y::Number) = (y - x)^2
SD(x::AbstractArray, y::AbstractArray) = sum(y .- x).^2


# DERIVATIVE FUNCTIONS
d_log(x; base=ℯ) = 1/(x*log(base))

d_entropy(p::Number; base=ℯ) = -p * d_log(p, base=base) - d_log(p, base=base)
d_entropy(p::AbstractArray; base=ℯ) = -sum(p .* d_log.(p, base=base) .+ log.(base, p))

d_crossentropy(p::Number, q::Number; base=ℯ) = -q * d_log(p, base=base) - log(base, p)
d_crossentropy(p::AbstractArray, q::AbstractArray; base=ℯ) = -(q .* d_log.(p, base=base) .* log.(base, p))

d_KL_divergence(p::Number, q::Number; base=ℯ) = -p * d_log(base, q/p) * ((p-q)/(p^2)) - log(base, l/p)
d_KL_divergence(p::AbstractArray, q::AbstractArray; base=ℯ) = -sum(-p .* d_log.(base, q./p) .* ((p.-l)./(p.^2)) .+ log.(base, q./p))

d_SD(x::Number, y::Number; base=ℯ) = -2 * (y - x)
d_SD(x::AbstractArray, y::AbstractArray; base=ℯ) = sum(-2 .* (y .- x))



# 
# Zygote.@grad entropy(p, base=ℯ) = entropy(p, base=base), del -> (del * d_entropy(p, base=base),)
# Zygote.@grad crossentropy(p, l, base=ℯ) = crossentropy(p, l, base=base), del -> (del * d_crossentropy(p, l, base=base),)
# Zygote.@grad KL_divergence(p, l, base=ℯ) = KL_divergence(p, l, base=base), del -> (del * d_KL_divergence(p, l, base=base),)
# Zygote.@grad SD(x, y) = SD(x, y), del -> (del * d_SD(x, y),)
