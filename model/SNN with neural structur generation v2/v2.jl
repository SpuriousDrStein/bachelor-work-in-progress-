using Flux
using OpenAIGym

env = GymEnv(:Carnival, :v0)

encoder = Chain(
    Conv((4, 4), 3=>11, relu, stride=(2,2)),
    Conv((4, 4), 11=>19, relu, stride=(2,2)),
    Conv((4, 4), 19=>27, relu, stride=(2,2)),
    Conv((4, 4), 27=>35, relu, stride=(2,2)),
    Conv((4, 4), 35=>41, relu, stride=(2,2)),
    Conv((4, 4), 41=>46, relu, stride=(2,2)))

decoder = Chain(
    Flux.ConvTranspose((5,5), 46=>41, relu),
    Flux.ConvTranspose((5,5), 41=>35, relu, stride=(1,1)),
    Flux.ConvTranspose((5,5), 35=>27, relu, stride=(2,2), pad=(-2,1)),
    Flux.ConvTranspose((5,5), 27=>19, relu, stride=(2,2), pad=(-2,1)),
    Flux.ConvTranspose((5,5), 19=>11, relu, stride=(2,2), pad=(-2,1)),
    Flux.ConvTranspose((6,6), 11=>3, relu, stride=(2,2), pad=(-2,1)))

reconst_loss(x, y) = Flux.mse(y, decoder(encoder(x)))
pars = Flux.params(encoder, decoder)

for te in 1:10
    reset!(env)

    for ei in 1:10
        r, s = step!(env, rand(env.actions))
        s = convert(Array{Int16}, s)
        s = cat(s, dims = ndims(s) + 1)

        data = [(s, s)]

        Flux.train!(reconst_loss, pars, data, Flux.ADAMW())
        if env.done
            break
        end
    end

    r, s = step!(env, rand(env.actions))
    s = Array(s)
    s = cat(s, dims = ndims(s) + 1)
    println("episode $te ::: reconst loss = ", reconst_loss(s, s))
end


using Images

r, s = step!(env, 1)
s = convert(Array{Float32}, s)

colorview(RGB, permutedims(s, (3, 2, 1)))


s_rec = decoder(encoder(s))

imshow(s_rec)
