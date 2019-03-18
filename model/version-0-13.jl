# a version similar to 0-12
# but the derivative update is done on static weights for
# the length of the buffer
# i.e. only every iterations % buffer_length == 0

# further research could be done into the descision which derivatives to take depending on the sub_iteration_fitness
# i.e. the fitness of one static weight configuration

# hidden connections are still updated every step

# loop looks like:
#   for each iteration
#       pred = forward(NN, X)
#       loss = error_function(pred, y)
#
#       append!(loss_buffer, loss)
#
#       temporal_backward_update!(NN, BN)
#
#       if iteration % buffer_length == 0
#
#           derivs = get_derivs(NN, BN, loss_buffer)        # produces the derivatives for each iteraiton
#           fittest_deriv = assess(derivs)
#
#           spatial_backward_update!(NN, fittest_deriv)
#





# remember @inbound for performance improvement
