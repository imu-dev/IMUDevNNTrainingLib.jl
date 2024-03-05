using Pkg
Pkg.activate(joinpath(homedir(), ".julia", "dev", "IMUDevNNTrainingLib", "examples"))
using Revise
using IMUDevNNTrainingLib
using Lux
using Optimisers
using ParameterSchedulers
using ParameterSchedulers: Stateful
using Random

const NNTrLib = IMUDevNNTrainingLib

rng = Random.default_rng()
Random.seed!(rng, 0)

# Let's define a simple NN model
model = Dense(10, 1)

θ, Ω = Lux.setup(rng, model)

pd = PlateauDetector(; scheduler=Stateful(Exp(1e-3, 0.1)))
optimizer_state = Optimisers.setup(Adam(learning_rate(pd)), θ)

# let's define a sequence of artificial losses, which happen to increase
# (so that we will see the learning rate decrease)
losses = collect(1:100) .* 0.01

# let's step through the losses and update the learning rate when necessary
for loss in losses
    needs_update = NNTrLib.step!(pd, loss)
    if needs_update
        @info "Updating learning rate from $(learning_rate(pd))."
        Optimisers.adjust!(optimizer_state, pd)
        @info "Updated to $(learning_rate(pd)).\n\t----------------"
    end
end

function display_current_learning_rates(pd::PlateauDetector, o)
    intro = "Current learning rate stored in the"
    @info "$intro plateau detector: $(learning_rate(pd))"
    @info "$intro optimizer state: $(o.weight.rule.eta) (wieghts) and $(o.bias.rule.eta) (biases)."
end

display_current_learning_rates(pd, optimizer_state)

# let's reset the learning rate
Optimisers.adjust!(optimizer_state, pd, 1)
# and let's see what it is now
display_current_learning_rates(pd, optimizer_state)

# let's step through the losses again
for loss in losses
    needs_update = NNTrLib.step!(pd, loss)
    if needs_update
        @info "Updating learning rate from $(learning_rate(pd))."
        Optimisers.adjust!(optimizer_state, pd)
        @info "Updated to $(learning_rate(pd)).\n\t----------------"
    end
end

# and let's see the results
display_current_learning_rates(pd, optimizer_state)

# let's change the entire scheduler
Optimisers.adjust!(optimizer_state, pd, Stateful(CosAnneal(1e-3, 1e-4, 5)))
# and let's see what it is now
display_current_learning_rates(pd, optimizer_state)

# let's step through the losses again
for loss in losses
    needs_update = NNTrLib.step!(pd, loss)
    if needs_update
        @info "Updating learning rate from $(learning_rate(pd))."
        Optimisers.adjust!(optimizer_state, pd)
        @info "Updated to $(learning_rate(pd)).\n\t----------------"
    end
end

# and notice that we've cycled twice through the scheduler and we're back at the
# start again!