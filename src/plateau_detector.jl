"""
    PlateauDetector(; patience=10, ϵ=1e-8,
                      scheduler=ParameterSchedulers.Stateful(Exp(0.001, 0.7))

A plateau detector for schedulers of learning rate. It will adjust the learning
rate when the loss has stopped improving for a number of ticks.

!!! tip
    Most commonly one tick is equal either to one epoch or one batch, but
    the user is free to define it as they see fit.

$(TYPEDFIELDS)
"""
@kwdef mutable struct PlateauDetector
    """
    The number of consecutive ticks with no improvement before advancing the
    learning rate along the scheduler.
    """
    patience::Int = 10

    """
    **Internal only.** The last tick number on which the learning rate was
    adjusted.
    """
    last_tick::Int = 0

    """
    **Internal only.** The best loss found so far.
    """
    best_loss::Float64 = Inf

    """
    **Internal only.** The tick number of the best loss found so far.
    """
    best_tick::Int = 0

    """
    The minimum learning rate. If the learning rate is reduced to a value
    smaller than this, it will be clipped at this value.
    """
    ϵ::Float64 = 1e-8

    """
    The schedule along which the learning rate will be adjusted.
    """
    scheduler::ParameterSchedulers.Stateful = ParameterSchedulers.Stateful(Exp(1e-3, 0.7))
end

"""
    learning_rate(pd::PlateauDetector)

Return the current learning rate of the plateau detector `pd`.
"""
learning_rate(pd::PlateauDetector) = max(pd.ϵ, pd.scheduler.schedule(pd.scheduler.state))

"""
    step!(pd::PlateauDetector, loss::Real)

Advance the plateau detector by one tick. Update `pd` internals with information
about the current loss. Return `true` if the learning rate should be updated.
"""
function step!(pd::PlateauDetector, loss::Real)
    current_tick = pd.last_tick + 1
    update_learning_rate = if loss < pd.best_loss
        pd.best_loss = loss
        pd.best_tick = current_tick
        false
    elseif current_tick - pd.best_tick >= pd.patience
        pd.best_tick = current_tick
        true
    else
        false
    end
    pd.last_tick = current_tick
    return update_learning_rate
end

"""
    Flux.Optimisers.adjust!(o, pd::PlateauDetector)

Update the learning rate of the optimizer `o` according to the schedule defined
by the plateau detector `pd`.

    Flux.Optimisers.adjust!(o, pd::PlateauDetector, i::Int)

Reset the learning rate by moving to state `i` along the schedule defined by
`pd.scheduler`.

    Flux.Optimisers.adjust!(o, pd::PlateauDetector, s::ParameterSchedulers.Stateful)

Reset the learning rate by changing the scheduler to `s` and moving to the state
defined by it.

!!! tip
    These last two three-parameter functions are sometimes convenient when we
    want to manually restart the learning rate schedule (for instance, after
    loading a checkpoint and restarting training).
"""
function Flux.Optimisers.adjust!(o, pd::PlateauDetector)
    old_lr = ParameterSchedulers.next!(pd.scheduler)
    lr = learning_rate(pd)
    if lr == old_lr
        return nothing
    end
    return Flux.Optimisers.adjust!(o, lr)
end

function Flux.Optimisers.adjust!(o, pd::PlateauDetector, i::Int)
    pd.scheduler.state = i
    return Flux.Optimisers.adjust!(o, learning_rate(pd))
end

function Flux.Optimisers.adjust!(o, pd::PlateauDetector, s::ParameterSchedulers.Stateful)
    pd.scheduler = s
    return Flux.Optimisers.adjust!(o, learning_rate(pd))
end