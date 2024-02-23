"""
A plateau detector for learning rate schedulers.
It will adjust the learning rate when the loss has stopped improving for a
number of ticks.

!!! tip
    Most commonly one tick is equal either to one epoch or one batch, but
    the user is free to define it as they see fit.
"""
@kwdef mutable struct PlateauDetector
    """
    The number of consecutive ticks with no improvement before reducing the
    learning rate.
    """
    patience::Int = 10
    """The factor by which the learning rate will be reduced."""
    factor::Float64 = 0.1
    """
    Internal only. The last tick number that the learning rate was adjusted.
    """
    last_tick::Int = 0
    """
    Internal only. The best loss found so far.
    """
    best_loss::Float32 = Inf32
    """
    Internal only. The tick number of the best loss found so far.
    """
    best_tick::Int = 0
    """
    The minimum learning rate. If the learning rate is reduced to a value
    smaller than this, it will not be reduced further.
    """
    ϵ::Float64 = 1e-8
    """
    The current learning rate.
    """
    η::Float64
end

"""
Advance the plateau detector by one tick.
"""
function step!(pd::PlateauDetector, loss::Float32)
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

function update!(o, pd::PlateauDetector)
    if pd.η <= pd.ϵ
        return nothing
    end
    pd.η = max(pd.η * pd.factor, pd.ϵ)
    return Flux.Optimisers.adjust!(o, pd.η)
end