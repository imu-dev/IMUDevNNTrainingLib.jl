"""
    Checkpointer(;
        dir::String = ".",
        continue_from::Union{Int,Symbol} = :last,
        save_every::Int = 1)

A struct to manage checkpoints during training of neural networks.

!!! note
    The checkpointer is intended to save:
    
    - the model parameters
    - the model state
    - the optimizer state
    - the training log
    - other objects passed by the user

    Note in particular that it is not recommended to save the `model` itself.

$(TYPEDFIELDS)

# Examples

## Checkpointing during training
```julia
using Lux, Optimisers
using Random

rng = Random.default_rng()
Random.seed!(rng, 0)

# a function that updates the training log
model = ...
update_log!(...) = ...
ch = Checkpointer(dir=joinpath(homedir(), "checkpoints"),
                  continue_from=:last,
                  save_every=5)
chkp_data, start_epoch = start(ch)
parameters, states, opt_state, log, other = chkp_data

for epoch in start_epoch:100
    train!(parameters, states, model, data, opt_state)
    update_log!(log, model, data, epoch)
    checkpoint(ch, epoch; parameters, states, opt_state, log, other)
end
```

## Loading checkpoint for testing
```julia
model = Chain(...)
ch = Checkpointer(dir=joinpath(homedir(), "checkpoints"))
chkp_data = load_checkpoint(model, path_to_last_checkpoint(ch))
test(model, chkp_data, data)
```

"""
@kwdef struct Checkpointer
    """Root directory where the checkpoints will be saved to"""
    dir::String = "."
    """
    A default behaviour for resuming checkpointing. See [`start`](@ref) for
    details.
    """
    continue_from::Union{Int,Symbol} = :last
    """
    The frequency of saving checkpoints. A checkpoint will be saved every
    `save_every` epochs.
    """
    save_every::Int = 1
end

"""
    path_to_checkpoint(cp::Checkpointer, epoch::Int)

Return the path to the checkpoint file for the given epoch.
    
!!! warning
    This function does not check if the file exists.
"""
function path_to_checkpoint(cp::Checkpointer, epoch::Int)
    return joinpath(cp.dir, "checkpoint=$epoch.jld2")
end

"""
    path_to_last_checkpoint(cp::Checkpointer)

Return the path to the most recent checkpoint file for the given epoch.
"""
function path_to_last_checkpoint(cp::Checkpointer)
    i = index_of_last_checkpoint(cp)
    isnothing(i) && return nothing
    return path_to_checkpoint(cp, i)
end

"""
    list_existing_checkpoint_indices(cp::Checkpointer)

List the epoch indices of the existing checkpoints in the directory `cp.dir`.
"""
function list_existing_checkpoint_indices(cp::Checkpointer)
    reg = r"^checkpoint=(\d+).jld2$"
    checkpoints = filter(f -> occursin(reg, f), readdir(cp.dir))
    if isempty(checkpoints)
        return Int[]
    end
    indices = map(ch -> first(match(reg, ch)), checkpoints)
    return parse.(Int, indices)
end

"""
    index_of_last_checkpoint(cp::Checkpointer)

Return the index of the most recent checkpoint in the directory `cp.dir`. Return
`nothing` if no checkpoint exists.
"""
function index_of_last_checkpoint(cp::Checkpointer)
    indices = list_existing_checkpoint_indices(cp)
    isempty(indices) && return nothing
    return maximum(indices)
end

"""
    index_of_last_checkpoint_prior_to(cp::Checkpointer, i::Int)

Return the index of the most recent checkpoint in the directory `cp.dir` out of
checkpoints that are prior to the given index `i`. Return `nothing` if no such
checkpoint exists.
"""
function index_of_last_checkpoint_prior_to(cp::Checkpointer, i)
    indices = list_existing_checkpoint_indices(cp)
    indices = indices[indices .< i]
    isempty(indices) && return nothing
    return maximum(indices)
end

"""
    last_checkpoint_next_epoch(cp::Checkpointer)

Return the path to the most recent checkpoint in the directory `cp.dir` together
with the index of the subsequent epoch. Return `nothing` if no checkpoint exists.
"""
function last_checkpoint_next_epoch(cp::Checkpointer)
    i = index_of_last_checkpoint(cp)
    isnothing(i) && return nothing, 1
    return path_to_checkpoint(cp, i), i + 1
end

"""
    ith_checkpoint_next_epoch(cp::Checkpointer, i::Int)

Return the path to the checkpoint with index `i` in the directory `cp.dir`
together with the index of the subsequent epoch. If the checkpoint with index
`i` does not exist, find the last checkpoint prior to index `i` and return
the path to it together with the index of the subsequent epoch. Return `nothing`
(and index 1) if no checkpoint exists.
"""
function ith_checkpoint_next_epoch(cp::Checkpointer, i)
    file = path_to_checkpoint(cp, i)
    isfile(file) && return file, i + 1
    @warn "Checkpoint with index $i does not exist. Trying to find the last checkpoint prior to index $i."

    j = index_of_last_checkpoint_prior_to(cp, i)
    if isnothing(j)
        @warn "No checkpoint found prior to index $i."
        return nothing, 1
    end
    return path_to_checkpoint(cp, j), j + 1
end

"""
    start(cp::Checkpointer;
          move_to_device=Lux.cpu_device(),
          continue_from=cp.continue_from)

Start the checkpointing process. `continue_from` can be one of the following:
- `:none`, `:start`, `:restart`, `:nothing`: for starting from scratch;
- `:last`, `:latest`, `:recent`: for continuing from the last checkpoint;
- `Int`: for continuing from a specific checkpoint index.

The `move_to_device` function is used to move the model parameters, model states
and optimizer's state to the desired device (e.g. CPU or GPU). The function
returns a `NamedTuple` with the checkpointed data (or `nothing` if no checkpoint
is found) and an index of the subsequent training epoch.
"""
function start(cp::Checkpointer;
               move_to_device=Lux.cpu_device(),
               continue_from=cp.continue_from)
    chkp, start_epoch = if continue_from in [:none, :start, :restart, :nothing]
        @info "Starting new checkpointing"
        nothing, 1
    elseif continue_from in [:last, :latest, :recent]
        @info "Attempting to continue from the last checkpoint"
        c, i = isdir(cp.dir) ? last_checkpoint_next_epoch(cp) : (nothing, 1)
        if i == 1
            @info "No checkpoint found. Starting from scratch."
        else
            @info "Starting from checkpoint with index $(i-1)."
        end
        c, i
    elseif continue_from isa Int
        @info "Attempting to continue from checkpoint with index $continue_from"
        c, i = ith_checkpoint_next_epoch(cp, continue_from)
        if i == 1
            @info "No checkpoint found. Starting from scratch."
        elseif i == continue_from + 1
            @info "Checkpoint with index $continue_from found. Continuing..."
        else
            @info "Checkpoint with index $continue_from not found. " *
                  "Starting from checkpoint with index $(i-1) instead..."
        end
        c, i
    else
        throw("Unknown continue_from value: $continue_from")
    end

    if isnothing(chkp)
        return nothing, start_epoch
    end
    return load_checkpoint(chkp; move_to_device), start_epoch
end

"""
    load_checkpoint(path::String; move_to_device=Lux.cpu_device())

Load the model parameters, model states, optimizer state, training log and the
remaining variables from the checkpoint file. `move_to_device` is used to move
the:
- model parameters
- model states and
- optimizer's state
to the desired device (e.g. CPU or GPU).
"""
function load_checkpoint(path::String; move_to_device=Lux.cpu_device())
    ps = JLD2.load(path, "model_parameters")
    st = JLD2.load(path, "model_states")
    opt_state = JLD2.load(path, "opt_state")
    log = JLD2.load(path, "log")
    other = JLD2.load(path, "other")
    return (; parameters=move_to_device(ps),
            states=move_to_device(st),
            opt_state=move_to_device(opt_state),
            log,
            other)
end

"""
    checkpoint(cp::Checkpointer, epoch::Int; model, opt_state, log, kwargs...)

Save the model, the optimizer state and the training log to a checkpoint file if
the given epoch is a multiple of `cp.save_every`. Otherwise, do nothing.
"""
function checkpoint(cp::Checkpointer, epoch::Int;
                    parameters, states, opt_state, log, kwargs...)
    if epoch % cp.save_every != 0
        return nothing
    end
    mkpath(cp.dir)
    @info "Checkpointing epoch $epoch..."

    # The data must always be moved to a cpu for saving
    dev = Lux.cpu_device()

    jldsave(path_to_checkpoint(cp, epoch),
            true;
            model_parameters=dev(parameters),
            model_states=dev(states),
            opt_state=dev(opt_state),
            log,
            other=Dict(kwargs))
    @info "Done"
    return nothing
end
