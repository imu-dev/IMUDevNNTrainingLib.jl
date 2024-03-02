"""
    Checkpointer(;
        dir::String = ".",
        continue_from::Union{Int,Symbol} = :last,
        save_every::Int = 1)

A struct to manage checkpoints during training of neural networks.

$(TYPEDFIELDS)

# Examples

## Checkpointing during training
```julia
# a function that updates the training log
model = ...
update_log!(...) = ...
ch = Checkpointer(dir=joinpath(homedir(), "checkpoints"),
                  continue_from=:last,
                  save_every=5)
model, opt_state, log, start_epoch, other = start!(ch, model)
for epoch in start_epoch:100
    Flux.train!(model, data, opt_state)
    update_log!(log, model, data, epoch)
    checkpoint(ch, epoch; model, opt_state, log)
end
```

## Loading checkpoint for testing
```julia
model = Chain(...)
ch = Checkpointer(dir=joinpath(homedir(), "checkpoints"))
load!(model, path_to_checkpoint(ch, index_of_last_checkpoint(ch)))
test(model, data)
```

"""
@kwdef struct Checkpointer
    """Root directory where the checkpoints will be saved to"""
    dir::String = "."
    """
    A default behaviour for resuming checkpointing. See [`start!`](@ref) for
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
    start!(cp::Checkpointer, model;
           move_to_device=Flux.cpu,
           fallback_optimizer=m -> Flux.setup(Flux.Adam(1e-3), m),
           continue_from=cp.continue_from)

Start the checkpointing process. `continue_from` can be one of the following:
- `:none`, `:start`, `:restart`, `:nothing`: for starting from scratch;
- `:last`, `:latest`, `:recent`: for continuing from the last checkpoint;
- `Int`: for continuing from a specific checkpoint index.

The `move_to_device` function is used to move the model to the desired device
(e.g. CPU or GPU). The `fallback_optimizer` function is used to setup the
optimizer if the checkpoint does not exist. The function returns the model,
the optimizer state, the training log, the index of the subsequent training
epoch and the dictionary of other objects that were saved in the checkpoint.
"""
function start!(cp::Checkpointer, model;
                move_to_device=Flux.cpu,
                fallback_optimizer=m -> Flux.setup(Flux.Adam(1e-3), m),
                continue_from=cp.continue_from)
    chkp, start_epoch = if continue_from in [:none, :start, :restart, :nothing]
        @info "Starting new checkpointing"
        nothing, 1
    elseif continue_from in [:last, :latest, :recent]
        @info "Continuing from the last checkpoint"
        c, i = isdir(cp.dir) ? last_checkpoint_next_epoch(cp) : (nothing, 1)
        if i == 1
            @info "No checkpoint found. Starting from scratch."
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
        m = move_to_device(model)
        return m, move_to_device(fallback_optimizer(m)), [], start_epoch, Dict()
    end
    model, opt_state, log, other = load!(model, chkp)
    return move_to_device(model), move_to_device(opt_state), log, start_epoch, other
end

"""
    load!(model, checkpoint_path)

Load the model, the optimizer statem, the training log and the other variables
from the checkpoint file.
"""
function load!(model, checkpoint_path)
    model_state = JLD2.load(checkpoint_path, "model_state")
    Flux.loadmodel!(model, model_state)

    opt_state = JLD2.load(checkpoint_path, "opt_state")
    log = JLD2.load(checkpoint_path, "log")
    other = JLD2.load(checkpoint_path, "other")
    return model, opt_state, log, other
end

"""
    checkpoint(cp::Checkpointer, epoch::Int; model, opt_state, log, kwargs...)

Save the model, the optimizer state and the training log to a checkpoint file if
the given epoch is a multiple of `cp.save_every`. Otherwise, do nothing.
"""
function checkpoint(cp::Checkpointer, epoch::Int; model, opt_state, log, kwargs...)
    if epoch % cp.save_every != 0
        return nothing
    end
    mkpath(cp.dir)
    @info "Checkpointing epoch $epoch..."
    jldsave(path_to_checkpoint(cp, epoch);
            model_state=Flux.state(Flux.cpu(model)),
            opt_state=Flux.cpu(opt_state),
            log,
            other=Dict(kwargs))
    @info "Done"
    return nothing
end
