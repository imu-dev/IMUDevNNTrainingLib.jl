# Checkpointing

`Flux` already provides some methods for checkpointing; however, because of its agnostic nature with respect to the data-saving backend, the naming conventions and the choice of objects that need saving, it calls for writing quite an extensive boilerplate code. `IMUDevNNTrainingLib` makes all those choices for the user and in return streamlines the checkpointing process. In particular, the following choices are made:

- [`JLD2`](https://github.com/JuliaIO/JLD2.jl) is used as a backend for saving the state of the trained Neural Net
- A checkpoint comprises of saved:
  - `model_state`: parameters of the trained Neural Network
  - `opt_state`: parameters of the optimizer
  - `log`: a training log (with loss functions)
  - `other`: a dictionary of any other parameters that the user wishes to save. For instance, `plateau_detector` may be saved here.
- Names of checkpoints take the format `checkpoint=$(epoch).jld2`, where `$(epoch)` is the index of the epoch at the end of which the checkpoint has been made.

The main object is `Checkpointer`:
```@docs
Checkpointer
```

The usual training workflow involves `start!`ing it (which will load the specified checkpoint) as well as `checkpoint`ing it on every epoch, which will save the model when appropriate:

```@docs
start!
checkpoint
```

!!! note
    Sometimes the user may wish to save and, perhaps, update additional variables. For instance a common use case would be to save the state of the [`PlateauDetector`](@ref). In that case the following could be done:
    ```julia
    model = ...
    ch = Checkpointer()
    model, opt_state, log, start_epoch, other = start!(ch, model)
    pd = other[:plateau_detector]
    ```
    and if we'd like to start using a different schedule of learning rates we could adjust them with:
    ```julia
    new_schedule = ParameterSchedulers.Stateful(Exp(1e-3, 0.15))
    Flux.adjust!(opt_state, pd, new_schedule)
    ```
    before we resume the training.

For testing it is often enough to simply `load!` the model and skip all the other objects:

```@docs
IMUDevNNTrainingLib.load!
```

To pick the appropriate checkpoint one can make use of the following helper functions:

```@docs
path_to_checkpoint
index_of_last_checkpoint
IMUDevNNTrainingLib.index_of_last_checkpoint_prior_to
```

A typical example of loading for testing is given below:

```julia
# define model architecture
model = ...

chkp = Checkpointer(; dir="saved_checkpoints")
path = path_to_checkpoint(chkp, index_of_last_checkpoint(chkp))
IMUDevNNTrainingLib.load!(model, path)

# move to GPUs if needed
if USE_GPUs
    model = model |> gpu
end
```

To list available checkpoint indices you may use:

```@docs
IMUDevNNTrainingLib.list_existing_checkpoint_indices
```