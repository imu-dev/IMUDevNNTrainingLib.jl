# Checkpointing

`Lux` is agnostic with respect to the way the user checkpoints the trained models. In particular the data-saving backend, the naming conventions and the choice of objects that need saving are all left up to the user. Although great from the perspective of flexibility, it calls for writing quite an extensive boilerplate code. `IMUDevNNTrainingLib` makes all those choices for the user and in return streamlines the checkpointing process. In particular, the following choices are made:

- [`JLD2`](https://github.com/JuliaIO/JLD2.jl) is used as a backend for saving the state of the trained Neural Net
- A checkpoint comprises of saved:
  - `model_parameters`: parameters of the trained Neural Network
  - `model_states`: states (i.e. non-trainable parameters) of the trained Neural Network
  - `opt_state`: parameters of the optimizer
  - `log`: a training log (with loss functions)
  - `other`: a dictionary of any other parameters that the user wishes to save. For instance, `plateau_detector` may be saved here.
- Names of checkpoints take the format `checkpoint=$(epoch).jld2`, where `$(epoch)` is the index of the epoch at the end of which the checkpoint has been made.

The main object is `Checkpointer`:
```@docs
Checkpointer
```

The usual training workflow involves `start`ing it (which will load the specified checkpoint) as well as `checkpoint`ing it on every epoch, which will save the model when appropriate:

```@docs
start
checkpoint
```

!!! note
    Sometimes the user may wish to save and, perhaps, update additional variables. For instance a common use case would be to save the state of the [`PlateauDetector`](@ref). In that case the following could be done:
    ```julia
    model = ...
    ch = Checkpointer()
    chkp_data, start_epoch = start(ch)
    if isnothing(chkp_data)
        do_custom_initialization()
    end
    pd = chkp_data.other[:plateau_detector]
    ```
    and if we'd like to start using a different schedule of learning rates we could adjust them with:
    ```julia
    new_schedule = ParameterSchedulers.Stateful(Exp(1e-3, 0.15))
    Optimisers.adjust!(chkp_data.opt_state, pd, new_schedule)
    ```
    before we resume the training.

For testing it is often enough to call `load_checkpoint` directly, instead of trying to establish the starting epoch as well:

```@docs
load_checkpoint
```

To pick the appropriate checkpoint one can pick from the following helper functions:

```@docs
path_to_checkpoint
index_of_last_checkpoint
path_to_last_checkpoint
IMUDevNNTrainingLib.index_of_last_checkpoint_prior_to
```

A typical example of loading for testing is given below:

```julia
# define model architecture
model = ...

chkp = Checkpointer(; dir="saved_checkpoints")
chkp_data = load_checkpoint(model, path_to_last_checkpoint(chkp))
```

To list available checkpoint indices you may use:

```@docs
IMUDevNNTrainingLib.list_existing_checkpoint_indices
```