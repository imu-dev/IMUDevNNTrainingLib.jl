# Plateau detector

It is often recommended to reduce the learning rate when the tracked loss metric has stopped improving for some period of time (for instance, in `Python`, various Neural Net libraries implement `ReduceLROnPlateau`, see [keras's](https://keras.io/api/callbacks/reduce_lr_on_plateau/) or [pytorch's](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) implementations). To this end we introduce a `PlateauDetector` that tracks the target loss metric and updates learning rate in case it stops improving for a sufficient number of ticks.

!!! note
    We will use the word **tick** to refer to a sequence of operations that results in `PlateauDetector` checking the loss metric precisely once. It is up to the user to decide how often that happens.

!!! note
    Unlike `python`'s `ReduceLROnPlateau` we don't restrict the reduction of the learning rate to be of the form of an exponential decay. Instead, we let the user choose the schedule along which the learning rate is being modified using the interface of [`Stateful`](https://fluxml.ai/ParameterSchedulers.jl/dev/api/general/#ParameterSchedulers.Stateful) defined in [ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl). In particular, an exponential decay could be defined via:
    ```julia
    using IMUDevNNTrainingLib
    using ParameterSchedulers
    using ParameterSchedulers: Stateful

    init_learning_rate = 1e-3
    decay = 0.7
    scheduler = Stateful(Exp(init_learning_rate, decay))
    pd = PlateauDetector(; scheduler)
    ```

```@docs
PlateauDetector
IMUDevNNTrainingLib.step!
IMUDevNNTrainingLib.Optimisers.adjust!
```

!!! warning
    To avoid name clashes `step!` is not exported by `IMUDevNNTrainingLib`. It is recommended to use it as follows:
    ```julia
    using IMUDevNNTrainingLib
    using Optimisers
    const NNTrLib = IMUDevNNTrainingLib
    pd = PlateauDetector(; η=1e-4)
    optimizer_state = ...
    loss = ...
    needs_update = NNTrLib.step!(pd, loss)
    if needs_update
        Optimisers.adjust!(optimizer_state, pd)
    end
    ```

Additionally the following utility function is implemented:

```@docs
learning_rate
```

An overloaded `Optimisers.adjust!` makes it possible to change the previously set schedules and adjust the state of the optimizer accordingly.

See also the [examples/plateau_detector.jl](https://github.com/imu-dev/IMUDevNNTrainingLib.jl/blob/main/examples/plateau_detector.jl) file for more details on how to use a `PlateauDetector`.