# Pretty printing

`IMUDevNNTrainingLib` implements a handful of useful pretty printing features. To accompany them there are a couple of routines that extract dimension/size information from the [`Flux.DataLoader`](https://fluxml.ai/Flux.jl/stable/data/mlutils/#DataLoader) holding data of the type [`TemporalData`](https://imu-dev.github.io/IMUDevNNLib.jl/dev/pages/temporal_data/). These are:

```@docs
IMUDevNNTrainingLib.IMUDevNNLib.num_samples
IMUDevNNTrainingLib.feature_dim
IMUDevNNTrainingLib.target_dim
IMUDevNNTrainingLib.batch_size
```

All the above, as well as some additional ones, can be extracted all at once from the `loader` either in a form of a `NamedTuple` or a formatted `String` using convenience methods:

```@docs
IMUDevNNTrainingLib.basic_info
IMUDevNNTrainingLib.basic_info_as_string
```

However, all of the above are rarely used directly. Instead, they are being called by the method `start_info` (intended to be called at the beginning of training) that nicely formats the results:

```@docs
start_info
```

Additionally, the following could be called during training to display progress:

```@docs
echo_epoch
echo_summary
```


