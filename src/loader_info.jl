"""
    IMUDevNNLib.num_samples(loader::Flux.DataLoader{<:TemporalData})

The number of availble samples held by the `loader` (i.e. the maximal batch
size).
"""
IMUDevNNLib.num_samples(loader::Flux.DataLoader{<:TemporalData}) = num_samples(loader.data)

"""
    feature_dim(loader::Flux.DataLoader{<:TemporalData})

The dimension of the feature space of the data held by the `loader`. For
`TemporalData` the feature dimension is simply the size of the state vector.

!!! note
    If the state is a multidimensional tensor, `feature_dim` is equal to the
    total number of elements in a state tensor.
"""
feature_dim(loader::Flux.DataLoader{<:TemporalData}) = prod(state_dim(loader.data))

"""
    target_dim(loader::Flux.DataLoader{<:TemporalData})

The dimension of the target space of the data held by the `loader`. For
`TemporalData` the target dimension is simply the size of the observation
vector.

!!! note
    If the observation is a multidimensional tensor, `target_dim` is equal to
    the total number of elements in an observation tensor.
"""
target_dim(loader::Flux.DataLoader{<:TemporalData}) = prod(obs_dim(loader.data))

"""
    batch_size(loader::Flux.DataLoader)

The batch size of the `loader`.
"""
batch_size(loader::Flux.DataLoader) = loader.batchsize

"""
    basic_info(loader::Flux.DataLoader)

Return a `NamedTuple` with basic dimensional information about the `loader` and
the data it holds.
"""
function basic_info(loader::Flux.DataLoader{})
    return (; num_samples=digitsep(num_samples(loader); seperator="_"),
            feature_dim=feature_dim(loader),
            target_dim=target_dim(loader),
            batch_size=digitsep(batch_size(loader); seperator="_"),
            num_batches=digitsep(length(loader); seperator="_"),
            trajectory_length=digitsep(num_timepoints(loader.data); seperator="_"))
end

"""
    basic_info_as_string(loader::Flux.DataLoader{<:TemporalData})

Return a formatted string with basic dimensional information about the `loader`
and the data it holds.
"""
function basic_info_as_string(loader::Flux.DataLoader{<:TemporalData})
    i = basic_info(loader)
    return """feature dimension: $(i.feature_dim)
    target dimension: $(i.target_dim)
    trajectory length: $(i.trajectory_length)
    batch size: $(i.batch_size)
    number of batches: $(i.num_batches)
    number of samples: $(i.num_samples)"""
end