"""
    IMUDevNNLib.num_samples(layout::SingleArrayLayout, loader::DataLoader)

The number of availble samples held by the `loader` (i.e. the maximal batch
size).
"""
function IMUDevNNLib.num_samples(layout::SingleArrayLayout, loader::DataLoader)
    d = if loader.data isa Tuple
        first(loader.data)
    else
        loader.data
    end
    return num_samples(layout, d)
end

"""
    feature_dim(layout::SingleArrayLayout, loader::DataLoader)

The dimension of the feature space of the data held by the `loader`.

!!! note
    If the state is a multidimensional tensor, `feature_dim` is equal to the
    total number of elements in a state tensor.
"""
function feature_dim(layout::SingleArrayLayout, loader::DataLoader)
    d = if loader.data isa Tuple
        first(loader.data)
    else
        loader.data
    end

    return prod(state_dim(layout, d))
end

"""
    target_dim(layout::SingleArrayLayout, loader::DataLoader)

The dimension of the target space of the data held by the `loader`. For
`TemporalData` the target dimension is simply the size of the observation
vector.

!!! note
    If the observation is a multidimensional tensor, `target_dim` is equal to
    the total number of elements in an observation tensor.
"""
function target_dim(layout::SingleArrayLayout, loader::DataLoader)
    d = if loader.data isa Tuple
        last(loader.data)
    else
        throw("To compute target dimension `DataLoader.data` must be a Tuple")
    end
    return prod(state_dim(layout, d))
end

"""
    batch_size(loader::DataLoader)

The batch size of the `loader`.
"""
batch_size(loader::DataLoader) = loader.batchsize

"""
    basic_info(layout::SingleArrayLayout, loader::DataLoader)

Return a `NamedTuple` with basic dimensional information about the `loader` and
the data it holds.
"""
function basic_info(layout::SingleArrayLayout, loader::DataLoader)
    return (; num_samples=digitsep(num_samples(layout, loader); seperator="_"),
            feature_dim=feature_dim(layout, loader),
            target_dim=target_dim(layout, loader),
            batch_size=digitsep(batch_size(loader); seperator="_"),
            num_batches=digitsep(length(loader); seperator="_"),
            trajectory_length=digitsep.(num_timepoints.(layout, loader.data);
                                        seperator="_"))
end

"""
    basic_info_as_string(loader::DataLoader)

Return a formatted string with basic dimensional information about the `loader`
and the data it holds.
"""
function basic_info_as_string(layout::SingleArrayLayout, loader::DataLoader)
    i = basic_info(layout, loader)
    tl = if i.trajectory_length isa Tuple
        "($(join(i.trajectory_length, ", ")))"
    else
        i.trajectory_length
    end
    return """feature dimension: $(i.feature_dim)
    target dimension: $(i.target_dim)
    trajectory length$(i.trajectory_length isa Tuple ? "s" : ""): $tl
    batch size: $(i.batch_size)
    number of batches: $(i.num_batches)
    number of samples: $(i.num_samples)"""
end