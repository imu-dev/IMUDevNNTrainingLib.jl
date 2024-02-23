num_samples(loader::Flux.DataLoader) = num_samples(loader, loader.data)
num_samples(::Flux.DataLoader, data::TemporalData) = num_samples(data)

feature_dim(loader::Flux.DataLoader) = feature_dim(loader, loader.data)
feature_dim(::Flux.DataLoader, data::TemporalData) = prod(state_dim(data))

target_dim(loader::Flux.DataLoader) = target_dim(loader, loader.data)
target_dim(::Flux.DataLoader, data::TemporalData) = prod(obs_dim(data))

batch_size(loader::Flux.DataLoader) = loader.batchsize

function basic_info(loader::Flux.DataLoader)
    return (; num_samples=digitsep(num_samples(loader); seperator="_"),
            feature_dim=feature_dim(loader),
            target_dim=target_dim(loader),
            batch_size=digitsep(batch_size(loader); seperator="_"),
            trajectory_length=digitsep(num_timepoints(loader.data); seperator="_"))
end

basic_info_as_string(loader::Flux.DataLoader) = basic_info_as_string(loader, loader.data)
function basic_info_as_string(loader::Flux.DataLoader, ::TemporalData)
    i = basic_info(loader)
    return """feature dimension: $(i.feature_dim)
    target dimension: $(i.target_dim)
    trajectory length: $(i.trajectory_length)
    batch size: $(i.batch_size)
    number of samples: $(i.num_samples)"""
end