module IMUDevNNTrainingLib

using Reexport

using Flux
@reexport using IMUDevNNLib
using JLD2

include("temporal_data.jl")
include("checkpointer.jl")
include("plateau_detector.jl")

# temporal_data.jl
export TemporalData, temporal_data

export Checkpointer, checkpoint, index_of_last_checkpoint, path_to_checkpoint

export PlateauDetector

end
