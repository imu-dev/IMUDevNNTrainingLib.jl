module IMUDevNNTrainingLib

using Flux
using JLD2

include("checkpointer.jl")
include("plateau_detector.jl")

export Checkpointer, checkpoint, index_of_last_checkpoint, path_to_checkpoint

export PlateauDetector

end
