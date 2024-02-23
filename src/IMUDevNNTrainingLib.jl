module IMUDevNNTrainingLib

using Reexport

using Flux
using Humanize: digitsep
@reexport using IMUDevNNLib
using JLD2
using Term

include("checkpointer.jl")
include("plateau_detector.jl")
include("temporal_data.jl")
include("loader_info.jl")
include("progress_printing.jl")

# temporal_data.jl
export TemporalData, temporal_data, num_samples, num_timepoints, state_dim, obs_dim

export Checkpointer, checkpoint, index_of_last_checkpoint, path_to_checkpoint, start!

export PlateauDetector

export basic_info
export info_panel, summary_panel, start_info, echo_epoch, echo_summary

end
