module IMUDevNNTrainingLib

using Reexport

using Dates
using DocStringExtensions
using Humanize: digitsep
@reexport using IMUDevNNLib
using JLD2
using Lux
using MLUtils
using Optimisers
using ParameterSchedulers
using Term

include("utils.jl")
include("checkpointer.jl")
include("plateau_detector.jl")
include("loader_info.jl")
include("progress_printing.jl")

export currentvalue

export Checkpointer, checkpoint, index_of_last_checkpoint, path_to_checkpoint,
       start, load_checkpoint, path_to_last_checkpoint

export PlateauDetector, learning_rate

export basic_info
export info_panel, summary_panel, start_info, echo_epoch, echo_summary

end
