module IMUDevNNTrainingLib

using Reexport

using Dates
using Flux
using Humanize: digitsep
@reexport using IMUDevNNLib
using JLD2
using Term

include("checkpointer.jl")
include("plateau_detector.jl")
include("loader_info.jl")
include("progress_printing.jl")

export Checkpointer, checkpoint, index_of_last_checkpoint, path_to_checkpoint, start!

export PlateauDetector

export basic_info
export info_panel, summary_panel, start_info, echo_epoch, echo_summary

end
