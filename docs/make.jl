using Documenter
using IMUDevNNTrainingLib

makedocs(; sitename="IMUDevNNTrainingLib",
         format=Documenter.HTML(),
         modules=[IMUDevNNTrainingLib],
         checkdocs=:exports,
         pages=["Home" => "index.md",
                "Manual" => ["Plateau Detector" => joinpath("pages", "plateau_detector.md"),
                             "Checkpointer" => joinpath("pages", "checkpointer.md"),
                             "Printing" => joinpath("pages", "pretty_printing.md"),
                             "Other" => joinpath("pages", "other.md")]])

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
deploydocs(; repo="github.com/imu-dev/IMUDevNNTrainingLib.jl.git")
