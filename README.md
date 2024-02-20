# IMUDevNNTrainingLib

[![Build Status](https://github.com/mmider/IMUDevNNTrainingLib.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mmider/IMUDevNNTrainingLib.jl/actions/workflows/CI.yml?query=branch%3Amain)

An opinionated toolbox for training Neural Networks with [Flux](https://github.com/FluxML/Flux.jl).

A frequently repeated patterns employed during training of neural nets are abstracted away and conveniently packaged into various structs and functions, at the expense of making some choices for the user and depending on packages that aren't exactly slim.

## Checkpointing

Saving partially trained neural networks to disk. Uses JLD2 to serialize the data.
