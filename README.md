# IMUDevNNTrainingLib

[![][docs-dev-img]][docs-dev-url]
[![Build Status](https://github.com/imu-dev/IMUDevNNTrainingLib.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/imu-dev/IMUDevNNTrainingLib.jl/actions/workflows/CI.yml?query=branch%3Amain)

An opinionated toolbox for training Neural Networks with [Lux](https://github.com/LuxDL/Lux.jl).

A frequently repeated patterns employed during training of neural nets are abstracted away and conveniently packaged into various structs and functions, at the expense of making some choices for the user and depending on packages that aren't exactly slim.

> [!IMPORTANT]
> This package **<u>is not</u>** registered with Julia's [General Registry](https://github.com/JuliaRegistries/General), but instead, with `imu.dev`'s local [IMUDevRegistry](https://github.com/imu-dev/IMUDevRegistry). In order to use this package you will need to add [IMUDevRegistry](https://github.com/imu-dev/IMUDevRegistry) to the list of your registries.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://imu-dev.github.io/IMUDevNNTrainingLib.jl/dev
