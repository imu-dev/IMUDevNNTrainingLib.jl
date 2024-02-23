using IMUDevNNTrainingLib
using LinearAlgebra
using Test

const TLib = IMUDevNNTrainingLib

@testset "IMUDevNNTrainingLib.jl" begin
    @testset "Checkpointer" begin
        mktempdir() do tmpdir
            chkp = Checkpointer(; dir=tmpdir,
                                save_every=5,
                                continue_from=:last)
            @test TLib.list_existing_checkpoint_indices(chkp) == []
            @test TLib.isnothing(index_of_last_checkpoint(chkp))
            @test TLib.last_checkpoint_next_epoch(chkp) == (nothing, 1)
            # check that Flux throws out a warning that there are no
            # trainable paramters in the model (our model is a String, so there
            # really should be none...)
            @test_warn r".*no trainable parameters.*" TLib.start!(chkp, "dummy_model")

            model, opt_state, log, start_epoch = TLib.start!(chkp, "dummy_model")
            @test model == "dummy_model"
            @test opt_state == ()
            @test log == []
            @test start_epoch == 1
            return nothing
        end
    end
    @testset "temporal_data.jl" begin
        a = rand(3, 4, 5)
        b = rand(3, 4, 5)
        td = temporal_data(Float32, a, b; skipfirst=true)
        td2 = temporal_data(Float32.(a[:, :, 2:end]), Float32.(b[:, :, 2:end]))
        @test td.x == td2.x
        @test td.y == td2.y
        @test IMUDevNNLib.MLUtils.numobs(td) == 4
        @test maximum(norm.(IMUDevNNLib.MLUtils.getobs(td, 1)[1] .-
                            [x for x in eachslice(a[:, 1, 2:end]; dims=2)])) < 1e-7
        @test maximum(norm.(IMUDevNNLib.MLUtils.getobs(td, 1)[2] .-
                            [x for x in eachslice(b[:, 1, 2:end]; dims=2)])) < 1e-7
    end
end