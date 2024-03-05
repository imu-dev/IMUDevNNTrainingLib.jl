using IMUDevNNTrainingLib
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
            chkp_data, start_epoch = TLib.start(chkp, "dummy_model")
            @test isnothing(chkp_data)
            @test start_epoch == 1
            return nothing
        end
    end
end