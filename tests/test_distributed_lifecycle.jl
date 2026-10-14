# Run: julia --startup-file=no tests/test_distributed_lifecycle.jl
# Uses stdlib-only stubs: no PowerModels/Ipopt installation is needed.
using Distributed, Test
const VENDOR = normpath(joinpath(@__DIR__, "..", "data_generation", "src"))
addprocs(6)
@everywhere begin
    using Distributed, Dates
    @eval module PM
        struct QCLSPowerModel end
    end
    @eval module JuMP
        optimizer_with_attributes(args...) = nothing
    end
    @eval module Ipopt
        const Optimizer = nothing
    end
    const DEFAULT_INPUTS = String[]
    const DEFAULT_OUTPUTS = String[]
    const DEFAULT_DUALS = String[]
    const TOL = 1e-6
    sample_polytope_cprnd(args...) = nothing
    function initialize(args...)
        return [], [], [], Dict{String, Any}("duals"=>Dict{String, Any}("unique_active_sets"=>Set{Int}())), nothing, [], Dict()
    end
    function sample_support(support_ch, sampler, sample_ch, final_ch)
        while !isready(final_ch)
            sleep(0.01)
        end
        # Exceeds Channel capacity (28 for seven Julia processes).
        for i in 1:45
            put!(sample_ch, i)
        end
        return :support_done
    end
    function sample_producer(args...)
        sample_ch, final_ch, support_ch = args[26], args[29], args[32]
        sleep(0.1)
        put!(final_ch, "DONE FLAG")
        put!(support_ch, "DONE FLAG")
        for _ in 1:args[12]
            put!(sample_ch, "DONE FLAG")
        end
        return args[13]
    end
    function sample_processor(net, net_r, r_solver, opf_solver, sample_ch, final_ch, polytope_ch, result_ch, args...)
        while !isready(final_ch)
            sleep(0.01)
        end
        # Simulate final publishers blocked on full queues after DONE.
        for i in 1:25
            put!(result_ch, i)
            put!(polytope_ch, i)
        end
        return :processor_done
    end
end
include(joinpath(VENDOR, "create_samples_distributed.jl"))
net = Dict("name"=>"stub")
@testset "late queue shutdown, repeatable worker reuse" begin
    for batch in 1:3
        result, _ = dist_create_samples(net, 1; nproc=7)
        @test result["duals"]["unique_active_sets"] == Int[]
        @test nprocs() == 7
    end
end
# A remote error must stop all old workers and fail, even if producer is waiting.
@everywhere function sample_processor(net, net_r, r_solver, opf_solver, sample_ch, final_ch, polytope_ch, result_ch, args...)
    error("deliberate worker failure")
end
@testset "worker exception propagates" begin
    @test_throws RemoteException dist_create_samples(net, 1; nproc=7)
    @test nprocs() == 1
end
# Exercise the real processor with an infeasible OPF and an unsuccessful
# relaxed projection. The result must still reach its consumer.
const PowerModels = PM
@eval PM instantiate_model(args...) = nothing
build_opf_var_load(args...) = nothing
find_nearest_feasible_model(args...; kwargs...) = nothing
set_network_load!(args...; kwargs...) = nothing
perturb_topology!(args...; kwargs...) = nothing
perturb_costs!(args...; kwargs...) = nothing
change_bus_type!(args...) = nothing
run_ac_opf_pfdelta(args...; kwargs...) = (nothing, false, nothing)
find_nearest_feasible(args...; kwargs...) = (0.0, [1.0], [1.0], false)
const R_TOLERANCE = 1e-6
include(joinpath(VENDOR, "tasks.jl"))
@testset "infeasible sample still produces feedback" begin
    sample = RemoteChannel(() -> Channel{Any}(8))
    done = RemoteChannel(() -> Channel{Any}(1))
    poly = RemoteChannel(() -> Channel{Any}(8))
    result = RemoteChannel(() -> Channel{Any}(8))
    put!(sample, [1.0, 1.0])
    worker = @async sample_processor(Dict(), Dict(), nothing, nothing, sample, done, poly, result, "none", "none")
    status = timedwait(() -> isready(result) || istaskdone(worker), 5.0)
    @test status == :ok
    @test isready(result)
    if isready(result)
        feedback = take!(result)
        @test feedback[3] == false
    end
    put!(done, "DONE FLAG")
    put!(sample, "DONE FLAG")
    @test timedwait(() -> istaskdone(worker), 5.0) == :ok
    fetch(worker)
end
chebyshev_center(A, b) = ([1.0, 1.0], 1.0)
store_feasible_sample(s, u, v, k, i, K, stats, args...) = (s, u, v, k + 1, stats)
store_feasible_sample_json(args...) = nothing
@testset "real producer replenishes with a sole support consumer" begin
    sample = RemoteChannel(() -> Channel{Any}(8))
    done = RemoteChannel(() -> Channel{Any}(1))
    poly = RemoteChannel(() -> Channel{Any}(8))
    result = RemoteChannel(() -> Channel{Any}(8))
    support = RemoteChannel(() -> Channel{Any}(8))
    sampler = (A, b, x0, n; kwargs...) -> ones(2, n)
    state = Dict("inputs"=>Dict(), "outputs"=>Dict(), "duals"=>Dict())
    attempt_count = Ref(0)
    sampler_task = @async sample_support(support, sampler, sample, done)
    processor_task = @async begin
        while !isready(done)
            x = take!(sample)
            x isa String && break
            attempt_count[] += 1
            # First two attempts fail; subsequent points replenish and succeed.
            feasible = attempt_count[] > 2
            sleep(0.02)
            put!(result, (x, nothing, feasible, false, 0.02, Dict(), Dict()))
        end
    end
    producer_task = @async sample_producer(
        zeros(0, 2), zeros(0, 1), sampler, Dict{Symbol, Any}(),
        reshape([1.0, 1.0], 1, 2), 6, 0.0, 0.0, 0.0, 100, 30, 1,
        state, false, false, 0, false, false, 0, false, "stub", String[],
        String[], false, "", sample, poly, result, done, 0, 0, support)
    @test timedwait(() -> istaskdone(producer_task), 20.0) == :ok
    @test fetch(producer_task) === state
    @test attempt_count[] >= 8
    @test timedwait(() -> istaskdone(sampler_task) && istaskdone(processor_task), 5.0) == :ok
    fetch(sampler_task)
    fetch(processor_task)
end
println("VENDOR SHUTDOWN TEST PASSED")
