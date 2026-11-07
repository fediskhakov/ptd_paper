module HSZRefinementBudgetTests
using Test
include(joinpath(@__DIR__, "..", "scripts", "5_hsz.jl"))

@testset "HSZ refinement defaults to one modest grid with a bounded worker" begin
    request = _refinement_arguments(String[])
    @test request.points == 18
    @test request.max_seconds == 900.0
    @test _refinement_arguments(["--run", "base", "out", "19",
        "--max-seconds", "600"]).max_seconds == 600.0
    @test_throws ArgumentError _refinement_arguments(["--run", "base", "out", "18", "19"])
    @test_throws ArgumentError _refinement_budget(901)
    @test_throws ArgumentError _refinement_budget(0)
    @test_throws ArgumentError _refinement_budget(Inf)
    @test _refinement_budget(0.1) == 0.1
    worker = collect(_refinement_worker_command("baseline.jld2", "output.jld2", 19))
    @test "--threads=$(Threads.nthreads())" in worker
    worker_flag = findfirst(==("--refinement"), worker)
    @test !isnothing(worker_flag)
    @test endswith(worker[worker_flag - 1], "scripts/5_hsz.jl")
    @test worker[worker_flag:end] == ["--refinement", "--worker",
        "baseline.jld2", "output.jld2", "19"]
    @test_throws ArgumentError main(["--refinement", "--invalid"])
    @test_throws ArgumentError main(["--timing-only", "--invalid"])
    @test_throws ArgumentError main(["--timing-only", "--output", "--resume"])
    @test_throws ArgumentError main(["--render-only", "--output-dir", projectdir("data")])
    @test_throws ArgumentError main(["--timing-only", "--output", projectdir("data", "nested", "result.jld2")])
    @test_throws ArgumentError main(["--refinement", "--run", "baseline.jld2", projectdir("data", "result.txt")])
end

@testset "deadline stops only the refinement subprocess and retains its log" begin
    mktempdir() do directory
        marker = joinpath(directory, "late-output.txt")
        log_path = joinpath(directory, "timeout.log")
        # The worker is deliberately idle; no numerical solver is launched.
        code = "println(\"worker started\"); flush(stdout); sleep(10); write(ARGS[1], \"late\")"
        result = _run_refinement_worker(`$(Base.julia_cmd()) --startup-file=no -e $code $marker`;
            max_seconds=0.2, log_path)
        @test result.status === :timed_out
        @test isfile(log_path)
        @test !isfile(marker)
        # A fresh worker still completes: the caller and its execution context survive.
        success_log = joinpath(directory, "success.log")
        completed = _run_refinement_worker(
            `$(Base.julia_cmd()) --startup-file=no -e 'println("completed")'`;
            max_seconds=20, log_path=success_log)
        @test completed.status === :completed
        @test completed.exit_code == 0
        @test occursin("completed", read(success_log, String))
        failure = _run_refinement_worker(
            `$(Base.julia_cmd()) --startup-file=no -e 'exit(3)'`;
            max_seconds=20, log_path=joinpath(directory, "failed.log"))
        @test failure.status === :failed
        @test failure.exit_code == 3
        request = (; baseline="baseline.jld2", output="result.jld2", points=18)
        status_path = joinpath(directory, "status.toml")
        _write_refinement_budget_status(status_path, result, request, log_path)
        status = TOML.parsefile(status_path)
        @test status["status"] == "timed_out"
        @test status["finer_nodes_per_exogenous_dimension"] == 18
        @test status["maximum_worker_seconds"] == 0.2
        @test occursin("warmup", status["budget_scope"])
    end
end

@testset "refinement diagnostics use scratch rather than the data directory" begin
    paths = _refinement_runtime_paths()
    try
        @test isdir(paths.directory)
        @test dirname(paths.log_path) == paths.directory
        @test dirname(paths.status_path) == paths.directory
        @test startswith(relpath(paths.directory, projectdir("data")), "..")
        request = (; baseline="baseline.jld2", output="result.jld2", points=18)
        result = (; status=:timed_out, max_seconds=0.2, elapsed_seconds=0.3,
            exit_code=0, termination_signal=15)
        write(paths.log_path, "worker stopped")
        _write_refinement_budget_status(paths.status_path, result, request, paths.log_path)
        @test TOML.parsefile(paths.status_path)["worker_log"] == paths.log_path
        @test read(paths.log_path, String) == "worker stopped"
    finally
        rm(paths.directory; recursive=true)
    end
end

# Only the arguments are parsed here; a render would write into Tables/.
@testset "refinement render-only composes the footnote's saved artifacts" begin
    shipped = Ptd.HSZ.paper_hsz_grid_refinement_paths()
    request = _refinement_arguments(["--render-only"])
    @test request.refinement_paths == shipped
    @test request.baseline == Ptd.HSZ.default_hsz_utility_artifact_path()
    @test !request.include_timings
    timed = _refinement_arguments(["--include-timings", "--render-only"])
    @test timed.include_timings
    @test timed.refinement_paths == shipped
    explicit = _refinement_arguments(
        ["--render-only", "baseline.jld2", "refinement.jld2"])
    @test explicit.baseline == abspath("baseline.jld2")
    @test explicit.refinement_paths == [abspath("refinement.jld2")]
    @test !explicit.include_timings
    # --run renders only the artifact that it writes.
    run_request = _refinement_arguments(["--run", "baseline.jld2",
        "out.jld2", "19", "--include-timings"])
    @test run_request.refinement_paths == [abspath("out.jld2")]
    @test run_request.points == 19
    @test run_request.include_timings
    @test _refinement_arguments(String[]).refinement_paths ==
        [Ptd.HSZ.default_hsz_grid_refinement_path()]
    # A misspelled option is not taken for an artifact path.
    @test_throws ArgumentError _refinement_arguments(
        ["--render-only", "--include-timing"])
    @test_throws ArgumentError _refinement_arguments(
        ["--run", "baseline.jld2", "--max-second", "10"])
end
end
