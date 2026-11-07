using Test
using Ptd
using JLD2
using LinearAlgebra

const SinglePathTimingTest = Ptd.FixedCalibrationTiming

@testset "single-path latency measures fresh policies" begin
    timing = SinglePathTimingTest
    BLAS.set_num_threads(1)
    @test_throws ErrorException timing.main(["--smoke"])
    @test_throws ErrorException timing.main(["--replace-panel", "ar1", "unused.jld2"])
    @test_throws ErrorException timing.main(["--replace-panel", "ar1"])
    @test_throws ErrorException timing.main(["--allow-deterministic-grid-refresh"])
    @test isnothing(timing.timing_arguments(String[]).panels)
    for panels in ((:deterministic,), (:iid_baseline, :iid_high_variance), (:ar1,), (:hsz,))
        @test timing.timing_arguments(String[]; panels).panels == panels
        @test timing.timing_arguments(["--panels", "ar1,hsz"]; panels).panels == [:ar1, :hsz]
    end
    input_files = timing._timing_input_files()
    @test input_files == Ptd._single_path_source_files()
    @test length(input_files) == length(unique(getproperty.(input_files, :path)))
    @test any(file -> file.path == "src/simulation/common/fixed_calibration_timing.jl", input_files)
    @test all(file -> file.path != "scripts/6_timing.jl", input_files)
    @test timing.timing_repetitions() == (deterministic=100, iid_baseline=10,
        iid_high_variance=10, ar1=5, hsz=5)
    @test all(==(2), values(timing.timing_repetitions(smoke=true)))
    @test all(==(7), values(timing.timing_repetitions(repeats=7)))
    @test all(==(1), values(timing.timing_repetitions(repeats=1, smoke=true)))
    @test_throws ArgumentError timing.timing_repetitions(repeats=0)
    grids = timing.deterministic_timing_grids()
    @test grids.dcegm1 == collect(range(0.0, 50.0; length=100))
    @test grids.dcegm2 == collect(range(0.0, 100.0; length=200))
    @test grids.dcegm3 == collect(range(0.0, 100.0; length=400))
    @test length(grids.dcegm4) == 400
    @test first(grids.dcegm4) == 0.0 && last(grids.dcegm4) == 10_000.0
    @test all(isapprox.(diff(log1p.(grids.dcegm4)), log1p(10_000.0) / 399;
        rtol=1e-10, atol=1e-13))
    if Threads.nthreads() == 1
        @test_throws ArgumentError timing.run_single_path_timings(repeats=0, smoke=true)
        cases = timing.timing_cases(smoke=true)
        case = only(filter(c -> c.panel === :iid_baseline, cases))
        deterministic = only(filter(c -> c.panel === :deterministic, cases))
        @test timing._selected_timing_panels([:iid_high_variance, :iid_baseline], cases) ==
            [:iid_baseline, :iid_high_variance]
        @test_throws ArgumentError timing._selected_timing_panels([:bad_panel], cases)
        @test_throws ArgumentError timing._selected_timing_panels([:ar1, :ar1], cases)
        @test deterministic.methods == (:ptd, :dcegm1, :dcegm2, :dcegm3, :dcegm4)
        @test deterministic.settings.asset_grids == grids
        @test deterministic.model.beta == 0.98 && deterministic.model.gamma == 1.0
        @test deterministic.model.return_rate == 0.1 && deterministic.model.cash_floor == 3.0
        @test deterministic.initial.cash == 16.0 && all(==(1.0), deterministic.settings.incomes)
        @test !deterministic.settings.allow_uncertified_tail
        for method in deterministic.methods
            policy = deterministic.build(method)
            deterministic.validate(policy, method, deterministic.model, deterministic.process)
            evaluation = deterministic.evaluate(policy, method, deterministic.model,
                deterministic.process, deterministic.initial,
                timing._timing_shocks(deterministic, 1))
            @test isnothing(timing._validate_deterministic_timing_path(evaluation, deterministic))
            @test only(evaluation.utility) ≈ sum(0.98^(t-1) * log(evaluation.consumption[t])
                for t in 1:3)
            @test all(evaluation.cash .>= 3.0)
            if method !== :ptd
                @test policy.bf.a == grids[method]
            end
            invalid_cash = copy(evaluation.cash)
            invalid_cash[2] += 0.5
            @test_throws ErrorException timing._validate_deterministic_timing_path(
                merge(evaluation, (; cash=invalid_cash)), deterministic)
            invalid_consumption = copy(evaluation.consumption)
            invalid_consumption[end] /= 2
            @test_throws ErrorException timing._validate_deterministic_timing_path(
                merge(evaluation, (; consumption=invalid_consumption)), deterministic)
        end
        policies = Any[]
        tracked = merge(case, (; build=method -> begin
            policy = case.build(method)
            push!(policies, policy)
            policy
        end))
        shocks = timing._timing_shocks(case, 1)
        first_run = timing.measure_single_path(tracked, :ptd_osa, shocks)
        second_run = timing.measure_single_path(tracked, :ptd_osa, shocks)
        @test length(policies) == 2
        @test policies[1].cache !== policies[2].cache
        @test Ptd.plan_cache_statistics(policies[1].cache).builds > 0
        @test Ptd.plan_cache_statistics(policies[2].cache).builds > 0
        @test first_run.utility == second_run.utility

        mktempdir() do directory
            result = timing.main(["--smoke", "--repeats", "2", "--output-dir", directory])
            saved = timing.load_single_path_timings(result.artifact_path)
            @test saved.configuration.smoke
            checkpoint_path = joinpath(directory, "single-path-timings.partial.jld2")
            @test isfile(checkpoint_path)
            checkpoint = JLD2.load(checkpoint_path, "checkpoint")
            @test checkpoint.status === :partial
            @test length(checkpoint.rows) == 17
            @test all(row -> length(row.samples) == 2, checkpoint.rows)
            @test saved.configuration.julia_threads == saved.configuration.blas_threads == 1
            @test saved.configuration.paths_per_measurement == 1
            @test length(saved.configuration.configurations) == 5
            @test length(saved.rows) == 17
            @test saved.schema == "ptd-single-path-timing-v2"
            @test saved.configuration.repeats_override == 2
            @test all(==(2), values(saved.configuration.repeats_by_panel))
            @test all(c -> c.repeats == saved.configuration.repeats_by_panel[c.panel],
                saved.configuration.configurations)
            @test any(file -> file.path == joinpath("scripts", "2_deterministic.jl"), saved.input_files)
            @test all(c -> size(c.innovations, 1) == 1 && c.horizon == 3,
                saved.configuration.configurations)
            @test all(row -> row.repeats == saved.configuration.repeats_by_panel[row.panel] ==
                    length(row.samples) == 2 &&
                all(s -> s.total_seconds == s.construction_seconds + s.evaluation_seconds &&
                    s.utility == row.utility, row.samples), saved.rows)
            @test all(row -> row.construction_seconds == timing.median(
                    getproperty.(row.samples, :construction_seconds)) &&
                row.evaluation_seconds == timing.median(getproperty.(row.samples, :evaluation_seconds)) &&
                row.total_seconds == timing.median(getproperty.(row.samples, :total_seconds)), saved.rows)
            @test all(row -> row.minimum_total_seconds <= row.total_seconds <=
                row.maximum_total_seconds, saved.rows)
            @test occursin("Smoke test", read(result.table_path, String))
            @test occursin("separately timed", read(result.table_path, String))
            @test occursin("Repetitions", read(result.table_path, String))
            @test occursin("a count of one is a single measurement", read(result.table_path, String))
            @test occursin("Deterministic & DC-EGM4 & 2", read(result.table_path, String))
            rendered = timing.main(["--render-only", result.artifact_path,
                "--output-dir", joinpath(directory, "rendered")])
            @test read(rendered.table_path) == read(result.table_path)
            @test_throws ErrorException timing.main(["--render-only", result.artifact_path])
            damaged = merge(saved, (; configuration=merge(saved.configuration, (; repeats_override=999))))
            JLD2.jldsave(joinpath(directory, "damaged.jld2"); artifact=damaged)
            @test_throws ErrorException timing.load_single_path_timings(joinpath(directory, "damaged.jld2"))

            # A refreshed panel can populate the combined table without
            # relabelling preserved panels or rewriting either input artifact.
            function save_fixture(path, candidate)
                payload = (; (key => getproperty(candidate, key) for key in keys(candidate)
                    if key !== :authentication_fingerprint)...)
                artifact = merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
                JLD2.jldsave(path; artifact)
                return artifact
            end
            shift_sample(sample) = merge(sample, (;
                construction_seconds=sample.construction_seconds + 1,
                evaluation_seconds=sample.evaluation_seconds + 2,
                total_seconds=(sample.construction_seconds + 1) + (sample.evaluation_seconds + 2)))
            raw_ar1 = [merge(row, (; samples=row.panel === :ar1 ?
                shift_sample.(row.samples) : NamedTuple[])) for row in saved.raw_measurements]
            rows_ar1 = map(filter(row -> row.panel === :ar1, saved.rows)) do row
                samples = shift_sample.(row.samples)
                totals = getproperty.(samples, :total_seconds)
                merge(row, (; samples,
                    construction_seconds=timing.median(getproperty.(samples, :construction_seconds)),
                    evaluation_seconds=timing.median(getproperty.(samples, :evaluation_seconds)),
                    total_seconds=timing.median(totals), minimum_total_seconds=minimum(totals),
                    maximum_total_seconds=maximum(totals)))
            end
            ar1_config = merge(saved.configuration, (; selected_panels=[:ar1],
                configurations=filter(case -> case.panel === :ar1, saved.configuration.configurations)))
            fixture_files = [file.path == joinpath("src", "g2egm", "local_envelope.jl") ?
                merge(file, (; sha256=repeat("a", 64))) : file for file in saved.input_files]
            fixture_provenance = merge(saved.provenance,
                (; configuration=Ptd.configuration_metadata(ar1_config)))
            refresh_path = joinpath(directory, "ar1-refresh.jld2")
            refreshed = save_fixture(refresh_path, merge(saved, (;
                configuration=ar1_config, rows=rows_ar1, raw_measurements=raw_ar1,
                input_files=fixture_files, input_fingerprint=Ptd.stable_fingerprint(fixture_files),
                provenance=fixture_provenance)))
            original_bytes, refreshed_bytes = read(result.artifact_path), read(refresh_path)
            composed_path = joinpath(directory, "composed.tex")
            composed = timing.render_composed_single_path_timings(result.artifact_path,
                [:ar1 => refresh_path], composed_path)
            @test length(composed.rows) == 17
            @test all(source -> source.artifact_fingerprint ==
                (source.panel === :ar1 ? refreshed.authentication_fingerprint : saved.authentication_fingerprint),
                composed.panel_sources)
            @test all(source -> source.source_fingerprint ==
                (source.panel === :ar1 ? refreshed.input_fingerprint : saved.input_fingerprint),
                composed.panel_sources)
            @test all(row -> row.total_seconds == only(filter(old -> old.panel === row.panel &&
                old.method === row.method, row.panel === :ar1 ? refreshed.rows : saved.rows)).total_seconds,
                composed.rows)
            @test read(result.artifact_path) == original_bytes && read(refresh_path) == refreshed_bytes
            @test !occursin("Panels retain their individual recorded source versions", read(composed_path, String))
            @test count("% Timing panel:", read(composed_path, String)) == 5
            # Header comments name each artifact without its local folder.
            composed_text = read(composed_path, String)
            @test !occursin(directory, composed_text)
            @test occursin("% Artifact path: ar1-refresh.jld2\n",
                composed_text)
            cli_composed = timing.main(["--render-only", result.artifact_path,
                "--replace-panel", "ar1", refresh_path, "--output-dir", joinpath(directory, "composed")])
            @test read(cli_composed.table_path) == read(composed_path)
            @test_throws ErrorException timing.main(["--render-only", result.artifact_path,
                "--replace-panel", "ar1", refresh_path])
            @test_throws ArgumentError timing.render_composed_single_path_timings(result.artifact_path,
                [:unknown => refresh_path], composed_path)
            @test_throws ArgumentError timing.render_composed_single_path_timings(result.artifact_path,
                [:ar1 => refresh_path, :ar1 => refresh_path], composed_path)
            @test_throws ArgumentError timing.render_composed_single_path_timings(result.artifact_path,
                [:iid_baseline => refresh_path], composed_path)
            mismatch_config = merge(ar1_config, (; configurations=[merge(only(ar1_config.configurations),
                (; gamma=2.0))]))
            mismatch_path = joinpath(directory, "mismatched-refresh.jld2")
            save_fixture(mismatch_path, merge(refreshed, (; configuration=mismatch_config,
                provenance=merge(refreshed.provenance,
                    (; configuration=Ptd.configuration_metadata(mismatch_config))))))
            @test_throws ArgumentError timing.render_composed_single_path_timings(result.artifact_path,
                [:ar1 => mismatch_path], composed_path)
            @test_throws ArgumentError timing.render_composed_single_path_timings(result.artifact_path,
                [:ar1 => joinpath(directory, "damaged.jld2")], composed_path)

            # A lower count/panel subset reuses an authenticated prefix without
            # constructing new policies; exact full-case shocks stay unchanged.
            fewer = timing.main(["--smoke", "--resume", "--repeats", "1",
                "--panels", "iid_baseline,iid_high_variance", "--output-dir", directory]).artifact
            @test length(fewer.rows) == 6
            @test all(row -> row.repeats == row.reused_repeats == 1 &&
                row.available_repeats == 2, fewer.rows)
            @test all(row -> row.samples == only(filter(old -> old.panel === row.panel &&
                old.method === row.method, saved.rows)).samples[1:1], fewer.rows)
            @test all(c -> c.innovations == only(filter(old -> old.panel === c.panel,
                saved.configuration.configurations)).innovations, fewer.configuration.configurations)
            @test all(row -> length(row.samples) == 2, fewer.raw_measurements)
            @test_throws ErrorException timing.main(["--smoke", "--repeats", "1",
                "--output-dir", directory])

            # Increasing only one panel adds its missing repetitions and leaves
            # every prior timing observation byte-for-byte unchanged.
            more = timing.main(["--smoke", "--resume", "--repeats", "3",
                "--panels", "deterministic", "--output-dir", directory]).artifact
            @test length(more.rows) == 5
            @test all(row -> row.repeats == row.available_repeats == 3 &&
                row.reused_repeats == 2, more.rows)
            @test all(row -> row.samples[1:2] == only(filter(old -> old.panel === row.panel &&
                old.method === row.method, saved.rows)).samples, more.rows)
            @test all(row -> length(row.samples) == (row.panel === :deterministic ? 3 : 2),
                more.raw_measurements)

            invalid_path = joinpath(directory, "invalid-checkpoint.jld2")
            invalid = merge(checkpoint, (; status=:complete))
            JLD2.jldsave(invalid_path; checkpoint=invalid)
            @test_throws ErrorException timing._load_timing_checkpoint(invalid_path,
                checkpoint.input_files, checkpoint.configuration)
            @test_throws ErrorException timing._load_timing_checkpoint(checkpoint_path,
                checkpoint.input_files[2:end], checkpoint.configuration)
            @test_throws ErrorException timing._load_timing_checkpoint(checkpoint_path,
                checkpoint.input_files, merge(checkpoint.configuration, (; smoke=false)))
            invalid = merge(checkpoint, (; sessions=["altered-session"]))
            JLD2.jldsave(invalid_path; checkpoint=invalid)
            @test_throws ErrorException timing._load_timing_checkpoint(invalid_path,
                checkpoint.input_files, checkpoint.configuration)
        end
    else
        @test_throws ErrorException timing.run_single_path_timings(smoke=true)
    end
end

@testset "deterministic grid refresh preserves the rest of the timing protocol" begin
    timing = SinglePathTimingTest
    new_grids = timing.deterministic_timing_grids()
    old_grids = merge(new_grids, (; dcegm4=timing.iid_dcegm_asset_grid(400; asset_maximum=60_000.0)))
    original = (; panel=:deterministic, beta=0.98, gamma=1.0, initial_cash=16.0,
        horizon=50, repeats=100, innovations=zeros(1, 49),
        settings=(; asset_grids=old_grids, income=1.0))
    replacement = merge(original, (; settings=merge(original.settings, (; asset_grids=new_grids))))
    @test isnothing(timing._validate_timing_panel_replacement(:deterministic, original, original))
    @test_throws ArgumentError timing._validate_timing_panel_replacement(:deterministic, original, replacement)
    @test isnothing(timing._validate_timing_panel_replacement(:deterministic, original, replacement;
        allow_deterministic_grid_refresh=true))
    @test_throws ArgumentError timing._validate_timing_panel_replacement(:ar1, original, replacement;
        allow_deterministic_grid_refresh=true)
    for changed in (merge(replacement, (; gamma=2.0)), merge(replacement, (; repeats=99)),
        merge(replacement, (; innovations=ones(1, 49))),
        merge(replacement, (; settings=merge(replacement.settings,
            (; asset_grids=merge(new_grids, (; dcegm3=new_grids.dcegm3[1:end-1])))))))
        @test_throws ArgumentError timing._validate_timing_panel_replacement(:deterministic, original, changed;
            allow_deterministic_grid_refresh=true)
    end
    @test_throws ArgumentError timing.render_composed_single_path_timings("unused", [], "unused";
        allow_deterministic_grid_refresh=true)
end
