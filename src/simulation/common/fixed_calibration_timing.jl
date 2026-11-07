"""Measure fresh policy construction and one-path simulation on one thread for fixed calibrations."""

using Dates
using ..Ptd
using JLD2
using LinearAlgebra
using Printf
using Random
using SHA
using Statistics

const SINGLE_PATH_TIMING_SCHEMA = "ptd-single-path-timing-v2"
const SINGLE_PATH_TIMING_REPEATS = (deterministic=100, iid_baseline=10,
    iid_high_variance=10, ar1=5, hsz=5)
const SINGLE_PATH_CHECKPOINT_SCHEMA = "ptd-single-path-timing-checkpoint-v1"
const SINGLE_PATH_TIMING_SEED = UInt64(0x5054445f54494d45)

"""Resolve measured repetitions per panel; discarded warmups are additional."""
function timing_repetitions(; repeats::Union{Nothing,Int}=nothing, smoke::Bool=false)
    isnothing(repeats) || repeats >= 1 || throw(ArgumentError("repeats must be positive"))
    return (; (panel => (isnothing(repeats) ? (smoke ? 2 : count) : repeats)
        for (panel, count) in pairs(SINGLE_PATH_TIMING_REPEATS))...)
end

"""Match the four grids in scripts/2_deterministic.jl, including their node budgets."""
function deterministic_timing_grids()
    return (dcegm1=collect(range(0.0, 50.0; length=100)),
        dcegm2=collect(range(0.0, 100.0; length=200)),
        dcegm3=collect(range(0.0, 100.0; length=400)),
        dcegm4=iid_dcegm_asset_grid(400; asset_maximum=10_000.0))
end

function _build_deterministic_timing_policy(method, model, grids)
    horizon = model.terminal.final_date
    if method === :ptd
        policy = DeterministicProblem(β=model.beta, γ=model.gamma,
            r=model.return_rate, xFloor=model.cash_floor, incomes=fill(1.0, 1, horizon))
        deterministic_ptd!(policy; options=FusionOptions())
        return policy
    end
    method in keys(grids) || throw(ArgumentError("unknown deterministic method: $method"))
    policy = Ptd.DcEgm.build_agent(model="deterministic", β=model.beta, γ=model.gamma,
        r=model.return_rate, xFloor=model.cash_floor, tMax=horizon,
        a=grids[method], incomes=fill(1.0, horizon))
    Ptd.DcEgm.egm!(policy)
    return policy
end

function _validate_deterministic_timing_policy(policy, method, model, process)
    process === nothing || error("deterministic timing requires a fixed income path")
    fields = if method === :ptd
        policy isa DeterministicProblem || error("wrong deterministic PTD policy type")
        for collection in policy.policy
            Ptd.validate_policy_collection(collection)
        end
        policy
    else
        policy isa Ptd.DcEgm.AgentD || error("wrong deterministic DC-EGM policy type")
        policy.bf
    end
    fields.β == model.beta && fields.γ == model.gamma &&
        fields.r == model.return_rate && fields.xFloor == model.cash_floor &&
        fields.tMax == model.terminal.final_date &&
        length(policy.incomes) == fields.tMax && all(==(1.0), policy.incomes) ||
        error("deterministic timing policy differs from its declared calibration")
    return nothing
end

_timing_consumption(policy::DeterministicProblem, cash, date) = c_x(policy, cash; i=1, t=date)
_timing_consumption(policy::Ptd.DcEgm.AgentD, cash, date) = Ptd.DcEgm.c_x(policy, cash; t=date)
_timing_utility(policy::DeterministicProblem, consumption) = u(policy, consumption)
_timing_utility(policy::Ptd.DcEgm.AgentD, consumption) = Ptd.DcEgm.u(policy, consumption)
_timing_next_cash(policy::DeterministicProblem, assets, date) = xNext_a(policy, assets; i=1, t=date)
_timing_next_cash(policy::Ptd.DcEgm.AgentD, assets, date) = Ptd.DcEgm.xNext_a(policy, assets; t=date)

function _evaluate_deterministic_timing_path(policy, method, model, process, initial, shocks)
    horizon = model.terminal.final_date
    cash, consumption = Vector{Float64}(undef, horizon), Vector{Float64}(undef, horizon)
    cash[1] = initial.cash
    utility, transfers, hits = 0.0, 0.0, 0
    for date in 1:horizon
        x = cash[date]
        c = _timing_consumption(policy, x, date)
        isfinite(c) && 0.0 < c <= x || error("infeasible deterministic timing action")
        consumption[date] = c
        utility += model.beta^(date - 1) * _timing_utility(policy, c)
        if date < horizon
            cash[date + 1] = _timing_next_cash(policy, x - c, date)
            pretransfer_cash = (1 + model.return_rate) * (x - c) + 1.0
            transfers += max(model.cash_floor - pretransfer_cash, 0.0)
            hits += pretransfer_cash < model.cash_floor
        end
    end
    return (; method, utility=[utility], total_transfers=[transfers], floor_hits=[hits],
        cash, consumption)
end

"""Validate the executed deterministic path, separately from the measured loop."""
function _validate_deterministic_timing_path(result, case)
    horizon = case.model.terminal.final_date
    length(result.cash) == length(result.consumption) == horizon &&
        first(result.cash) == case.initial.cash && all(isfinite, result.cash) &&
        all(isfinite, result.consumption) || error("invalid deterministic timing path")
    for date in 1:horizon
        x, c = result.cash[date], result.consumption[date]
        0.0 < c <= x || error("infeasible deterministic timing path")
        if date < horizon
            expected = max((1 + case.model.return_rate) * (x - c) + 1.0,
                case.model.cash_floor)
            result.cash[date + 1] == expected || error("incorrect deterministic cash transition")
        else
            c == x || error("deterministic terminal consumption must exhaust cash")
        end
    end
    return nothing
end

function timing_cases(; smoke::Bool=false)
    horizon = smoke ? 3 : 50
    model = StochasticPTDConfig(beta=0.98, gamma=1.0, return_rate=0.1,
        cash_floor=3.0, terminal_date=horizon)
    cases = NamedTuple[]
    grids = deterministic_timing_grids()
    push!(cases, (; panel=:deterministic, model, process=nothing,
        initial=ObservedState(1, 16.0, IIDState()),
        methods=(:ptd, keys(grids)...),
        build=method -> _build_deterministic_timing_policy(method, model, grids),
        validate=_validate_deterministic_timing_policy,
        evaluate=_evaluate_deterministic_timing_path,
        settings=(income=1.0, incomes=fill(1.0, horizon),
            grid_source="scripts/2_deterministic.jl:deterministic_asset_grids",
            asset_grids=grids, grid_spacing=(dcegm1=:linear, dcegm2=:linear,
                dcegm3=:linear, dcegm4=:log),
            fusion_search_bound=FusionOptions().search_bound, allow_uncertified_tail=false)))
    for (panel, sigma) in ((:iid_baseline, 0.1), (:iid_high_variance, 0.5))
        process = IIDLognormalProcess(sigma=sigma)
        rule = gauss_hermite_rule(process, 5)
        assets = iid_dcegm_asset_grid(smoke ? 101 : Ptd.IID_DCEGM_REPORTED_POINTS)
        build = method -> build_iid_contest_policy(method, model, process;
            base_rule=rule, asset_grid=method === :dcegm ? assets : nothing)
        push!(cases, (; panel, model, process, initial=ObservedState(1, 16.0, IIDState()),
            methods=(:ptd_sceq, :ptd_osa, :dcegm), build,
            validate=validate_iid_contest_policy_alignment, evaluate=evaluate_policy_paths,
            settings=(sigma=sigma, mean_income=1.0, quadrature_nodes=5,
                asset_grid=assets)))
    end
    ar1_config = Ptd.AR1.production_run_config()
    ar1_process = LogAR1Process(rho=ar1_config.rho, sigma=ar1_config.sigma)
    ar1_rule = gauss_hermite_rule(ar1_process, 5)
    ar1_grids = smoke ? AR1G2ContestGridConfiguration(:single_path_smoke,
        Ptd._single_log_closed_grid(30_000.0, 101),
        collect(range(-3.0, 3.0; length=9)),
        Ptd._shifted_log_closed_grid(3.0, 33_000.0 + exp(3.0), 101),
        collect(range(-3.0, 3.0; length=9))) : ar1_config.g2egm_grids
    ar1_build = method -> build_ar1_contest_policy(method, model, ar1_process;
        base_rule=ar1_rule, g2egm_grids=ar1_grids,
        g2egm_options=ar1_config.g2egm_options,
        ptd_osa_options=ar1_config.ptd_osa_options)
    push!(cases, (; panel=:ar1, model, process=ar1_process,
        initial=ObservedState(1, 16.0, LogAR1State(0.0)),
        methods=(:ptd_sceq, :ptd_osa, :g2egm), build=ar1_build,
        validate=validate_ar1_contest_policy_alignment, evaluate=evaluate_policy_paths,
        settings=(rho=ar1_config.rho, sigma=ar1_config.sigma, initial_income=1.0,
            quadrature_nodes=5, grid=Ptd._plain_ar1_g2_grid_configuration(ar1_grids),
            solver_options=Ptd.AR1._plain_g2egm_options(ar1_config.g2egm_options))))
    hsz_config = smoke ? Ptd.HSZ.HSZExampleConfig(horizon=horizon,
        asset_points=41, common_cash_points=41,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5) :
        Ptd.HSZ.production_hsz_example_config()
    _, hsz_process = Ptd.HSZ._model(hsz_config)
    hsz_grid = Ptd.HSZ._g2_grid(hsz_config, hsz_process)
    hsz_rule = gauss_hermite_rule(hsz_process, 5)
    hsz_build = method -> build_hsz_contest_policy(method, model, hsz_process;
        base_rule=hsz_rule, g2egm_grid=hsz_grid)
    push!(cases, (; panel=:hsz, model, process=hsz_process,
        initial=ObservedState(1, 16.0, HSZState(
            hsz_process.unconditional_log_earnings,
            hsz_process.unconditional_log_medical)),
        methods=(:ptd_sceq, :ptd_osa, :g2egm), build=hsz_build,
        validate=validate_hsz_contest_policy_alignment, evaluate=evaluate_policy_paths,
        settings=Ptd.HSZ._scientific_configuration(hsz_config)))
    return cases
end

function _timing_shocks(case, index)
    rng = Xoshiro(SINGLE_PATH_TIMING_SEED + UInt64(index))
    steps = case.model.terminal.final_date - 1
    case.process === nothing && return zeros(1, steps)
    dimension = innovation_dimension(case.process)
    return dimension == 1 ? randn(rng, 1, steps) : randn(rng, 1, steps, dimension)
end

function _validate_single_evaluation(result, method)
    result.method === method && length(result.utility) == 1 &&
        all(isfinite, result.utility) && all(isfinite, result.total_transfers) &&
        all(>=(0), result.floor_hits) || error("invalid single-path evaluation")
    return only(result.utility)
end

"""Time one fresh construction and one path; explicit alignment checks are untimed."""
function measure_single_path(case, method, shocks)
    GC.gc()
    started = time_ns()
    policy = case.build(method)
    construction_seconds = (time_ns() - started) / 1e9
    case.validate(policy, method, case.model, case.process)
    started = time_ns()
    evaluation = case.evaluate(policy, method, case.model,
        case.process, case.initial, shocks)
    evaluation_seconds = (time_ns() - started) / 1e9
    utility = _validate_single_evaluation(evaluation, method)
    case.process === nothing && _validate_deterministic_timing_path(evaluation, case)
    return (; construction_seconds, evaluation_seconds,
        total_seconds=construction_seconds + evaluation_seconds, utility)
end

_timing_input_files() = Ptd._single_path_source_files()

"""Progress writes and flushing are outside every construction/evaluation clock."""
function _timing_progress(parts...)
    println(Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS.sss"), " UTC ", parts...)
    flush(stdout)
end

function _logged_timing_sample(case, method, shocks, stage)
    _timing_progress(stage, ": ", case.panel, " / ", method)
    sample = measure_single_path(case, method, shocks)
    _timing_progress("Completed ", stage, ": ", case.panel, " / ", method,
        @sprintf("; construction %.6fs; path %.6fs; total %.6fs",
            sample.construction_seconds, sample.evaluation_seconds, sample.total_seconds))
    return sample
end

function _timing_case_configuration(case, index)
    shocks = _timing_shocks(case, index)
    return (; panel=case.panel, beta=case.model.beta, gamma=case.model.gamma,
        return_rate=case.model.return_rate, cash_floor=case.model.cash_floor,
        horizon=case.model.terminal.final_date, initial_cash=case.initial.cash,
        methods=case.methods, settings=case.settings, innovations=shocks,
        shock_fingerprint=Ptd.stable_fingerprint(shocks))
end

function _selected_timing_panels(panels, cases)
    available = [case.panel for case in cases]
    isnothing(panels) && return available
    requested = Symbol.(collect(panels))
    !isempty(requested) && length(unique(requested)) == length(requested) &&
        all(panel -> panel in available, requested) ||
        throw(ArgumentError("panels must be distinct names from $(join(available, ','))"))
    return filter(panel -> panel in requested, available)
end

function _timing_checkpoint_configuration(cases, smoke)
    return (; smoke, seed=SINGLE_PATH_TIMING_SEED,
        configurations=[_timing_case_configuration(case, index)
            for (index, case) in enumerate(cases)],
        environment=(julia_version=string(VERSION), kernel=string(Sys.KERNEL),
            machine=Sys.MACHINE, cpu_name=Sys.CPU_NAME,
            julia_threads=Threads.nthreads(), blas_threads=BLAS.get_num_threads(),
            blas_vendor=string(BLAS.vendor())),
        paths_per_measurement=1, warmup=:disposable_full_construction_and_path,
        cache=:fresh_policy_each_repetition)
end

function _validate_timing_samples(samples)
    all(sample -> all(isfinite, (sample.construction_seconds, sample.evaluation_seconds,
            sample.total_seconds, sample.utility)) &&
        sample.construction_seconds >= 0.0 && sample.evaluation_seconds >= 0.0 &&
        sample.total_seconds == sample.construction_seconds + sample.evaluation_seconds,
        samples) || error("checkpoint contains invalid timing samples")
    isempty(samples) || all(sample -> isequal(sample.utility, first(samples).utility), samples) ||
        error("single-path utility changed across fresh-policy repetitions")
    return nothing
end

function _load_timing_checkpoint(path, input_files, configuration)
    checkpoint = JLD2.load(path, "checkpoint")
    checkpoint.schema == SINGLE_PATH_CHECKPOINT_SCHEMA && checkpoint.status === :partial ||
        error("unknown single-path checkpoint schema/status")
    payload = (; (name => getproperty(checkpoint, name) for name in keys(checkpoint)
        if name !== :authentication_fingerprint)...)
    Ptd.stable_fingerprint(payload) == checkpoint.authentication_fingerprint ||
        error("single-path checkpoint fingerprint mismatch")
    checkpoint.input_files == input_files || error("timing checkpoint source inputs changed")
    Ptd.stable_fingerprint(checkpoint.configuration) == Ptd.stable_fingerprint(configuration) ||
        error("timing checkpoint calibration, shocks, or execution conditions changed")
    expected = [(panel=c.panel, method=method) for c in configuration.configurations for method in c.methods]
    [(; row.panel, row.method) for row in checkpoint.rows] == expected ||
        error("timing checkpoint method registration changed")
    for row in checkpoint.rows
        _validate_timing_samples(row.samples)
        all(sample -> sample.session in checkpoint.sessions, row.samples) ||
            error("timing checkpoint sample session is unregistered")
    end
    for case in configuration.configurations
        counts = [length(row.samples) for row in checkpoint.rows if row.panel === case.panel]
        all(==(first(counts)), counts) || error("checkpoint repetition is incomplete")
    end
    return checkpoint
end

function _sync_timing_file(path)
    open(path, "r") do io
        status = if Sys.iswindows()
            ccall(:_commit, Cint, (Cint,), fd(io))
        else
            ccall(:fsync, Cint, (Cint,), fd(io))
        end
        status == 0 || error("could not sync timing checkpoint to disk")
    end
end

"""Atomically replace a separate partial checkpoint after a complete repetition."""
function _save_timing_checkpoint(path, configuration, input_files, rows, sessions)
    input_files == _timing_input_files() || error("source inputs changed during timing run")
    payload = (; schema=SINGLE_PATH_CHECKPOINT_SCHEMA, status=:partial,
        configuration, input_files, rows, sessions)
    checkpoint = merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
    destination = Ptd._validate_data_output(path)
    mkpath(dirname(destination))
    temporary = Ptd._temporary_jld_path(destination)
    try
        JLD2.jldsave(temporary; checkpoint)
        _sync_timing_file(temporary)
        Base.rename(temporary, destination)
        if Sys.isunix()
            directory_fd = ccall(:open, Cint, (Cstring, Cint), dirname(destination), 0)
            directory_fd >= 0 || error("could not open timing checkpoint directory for sync")
            try
                ccall(:fsync, Cint, (Cint,), directory_fd) == 0 ||
                    error("could not sync timing checkpoint directory")
            finally
                ccall(:close, Cint, (Cint,), directory_fd)
            end
        end
    finally
        isfile(temporary) && rm(temporary)
    end
    return checkpoint.authentication_fingerprint
end

function run_single_path_timings(; repeats::Union{Nothing,Int}=nothing, smoke::Bool=false,
    panels=nothing, checkpoint_path::Union{Nothing,AbstractString}=nothing, resume::Bool=false)
    Threads.nthreads() == 1 || error("single-path timings require Julia --threads=1")
    BLAS.get_num_threads() == 1 || error("single-path timings require one BLAS thread")
    repeats_by_panel = timing_repetitions(; repeats, smoke)
    resume && isnothing(checkpoint_path) && throw(ArgumentError("resume requires a checkpoint path"))
    _timing_progress("Preparing timing configuration and source fingerprints")
    input_files = _timing_input_files()
    cases = timing_cases(; smoke)
    selected_panels = _selected_timing_panels(panels, cases)
    checkpoint_configuration = _timing_checkpoint_configuration(cases, smoke)
    stored_rows = [(panel=case.panel, method=method, samples=NamedTuple[])
        for case in cases for method in case.methods]
    sessions = String[]
    if resume
        isfile(checkpoint_path) || error("timing checkpoint does not exist: $checkpoint_path")
        checkpoint = _load_timing_checkpoint(checkpoint_path, input_files, checkpoint_configuration)
        stored_rows, sessions = checkpoint.rows, checkpoint.sessions
        _timing_progress("Resuming authenticated checkpoint: ", checkpoint_path)
    elseif !isnothing(checkpoint_path) && isfile(checkpoint_path)
        error("timing checkpoint already exists; use --resume or another output directory")
    end
    reused = Dict((row.panel, row.method) => length(row.samples) for row in stored_rows)
    session = string(Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS.sss"), "-", getpid(), "-", time_ns())
    push!(sessions, session)
    checkpoint_fingerprint = isnothing(checkpoint_path) ? nothing :
        _save_timing_checkpoint(checkpoint_path, checkpoint_configuration, input_files, stored_rows, sessions)
    rows = NamedTuple[]
    configurations = NamedTuple[]
    for (case_index, case) in enumerate(cases)
        case.panel in selected_panels || continue
        case_configuration = checkpoint_configuration.configurations[case_index]
        shocks = case_configuration.innovations
        measured_repeats = repeats_by_panel[case.panel]
        push!(configurations, merge(case_configuration, (; repeats=measured_repeats)))
        samples = Dict(row.method => row.samples for row in stored_rows if row.panel === case.panel)
        pending = [method for method in case.methods if length(samples[method]) < measured_repeats]
        # Resume never reuses a policy or cache. Every method with new work gets
        # a disposable full warmup in this process before any new measurements.
        for method in pending
            _logged_timing_sample(case, method, shocks, "Warmup")
        end
        for repetition in 1:measured_repeats
            measured_new_sample = false
            for offset in 0:length(case.methods)-1
                method = case.methods[mod1(repetition + offset, length(case.methods))]
                repetition <= length(samples[method]) && continue
                sample = _logged_timing_sample(case, method, shocks, "Timing repeat $repetition")
                push!(samples[method], merge(sample, (; session)))
                measured_new_sample = true
            end
            if measured_new_sample && !isnothing(checkpoint_path)
                checkpoint_fingerprint = _save_timing_checkpoint(checkpoint_path,
                    checkpoint_configuration, input_files, stored_rows, sessions)
            end
        end
        for method in case.methods
            retained = samples[method]
            _validate_timing_samples(retained)
            values = retained[1:measured_repeats]
            construction = getproperty.(values, :construction_seconds)
            evaluation = getproperty.(values, :evaluation_seconds)
            totals = getproperty.(values, :total_seconds)
            push!(rows, (; panel=case.panel, method, repeats=measured_repeats,
                available_repeats=length(retained), reused_repeats=min(reused[(case.panel, method)], measured_repeats),
                construction_seconds=median(construction),
                evaluation_seconds=median(evaluation), total_seconds=median(totals),
                minimum_total_seconds=minimum(totals), maximum_total_seconds=maximum(totals),
                utility=first(values).utility, samples=values))
        end
        _timing_progress("Completed panel: ", case.panel, "; requested repetitions ", measured_repeats)
    end
    input_files == _timing_input_files() || error("source inputs changed during timing run")
    configuration = (; smoke, repeats_override=repeats, repeats_by_panel, selected_panels,
        seed=SINGLE_PATH_TIMING_SEED, configurations,
        julia_threads=1, blas_threads=1, paths_per_measurement=1,
        warmup=:disposable_full_construction_and_path, warmups_per_method_per_measured_session=1,
        cache=:fresh_policy_each_repetition,
        total=:sum_of_separately_measured_construction_and_evaluation,
        summary=:median_of_each_component_and_repetition_total,
        single_repetition_summary=:single_measurement,
        sample_selection=:first_requested_repetitions_in_collection_order,
        resumed=resume, sessions, checkpoint_fingerprint,
        outside_timers=(:shock_generation, :explicit_policy_alignment,
            :output_validation, :file_io, :progress_logging, :checkpoint_io, :warmup, :premeasurement_gc),
        inside_timers=(:constructor_invariants, :action_feasibility_checks,
            :evaluator_action_audit, :on_demand_ptd_plan_construction))
    payload = (; schema=SINGLE_PATH_TIMING_SCHEMA, configuration, rows,
        raw_measurements=stored_rows, input_files,
        input_fingerprint=Ptd.stable_fingerprint(input_files),
        provenance=Ptd.simulation_metadata(configuration, pkgdir(Ptd)))
    return merge(payload, (; authentication_fingerprint=Ptd.stable_fingerprint(payload)))
end

function load_single_path_timings(path)
    artifact = JLD2.load(path, "artifact")
    artifact.schema in (SINGLE_PATH_TIMING_SCHEMA, "ptd-single-path-timing-v1") ||
        error("unknown single-path timing schema")
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    Ptd.stable_fingerprint(payload) == artifact.authentication_fingerprint ||
        error("single-path timing fingerprint mismatch")
    return artifact
end

function render_single_path_timings(artifact, output_path; panel_sources=nothing)
    labels = Dict(:deterministic => "Deterministic", :ptd => "PTD",
        :dcegm1 => "DC-EGM1", :dcegm2 => "DC-EGM2", :dcegm3 => "DC-EGM3",
        :dcegm4 => "DC-EGM4", :iid_baseline => "IID baseline", :iid_high_variance => "IID high variance",
        :ar1 => "AR(1)", :hsz => "HSZ", :ptd_sceq => "PTD-SCEQ", :ptd_osa => "PTD-OSA",
        :dcegm => "DC-EGM", :g2egm => "G2EGM")
    destination = Ptd._validate_data_output(output_path)
    mkpath(dirname(destination))
    open(destination, "w") do io
        println(io, "% Generated by Ptd.FixedCalibrationTiming via scripts/2_deterministic.jl, 3_iid.jl, 4_ar1.jl, or 5_hsz.jl --fixed-timing; do not edit.")
        if isnothing(panel_sources)
            println(io, "% Source fingerprint: ", artifact.input_fingerprint)
            println(io, "% Artifact fingerprint: ", artifact.authentication_fingerprint)
        else
            for source in panel_sources
                println(io, "% Timing panel: ", source.panel)
                println(io, "% Source fingerprint: ", source.source_fingerprint)
                println(io, "% Artifact fingerprint: ", source.artifact_fingerprint)
                println(io, "% Artifact SHA256: ", source.artifact_sha256)
                println(io, "% Artifact path: ",
                    Ptd.portable_path(source.source_path))
            end
        end
        println(io, "\\begin{tabular}{llrrrr}")
        println(io, "\\toprule")
        println(io, "Environment & Method & Repetitions & Construction (s) & One path (s) & Total (s) \\\\")
        println(io, "\\midrule")
        for row in artifact.rows
            println(io, labels[row.panel], " & ", labels[row.method], " & ", row.repeats, " & ",
                @sprintf("%.6f", row.construction_seconds), " & ",
                @sprintf("%.6f", row.evaluation_seconds), " & ",
                @sprintf("%.6f", row.total_seconds), " \\\\")
        end
        println(io, "\\bottomrule\n\\end{tabular}")
        artifact.configuration.smoke && println(io,
            "\\par\\smallskip{\\footnotesize Smoke test: reduced horizon and grids; not a paper benchmark.}")
        println(io, "\\par\\smallskip{\\footnotesize Repetitions count measured fresh-policy runs after an excluded full warmup. ",
            "Entries are medians when repetitions exceed one; a count of one is a single measurement. ",
            "All runs use one Julia thread and one BLAS thread; ",
            raw"$\beta=0.98$, $\gamma=1$, and initial cash 16. ",
            "Total is the median of the per-run sums of separately timed construction and evaluation; component medians need not sum to it. ",
            "Construction includes the complete deterministic PTD and DC-EGM/G2EGM backward solves. Stochastic PTD plans are built on demand during path evaluation. ",
            "Deterministic DC-EGM1--4 use the grids in the main deterministic comparison. ",
            "Warmup, shock generation, explicit alignment checks, and file I/O are outside the timers; constructor invariants and action checks are included.}")
    end
    return destination
end

"""Validate an unchanged timing panel or the explicitly requested DC-EGM4 grid refresh."""
function _validate_timing_panel_replacement(panel, original, replacement;
    allow_deterministic_grid_refresh::Bool=false)
    Ptd.stable_fingerprint(replacement) == Ptd.stable_fingerprint(original) && return nothing
    panel === :deterministic && allow_deterministic_grid_refresh ||
        throw(ArgumentError("replacement panel $panel changes calibration, grids, shocks, methods or repetitions"))
    old_grids = original.settings.asset_grids
    new_grids = replacement.settings.asset_grids
    current_grids = deterministic_timing_grids()
    old_grids.dcegm4 == iid_dcegm_asset_grid(400; asset_maximum=60_000.0) &&
        new_grids == current_grids &&
        all(method -> old_grids[method] == new_grids[method], (:dcegm1, :dcegm2, :dcegm3)) ||
        throw(ArgumentError("deterministic grid refresh permits only the declared 400-node DC-EGM4 change from [0,60000] to [0,10000]"))
    restored = merge(replacement, (; settings=merge(replacement.settings, (; asset_grids=old_grids))))
    Ptd.stable_fingerprint(restored) == Ptd.stable_fingerprint(original) ||
        throw(ArgumentError("deterministic grid refresh also changes calibration, shocks, methods or repetitions"))
    return nothing
end

"""Compose a timing table while retaining each authenticated panel's source."""
function render_composed_single_path_timings(base_path::AbstractString, replacements,
    output_path::AbstractString; allow_deterministic_grid_refresh::Bool=false)
    !allow_deterministic_grid_refresh || any(first(item) === :deterministic for item in replacements) ||
        throw(ArgumentError("a deterministic grid refresh requires a deterministic replacement artifact"))
    base = load_single_path_timings(base_path)
    production = !base.configuration.smoke
    base_summary = Ptd.load_single_path_timing_summary(base_path; require_production=production)
    base.authentication_fingerprint == base_summary.artifact_fingerprint ||
        error("base timing artifact changed while preparing the table")
    selected = Symbol.(base.configuration.selected_panels)
    sources = Dict(panel => base_summary for panel in selected)
    replaced = Set{Symbol}()
    for (panel, path) in replacements
        panel in selected || throw(ArgumentError("replacement panel $panel is absent from the base artifact"))
        panel in replaced && throw(ArgumentError("duplicate replacement panel $panel"))
        replacement = Ptd.load_single_path_timing_summary(path;
            required_panels=(panel,), require_production=production)
        _validate_timing_panel_replacement(panel, base_summary.configurations[panel],
            replacement.configurations[panel]; allow_deterministic_grid_refresh)
        sources[panel] = replacement
        push!(replaced, panel)
    end
    rows = [sources[panel].rows[(panel, method)] for panel in selected
        for method in base_summary.configurations[panel].methods]
    panel_sources = [(; panel, source_path=sources[panel].source_path,
        source_fingerprint=sources[panel].source_fingerprint,
        artifact_fingerprint=sources[panel].artifact_fingerprint,
        artifact_sha256=sources[panel].artifact_sha256) for panel in selected]
    # This rendering view is never saved as a timing artifact: each row remains
    # authenticated by its own underlying artifact and original raw samples.
    view = (; configuration=base.configuration, rows)
    render_single_path_timings(view, output_path; panel_sources)
    return (; table_path=abspath(output_path), panel_sources, rows)
end

"""Parse fixed-calibration timing options, with the calling experiment as the default panel."""
function timing_arguments(args; panels=nothing)
    repeats, smoke, output_dir, render_path = nothing, false, nothing, nothing
    resume = false
    allow_deterministic_grid_refresh = false
    replacements = Pair{Symbol,String}[]
    index = 1
    while index <= length(args)
        option = args[index]
        if option == "--smoke"
            smoke = true
        elseif option == "--resume"
            resume = true
        elseif option == "--allow-deterministic-grid-refresh"
            allow_deterministic_grid_refresh = true
        elseif option == "--replace-panel"
            index + 2 <= length(args) || error("--replace-panel requires PANEL ARTIFACT")
            push!(replacements, Symbol(args[index + 1]) => abspath(args[index + 2]))
            index += 2
        elseif option in ("--repeats", "--panels", "--output-dir", "--render-only")
            index < length(args) || error("missing value for $option")
            index += 1
            if option == "--repeats"
                repeats = parse(Int, args[index])
            elseif option == "--panels"
                panels = Symbol.(split(args[index], ','))
            elseif option == "--output-dir"
                output_dir = abspath(args[index])
            else
                render_path = abspath(args[index])
            end
        else
            error("unknown timing option: $option")
        end
        index += 1
    end
    isempty(replacements) || !isnothing(render_path) ||
        error("--replace-panel requires --render-only")
    !allow_deterministic_grid_refresh ||
        (!isnothing(render_path) && any(first(item) === :deterministic for item in replacements)) ||
        error("--allow-deterministic-grid-refresh requires --render-only and --replace-panel deterministic ARTIFACT")
    smoke && isnothing(output_dir) && error("--smoke requires --output-dir DIR")
    return (; repeats, smoke, output_dir, render_path, panels, resume,
        allow_deterministic_grid_refresh, replacements)
end

"""Run or render fixed-calibration timings from an experiment script."""
function main(args=ARGS; panels=nothing)
    options = timing_arguments(args; panels)
    (; repeats, smoke, output_dir, render_path, panels, resume,
        allow_deterministic_grid_refresh, replacements) = options
    BLAS.set_num_threads(1)
    if !isnothing(output_dir)
        Ptd._validate_data_output(joinpath(output_dir, "single_path_timings.tex"))
    end
    if isnothing(render_path)
        artifact_path = isnothing(output_dir) ? joinpath(pkgdir(Ptd), "data", "single-path-timings.jld2") :
            joinpath(output_dir, "single-path-timings.jld2")
        Ptd._validate_data_output(artifact_path)
        checkpoint_path = joinpath(dirname(artifact_path), "single-path-timings.partial.jld2")
        artifact = run_single_path_timings(; repeats, smoke, panels, checkpoint_path, resume)
        mkpath(dirname(artifact_path))
        JLD2.jldsave(artifact_path; artifact)
    else
        artifact_path = render_path
        artifact = load_single_path_timings(artifact_path)
        artifact.configuration.smoke && isnothing(output_dir) &&
            error("rendering smoke timings requires --output-dir DIR")
    end
    table_path = isnothing(output_dir) ?
        Ptd.tables_dir("single_path_timings.tex") :
        joinpath(output_dir, "single_path_timings.tex")
    if isempty(replacements)
        render_single_path_timings(artifact, table_path)
    else
        rendered = render_composed_single_path_timings(artifact_path, replacements, table_path;
            allow_deterministic_grid_refresh)
        _timing_progress("Composed single-path timing table: ", table_path)
        return (; artifact_path, table_path, panel_sources=rendered.panel_sources)
    end
    _timing_progress("Single-path timing artifact: ", artifact_path)
    _timing_progress("Single-path timing table: ", table_path)
    return (; artifact_path, table_path, artifact)
end
