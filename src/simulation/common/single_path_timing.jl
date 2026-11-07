using JLD2
using Statistics: mean, median

const SINGLE_PATH_SUMMARY_SCHEMA = "ptd-single-path-timing-v2"
const _SINGLE_PATH_METHODS = (
    deterministic=(:ptd, :dcegm1, :dcegm2, :dcegm3, :dcegm4),
    iid_baseline=(:ptd_sceq, :ptd_osa, :dcegm),
    iid_high_variance=(:ptd_sceq, :ptd_osa, :dcegm),
    ar1=(:ptd_sceq, :ptd_osa, :g2egm), hsz=(:ptd_sceq, :ptd_osa, :g2egm))

function _single_path_source_files()
    paths = String[]
    for (directory, _, files) in walkdir(joinpath(PROJECT_ROOT, "src"))
        append!(paths, [joinpath(directory, file) for file in files if endswith(file, ".jl")])
    end
    append!(paths, [joinpath(PROJECT_ROOT, "scripts", script)
        for script in ("2_deterministic.jl", "3_iid.jl", "4_ar1.jl", "5_hsz.jl")])
    append!(paths, [joinpath(PROJECT_ROOT, "Project.toml"),
        joinpath(PROJECT_ROOT, "Manifest.toml")])
    return [(path=relpath(path, PROJECT_ROOT), sha256=bytes2hex(sha256(read(path))))
        for path in sort!(unique!(paths))]
end

function _validate_single_path_samples(samples, sessions)
    for sample in samples
        all(isfinite, (sample.construction_seconds, sample.evaluation_seconds,
            sample.total_seconds, sample.utility)) &&
            sample.construction_seconds >= 0 && sample.evaluation_seconds >= 0 &&
            sample.total_seconds == sample.construction_seconds + sample.evaluation_seconds ||
            throw(ArgumentError("invalid single-path timing sample"))
        sample.session in sessions || throw(ArgumentError("unregistered timing session"))
    end
    isempty(samples) || all(s -> isequal(s.utility, first(samples).utility), samples) ||
        throw(ArgumentError("single-path utility differs across timing repetitions"))
end

"""
    load_single_path_timing_summary(path; required_panels=(),
        require_current_source=false, require_production=true)

Authenticate a completed fixed-calibration timing artifact and verify its counts and medians
against the retained raw measurements. Compute mean total time from the same
selected raw samples, retaining `total_seconds` as the historical median alias.
Return rows keyed by `(panel, method)`
and the artifact/source fingerprints. Historical rendering validates recorded
provenance; `require_current_source=true` also matches every current input file.
"""
function load_single_path_timing_summary(path::AbstractString;
    required_panels=(), require_current_source::Bool=false,
    require_production::Bool=true)
    source_path = abspath(path)
    artifact_sha256 = bytes2hex(sha256(read(source_path)))
    artifact = JLD2.load(source_path, "artifact")
    artifact.schema == SINGLE_PATH_SUMMARY_SCHEMA || throw(ArgumentError(
        "paper timing summaries require a completed fixed-calibration v2 artifact"))
    payload = (; (name => getproperty(artifact, name) for name in keys(artifact)
        if name !== :authentication_fingerprint)...)
    stable_fingerprint(payload) == artifact.authentication_fingerprint ||
        throw(ArgumentError("single-path timing authentication mismatch"))
    config = artifact.configuration
    config.julia_threads == config.blas_threads == config.paths_per_measurement == 1 ||
        throw(ArgumentError("paper timings require one Julia thread, one BLAS thread, and one path"))
    config.warmup === :disposable_full_construction_and_path &&
        config.warmups_per_method_per_measured_session == 1 &&
        config.cache === :fresh_policy_each_repetition &&
        config.total === :sum_of_separately_measured_construction_and_evaluation &&
        config.summary === :median_of_each_component_and_repetition_total &&
        config.sample_selection === :first_requested_repetitions_in_collection_order ||
        throw(ArgumentError("unsupported single-path measurement protocol"))
    all(name -> name in config.outside_timers,
        (:shock_generation, :explicit_policy_alignment, :output_validation, :file_io, :warmup)) ||
        throw(ArgumentError("single-path timings must exclude warmup and validation"))
    require_production && config.smoke && throw(ArgumentError(
        "smoke timings cannot populate a paper accuracy table"))
    !isempty(config.sessions) && length(unique(config.sessions)) == length(config.sessions) ||
        throw(ArgumentError("invalid timing session registration"))
    all(s -> s isa AbstractString && !isempty(s), config.sessions) ||
        throw(ArgumentError("invalid timing session identity"))
    provenance = artifact.provenance
    provenance.configuration == configuration_metadata(config) ||
        throw(ArgumentError("timing provenance configuration mismatch"))
    provenance.runtime.threads == 1 && !isempty(provenance.runtime.julia_version) &&
        !isempty(provenance.runtime.cpu_name) && !isempty(provenance.source.commit) &&
        !isempty(provenance.packages) || throw(ArgumentError("incomplete timing runtime/source provenance"))
    source_files = artifact.input_files
    stable_fingerprint(source_files) == artifact.input_fingerprint ||
        throw(ArgumentError("timing source fingerprint mismatch"))
    source_paths = getproperty.(source_files, :path)
    length(unique(source_paths)) == length(source_paths) &&
        all(p -> !isabspath(p) && !(".." in splitpath(p)), source_paths) &&
        all(f -> occursin(r"^[0-9a-f]{64}$", f.sha256), source_files) ||
        throw(ArgumentError("invalid timing input-file inventory"))
    all(p -> p in source_paths, ("src/Ptd.jl", "scripts/2_deterministic.jl",
        "Project.toml", "Manifest.toml")) &&
        any(p -> p in source_paths, ("scripts/6_timing.jl",
            "src/simulation/common/fixed_calibration_timing.jl")) ||
        throw(ArgumentError("timing input-file inventory is incomplete"))
    require_current_source && source_files != _single_path_source_files() &&
        throw(ArgumentError("timing artifact source inputs differ from the current source"))

    selected = Symbol.(config.selected_panels)
    !isempty(selected) && length(unique(selected)) == length(selected) &&
        all(p -> haskey(_SINGLE_PATH_METHODS, p), selected) ||
        throw(ArgumentError("invalid timing panel registration"))
    all(p -> Symbol(p) in selected, required_panels) ||
        throw(ArgumentError("timing artifact is missing a required panel"))
    configurations = Dict{Symbol,Any}()
    for case in config.configurations
        case.panel in selected && !haskey(configurations, case.panel) ||
            throw(ArgumentError("duplicate or unexpected timing configuration"))
        Tuple(case.methods) == _SINGLE_PATH_METHODS[case.panel] &&
            case.repeats == config.repeats_by_panel[case.panel] && case.repeats >= 1 ||
            throw(ArgumentError("timing configuration method/count mismatch"))
        size(case.innovations, 1) == 1 && size(case.innovations, 2) == case.horizon - 1 &&
            all(isfinite, case.innovations) &&
            stable_fingerprint(case.innovations) == case.shock_fingerprint ||
            throw(ArgumentError("timing configuration shock mismatch"))
        if require_production
            (case.beta, case.gamma, case.return_rate, case.cash_floor,
                case.horizon, case.initial_cash) == (0.98, 1.0, 0.1, 3.0, 50, 16.0) ||
                throw(ArgumentError("paper timings must use the fixed 50-period calibration"))
        end
        configurations[case.panel] = case
    end
    Set(keys(configurations)) == Set(selected) ||
        throw(ArgumentError("timing configurations do not cover the selected panels"))
    raw = Dict{Tuple{Symbol,Symbol},Any}()
    for row in artifact.raw_measurements
        key = (row.panel, row.method)
        haskey(_SINGLE_PATH_METHODS, row.panel) &&
            row.method in _SINGLE_PATH_METHODS[row.panel] && !haskey(raw, key) ||
            throw(ArgumentError("duplicate or unexpected raw timing row"))
        _validate_single_path_samples(row.samples, config.sessions)
        raw[key] = row.samples
    end
    expected_raw = Set((panel, method) for (panel, methods) in pairs(_SINGLE_PATH_METHODS)
        for method in methods)
    Set(keys(raw)) == expected_raw || throw(ArgumentError("raw timing registration is incomplete"))
    for (panel, methods) in pairs(_SINGLE_PATH_METHODS)
        length(unique(length(raw[(panel, method)]) for method in methods)) == 1 ||
            throw(ArgumentError("raw timing repetitions are incomplete within a panel"))
    end
    rows = Dict{Tuple{Symbol,Symbol},NamedTuple}()
    for row in artifact.rows
        key = (row.panel, row.method)
        row.panel in selected && haskey(raw, key) && !haskey(rows, key) ||
            throw(ArgumentError("duplicate or unexpected timing summary row"))
        samples = raw[key]
        row.repeats == configurations[row.panel].repeats == length(row.samples) &&
            row.available_repeats == length(samples) >= row.repeats &&
            0 <= row.reused_repeats <= row.repeats &&
            row.samples == samples[1:row.repeats] ||
            throw(ArgumentError("timing counts or selected raw-sample prefix mismatch"))
        for component in (:construction_seconds, :evaluation_seconds, :total_seconds)
            getproperty(row, component) == median(getproperty.(row.samples, component)) ||
                throw(ArgumentError("timing median does not match raw measurements"))
        end
        row.minimum_total_seconds == minimum(s.total_seconds for s in row.samples) &&
            row.maximum_total_seconds == maximum(s.total_seconds for s in row.samples) &&
            isequal(row.utility, first(row.samples).utility) ||
            throw(ArgumentError("timing summary range or utility mismatch"))
        rows[key] = (; row.panel, row.method, row.repeats, row.construction_seconds,
            row.evaluation_seconds, row.total_seconds,
            mean_total_seconds=mean(s.total_seconds for s in row.samples),
            median_total_seconds=row.total_seconds)
    end
    Set(keys(rows)) == Set((panel, method) for panel in selected
        for method in _SINGLE_PATH_METHODS[panel]) ||
        throw(ArgumentError("timing summaries do not cover the selected methods"))
    bytes2hex(sha256(read(source_path))) == artifact_sha256 ||
        throw(ArgumentError("timing artifact changed while it was being read"))
    return (; rows, configurations, artifact_fingerprint=artifact.authentication_fingerprint,
        source_fingerprint=artifact.input_fingerprint, artifact_sha256, source_files,
        provenance_fingerprint=stable_fingerprint(provenance), source_path)
end
