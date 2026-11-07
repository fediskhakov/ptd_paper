using Dates
using JLD2
using SHA

const CONTEST_ARTIFACT_AUTHENTICATION_SCHEMA =
    "ptd-complete-contest-artifact-authentication-v1"
const CONTEST_UTILITY_PAYLOAD_SCHEMA =
    "ptd-complete-contest-utility-payload-v1"

function _contest_artifact_field(container, name::Symbol,
                                 location::AbstractString="artifact")
    if container isa NamedTuple
        hasproperty(container, name) && return getproperty(container, name)
    elseif container isa AbstractDict
        haskey(container, name) && return container[name]
        haskey(container, String(name)) && return container[String(name)]
    elseif hasproperty(container, name)
        return getproperty(container, name)
    end
    throw(ArgumentError("$location is missing required field $name"))
end

function _plain_comparison(comparison::PairedComparison)
    return (;
        method_a=comparison.method_a,
        method_b=comparison.method_b,
        n_paths=comparison.n_paths,
        confidence_level=comparison.confidence_level,
        mean_difference=comparison.mean_difference,
        standard_error=comparison.standard_error,
        ci_lower=comparison.ci_lower,
        ci_upper=comparison.ci_upper,
    )
end

function _plain_contest_summary(summary::ContestSummary, design_entry)
    return (;
        contest_id=summary.contest_id,
        panel=summary.panel,
        beta=Float64(_contest_artifact_field(design_entry, :beta,
            "outer design entry")),
        gamma=Float64(_contest_artifact_field(design_entry, :gamma,
            "outer design entry")),
        initial_cash=Float64(_contest_artifact_field(design_entry,
            :initial_cash, "outer design entry")),
        method_ids=copy(summary.method_ids),
        n_paths=summary.n_paths,
        confidence_level=summary.confidence_level,
        expected_utility=copy(summary.expected_utility),
        point_winners=copy(summary.point_winners),
        point_winner=summary.point_winner,
        resolved_winner=summary.resolved_winner,
        classification=summary.classification,
        comparisons=_plain_comparison.(summary.comparisons),
    )
end

function _contest_summaries(utilities, methods, panels, outer_design,
                            confidence_level::Float64)
    summaries = NamedTuple[]
    aggregates = NamedTuple[]
    for (panel_index, panel) in enumerate(panels)
        typed = ContestSummary[]
        for contest_index in eachindex(outer_design)
            contest_id = Int(_contest_artifact_field(
                outer_design[contest_index], :id, "outer_design[$contest_index]"))
            summary = summarize_contest(contest_id, panel, methods,
                view(utilities, :, :, contest_index, panel_index);
                level=confidence_level)
            push!(typed, summary)
            push!(summaries,
                _plain_contest_summary(summary, outer_design[contest_index]))
        end
        # Failed configurations have no fabricated utility observations. If all
        # configurations fail, the artifact carries only the completion audit.
        isempty(typed) || push!(aggregates, aggregate_outcomes(typed))
    end
    return summaries, aggregates
end

# Recomputing floating reductions under Julia's bounds-checking mode can change
# archived means and confidence limits by a few ULP. Authentication below still
# requires the exact hashes of every stored payload. Only this independent
# reconstruction check admits 64 ULP at unit-or-value scale for derived numeric
# statistics; configuration fields, winners, labels, and array shapes stay exact.
function _contest_summary_reproduces(stored, recomputed, field=nothing)
    if stored isa NamedTuple && recomputed isa NamedTuple
        keys(stored) == keys(recomputed) || return false
        return all(name -> _contest_summary_reproduces(
            getproperty(stored, name), getproperty(recomputed, name), name),
            keys(stored))
    elseif stored isa AbstractArray && recomputed isa AbstractArray
        axes(stored) == axes(recomputed) || return false
        return all(index -> _contest_summary_reproduces(
            stored[index], recomputed[index], field), eachindex(stored))
    elseif stored isa AbstractFloat && recomputed isa AbstractFloat &&
           field in (:expected_utility, :mean_difference, :standard_error,
                     :ci_lower, :ci_upper)
        isfinite(stored) && isfinite(recomputed) || return false
        scale = max(1.0, abs(Float64(stored)), abs(Float64(recomputed)))
        return abs(stored - recomputed) <= 64eps(scale)
    end
    return isequal(stored, recomputed)
end

function _utility_payload_fingerprint(utilities, methods, panels)
    all(isfinite, utilities) || throw(ArgumentError(
        "contest utilities must all be finite"))
    context = SHA.SHA2_256_CTX()
    header = string(CONTEST_UTILITY_PAYLOAD_SCHEMA, '|',
        join(size(utilities), ','), '|', join(String.(methods), ','), '|',
        join(String.(panels), ','), "|Float64|")
    SHA.update!(context, collect(codeunits(header)))
    words = Vector{UInt64}(undef, length(utilities))
    for (index, value) in enumerate(utilities)
        words[index] = hton(reinterpret(UInt64, Float64(value)))
    end
    SHA.update!(context, reinterpret(UInt8, words))
    return bytes2hex(SHA.digest!(context))
end

function _method_result(panel_result, method::Symbol)
    candidates = collect(_contest_artifact_field(
        panel_result, :methods, "panel result"))
    matching = filter(result -> Symbol(_contest_artifact_field(
        result, :method, "method result")) == method, candidates)
    length(matching) == 1 || throw(ArgumentError(
        "panel result must contain exactly one result for $method"))
    return only(matching)
end

function _utility_cube(welfare, methods::Vector{Symbol},
                       panels::Vector{Symbol}, n_paths::Int)
    results = collect(_contest_artifact_field(welfare, :results, "welfare"))
    successful = findall(result -> !(result isa ContestFailure), results)
    utilities = Array{Float64}(undef, n_paths, length(methods),
        length(successful), length(panels))
    for (contest_index, slot) in enumerate(successful)
        contest = results[slot]
        contest_id = Int(_contest_artifact_field(
            contest, :contest_id, "contest result"))
        contest_id == slot || throw(ArgumentError(
            "completed contests must retain their original slot identifiers"))
        panel_results = _contest_artifact_field(
            contest, :panels, "contest result")
        for (panel_index, panel) in enumerate(panels)
            panel_result = _contest_artifact_field(
                panel_results, panel, "contest panels")
            for (method_index, method) in enumerate(methods)
                result = _method_result(panel_result, method)
                evaluation = _contest_artifact_field(
                    result, :evaluation, "method result")
                values = Float64.(collect(_contest_artifact_field(
                    evaluation, :utility, "path evaluation")))
                length(values) == n_paths || throw(DimensionMismatch(
                    "method $method returned the wrong number of paths"))
                utilities[:, method_index, contest_index, panel_index] = values
            end
        end
    end
    return utilities
end

function _compact_run_audit(welfare, panels::Vector{Symbol})
    contests = NamedTuple[]
    failures = NamedTuple[]
    results = _contest_artifact_field(welfare, :results, "welfare")
    for (slot, contest) in enumerate(results)
        if contest isa ContestFailure
            contest.contest_id == slot || throw(ArgumentError(
                "failed contests must retain their original slot identifiers"))
            push!(failures, (; contest_id=contest.contest_id,
                exception=contest.exception, message=contest.message))
            continue
        end
        panel_rows = NamedTuple[]
        panel_results = _contest_artifact_field(
            contest, :panels, "contest result")
        for panel in panels
            panel_result = _contest_artifact_field(
                panel_results, panel, "contest panels")
            methods = [(;
                method=Symbol(_contest_artifact_field(
                    result, :method, "method result")),
                evaluation_fingerprint=String(_contest_artifact_field(
                    result, :evaluation_fingerprint, "method result")),
                quadrature=_contest_artifact_field(
                    result, :quadrature, "method result"),
                solver_diagnostics=_contest_artifact_field(
                    result, :solver_diagnostics, "method result"),
            ) for result in _contest_artifact_field(
                panel_result, :methods, "panel result")]
            policy_domain = if (panel_result isa NamedTuple &&
                    hasproperty(panel_result, :policy_domain)) ||
                    (panel_result isa AbstractDict &&
                     (haskey(panel_result, :policy_domain) ||
                      haskey(panel_result, "policy_domain")))
                _contest_artifact_field(panel_result, :policy_domain,
                    "panel result")
            else
                nothing
            end
            push!(panel_rows, (;
                panel,
                block_fingerprint=String(_contest_artifact_field(
                    panel_result, :block_fingerprint, "panel result")),
                policy_domain,
                methods,
            ))
        end
        push!(contests, (;
            contest_id=Int(_contest_artifact_field(
                contest, :contest_id, "contest result")),
            panels=panel_rows,
        ))
    end
    return (;
        completion=_contest_artifact_field(welfare, :sampler_audit, "welfare"),
        contests,
        failures,
    )
end

function _successful_contest_slots(run_audit, n_contests::Int)
    has_failures = hasproperty(run_audit, :failures) ||
        (run_audit isa AbstractDict &&
         (haskey(run_audit, :failures) || haskey(run_audit, "failures")))
    failures = has_failures ?
        _contest_artifact_field(run_audit, :failures, "run audit") : []
    failed_ids = Int[_contest_artifact_field(row, :contest_id, "failed contest")
        for row in failures]
    issorted(failed_ids) && length(unique(failed_ids)) == length(failed_ids) &&
        all(id -> 1 <= id <= n_contests, failed_ids) || throw(ArgumentError(
            "failed contest identifiers must be unique ordered original slots"))
    completion = _contest_artifact_field(run_audit, :completion, "run audit")
    successful = [id for id in 1:n_contests if !(id in failed_ids)]
    Int(_contest_artifact_field(completion, :completed_contests,
        "completion audit")) == length(successful) || throw(ArgumentError(
            "completion audit disagrees with successful contest count"))
    if hasproperty(completion, :failed_contests)
        Int(completion.failed_contests) == length(failed_ids) ||
            throw(ArgumentError("completion audit disagrees with failure count"))
        Tuple(completion.failed_slots) == Tuple(failed_ids) ||
            throw(ArgumentError("completion audit disagrees with failed slots"))
    end
    contest_ids = Int[_contest_artifact_field(row, :contest_id, "audit contest")
        for row in _contest_artifact_field(run_audit, :contests, "run audit")]
    contest_ids == successful || throw(ArgumentError(
        "run audit must identify every successful original slot"))
    return successful
end

function _embedded_timings(timing, panels::Vector{Symbol},
                           methods::Vector{Symbol}, completed_utc::String)
    thread_count = Int(_contest_artifact_field(
        timing, :thread_count, "timing result"))
    thread_count > 0 || throw(ArgumentError(
        "timing thread count must be positive"))
    rows = NamedTuple[]
    for total in _contest_artifact_field(timing, :totals, "timing result")
        panel = Symbol(_contest_artifact_field(total, :panel, "timing total"))
        method = Symbol(_contest_artifact_field(total, :method, "timing total"))
        seconds = Float64(_contest_artifact_field(
            total, :total_seconds, "timing total"))
        panel in panels && method in methods || throw(ArgumentError(
            "timing total has an unknown panel or method"))
        isfinite(seconds) && seconds >= 0.0 || throw(ArgumentError(
            "timing total must be finite and nonnegative"))
        identity = (panel, method, total_seconds=seconds, thread_count)
        push!(rows, merge(identity, (;
            status=:authenticated,
            completed_utc,
            relative_path="embedded/timings/$(panel)/$(method)",
            authentication_fingerprint=stable_fingerprint(identity),
        )))
    end
    expected = Set((panel, method) for panel in panels for method in methods)
    actual = Set((row.panel, row.method) for row in rows)
    actual == expected || throw(ArgumentError(
        "timing pass must contain exactly one total per panel and method"))
    return rows
end

function _artifact_authentication(metadata)
    return stable_fingerprint((;
        schema=CONTEST_ARTIFACT_AUTHENTICATION_SCHEMA,
        artifact_schema=String(_contest_artifact_field(metadata,
            :artifact_schema, "metadata")),
        status=Symbol(_contest_artifact_field(metadata, :status, "metadata")),
        completed_utc=String(_contest_artifact_field(metadata,
            :completed_utc, "metadata")),
        scientific_fingerprint=String(_contest_artifact_field(metadata,
            :scientific_fingerprint, "metadata")),
        run_fingerprint=String(_contest_artifact_field(metadata,
            :run_fingerprint, "metadata")),
        source_commit=String(_contest_artifact_field(metadata,
            :source_commit, "metadata")),
        input_tree_fingerprint=String(_contest_artifact_field(metadata,
            :input_tree_fingerprint, "metadata")),
        environment_fingerprint=String(_contest_artifact_field(metadata,
            :environment_fingerprint, "metadata")),
        confidence_level=Float64(_contest_artifact_field(metadata,
            :confidence_level, "metadata")),
        dimensions=(
            n_contests=Int(_contest_artifact_field(metadata,
                :n_contests, "metadata")),
            n_paths=Int(_contest_artifact_field(metadata, :n_paths, "metadata")),
            n_methods=Int(_contest_artifact_field(metadata,
                :n_methods, "metadata")),
            n_panels=Int(_contest_artifact_field(metadata,
                :n_panels, "metadata")),
        ),
        utility_payload_shape=Tuple(Int.(collect(_contest_artifact_field(
            metadata, :utility_payload_shape, "metadata")))),
        outer_design_fingerprint=String(_contest_artifact_field(metadata,
            :outer_design_fingerprint, "metadata")),
        simulation_metadata_fingerprint=String(_contest_artifact_field(metadata,
            :simulation_metadata_fingerprint, "metadata")),
        run_audit_fingerprint=String(_contest_artifact_field(metadata,
            :run_audit_fingerprint, "metadata")),
        utilities_fingerprint=String(_contest_artifact_field(metadata,
            :utilities_fingerprint, "metadata")),
        summaries_fingerprint=String(_contest_artifact_field(metadata,
            :summaries_fingerprint, "metadata")),
        aggregates_fingerprint=String(_contest_artifact_field(metadata,
            :aggregates_fingerprint, "metadata")),
        timings_fingerprint=String(_contest_artifact_field(metadata,
            :timings_fingerprint, "metadata")),
        result_fingerprint=String(_contest_artifact_field(metadata,
            :result_fingerprint, "metadata")),
    ))
end

"""
    build_contest_artifact(; artifact_schema, scientific_configuration,
                           welfare, timing, methods, panels, n_paths, ...)

Assemble one self-contained paper-contest artifact from a completed in-memory
welfare run and its aggregate timing pass. The outer design retains every
sampled configuration, including failures. The utility cube and statistical
summaries contain only successful configurations, ordered by their original
identifiers in `run_audit.contests`; `run_audit.failures` records unsuccessful
configurations without inventing utility values. Legacy all-success artifacts
remain readable. The artifact stores no checkpoints or timing sidecars.
"""
function build_contest_artifact(; artifact_schema::AbstractString,
    scientific_configuration, welfare, timing, methods, panels,
    n_paths::Integer, confidence_level::Real=0.95,
    completed_at::DateTime=now(UTC))

    method_ids = Symbol.(collect(methods))
    panel_ids = Symbol.(collect(panels))
    length(unique(method_ids)) == length(method_ids) || throw(ArgumentError(
        "contest methods must be unique"))
    length(unique(panel_ids)) == length(panel_ids) || throw(ArgumentError(
        "contest panels must be unique"))
    paths = Int(n_paths)
    paths >= 2 || throw(ArgumentError(
        "at least two paths are required for contest inference"))
    level = Float64(confidence_level)
    0.0 < level < 1.0 || throw(ArgumentError(
        "confidence_level must lie in (0,1)"))
    outer_design = collect(_contest_artifact_field(
        welfare, :outer_design, "welfare"))
    isempty(outer_design) && throw(ArgumentError(
        "a contest artifact requires a nonempty completed design"))
    length(_contest_artifact_field(welfare, :results, "welfare")) ==
        length(outer_design) || throw(DimensionMismatch(
            "welfare results must retain every original design slot"))
    run_audit = _compact_run_audit(welfare, panel_ids)
    successful = _successful_contest_slots(run_audit, length(outer_design))
    utilities = _utility_cube(welfare, method_ids, panel_ids, paths)
    summaries, aggregates = _contest_summaries(
        utilities, method_ids, panel_ids, outer_design[successful], level)
    completed_utc = Dates.format(completed_at,
        dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z"
    timings = _embedded_timings(
        timing, panel_ids, method_ids, completed_utc)

    scientific_fingerprint_value = stable_fingerprint(scientific_configuration)
    scientific_fingerprint_value == String(_contest_artifact_field(
        welfare, :scientific_fingerprint, "welfare")) || throw(ArgumentError(
            "welfare and scientific configuration fingerprints disagree"))
    run_fingerprint = String(_contest_artifact_field(
        welfare, :configuration_fingerprint, "welfare"))
    for (name, value) in ((:scientific_fingerprint, scientific_fingerprint_value),
                          (:run_fingerprint, run_fingerprint),
                          (:source_commit, String(_contest_artifact_field(
                              welfare, :source_commit, "welfare"))),
                          (:input_tree_fingerprint, String(_contest_artifact_field(
                              welfare, :input_tree_fingerprint, "welfare"))),
                          (:environment_fingerprint, String(_contest_artifact_field(
                              welfare, :environment_fingerprint, "welfare"))))
        String(_contest_artifact_field(timing, name, "timing result")) == value ||
            throw(ArgumentError("welfare and timing disagree on $name"))
    end
    String(_contest_artifact_field(timing,
        :outer_design_fingerprint, "timing result")) ==
        stable_fingerprint(outer_design) || throw(ArgumentError(
            "welfare and timing use different completed contest designs"))
    String(_contest_artifact_field(timing,
        :sampler_audit_fingerprint, "timing result")) ==
        stable_fingerprint(_contest_artifact_field(
            welfare, :sampler_audit, "welfare")) || throw(ArgumentError(
                "welfare and timing use different completion audits"))
    simulation = _contest_artifact_field(
        timing, :simulation_metadata, "timing result")
    utilities_fingerprint = _utility_payload_fingerprint(
        utilities, method_ids, panel_ids)
    summaries_fingerprint = stable_fingerprint(summaries)
    aggregates_fingerprint = stable_fingerprint(aggregates)
    timings_fingerprint = stable_fingerprint(timings)
    result_fingerprint = stable_fingerprint((;
        utilities_fingerprint,
        summaries_fingerprint,
        aggregates_fingerprint,
        timings_fingerprint,
    ))
    metadata_without_authentication = (;
        artifact_schema=String(artifact_schema),
        status=:complete,
        completed_utc,
        scientific_fingerprint=scientific_fingerprint_value,
        run_fingerprint,
        source_commit=String(_contest_artifact_field(
            welfare, :source_commit, "welfare")),
        input_tree_fingerprint=String(_contest_artifact_field(
            welfare, :input_tree_fingerprint, "welfare")),
        environment_fingerprint=String(_contest_artifact_field(
            welfare, :environment_fingerprint, "welfare")),
        confidence_level=level,
        n_contests=length(outer_design),
        n_paths=paths,
        n_methods=length(method_ids),
        n_panels=length(panel_ids),
        utility_payload_shape=Tuple(size(utilities)),
        outer_design_fingerprint=stable_fingerprint(outer_design),
        simulation_metadata_fingerprint=stable_fingerprint(simulation),
        run_audit_fingerprint=stable_fingerprint(run_audit),
        utilities_fingerprint,
        summaries_fingerprint,
        aggregates_fingerprint,
        timings_fingerprint,
        result_fingerprint,
    )
    metadata = merge(metadata_without_authentication, (;
        authentication_fingerprint=
            _artifact_authentication(metadata_without_authentication),
    ))
    return Dict{String,Any}(
        "metadata" => metadata,
        "scientific_configuration" => scientific_configuration,
        "outer_design" => outer_design,
        "methods" => method_ids,
        "panels" => panel_ids,
        "pathwise_utilities" => utilities,
        "contest_summaries" => summaries,
        "panel_aggregates" => aggregates,
        "timings" => timings,
        "run_audit" => run_audit,
        "simulation_metadata" => simulation,
    )
end

function _validate_contest_artifact(artifact; expected_schema=nothing)
    artifact isa AbstractDict || throw(ArgumentError(
        "contest artifact must be a dictionary"))
    metadata = _contest_artifact_field(artifact, :metadata)
    schema = String(_contest_artifact_field(metadata,
        :artifact_schema, "metadata"))
    isnothing(expected_schema) || schema == String(expected_schema) ||
        throw(ArgumentError("contest artifact schema mismatch"))
    Symbol(_contest_artifact_field(metadata, :status, "metadata")) ===
        :complete || throw(ArgumentError("contest artifact is incomplete"))
    methods = Symbol.(collect(_contest_artifact_field(artifact, :methods)))
    panels = Symbol.(collect(_contest_artifact_field(artifact, :panels)))
    outer_design = collect(_contest_artifact_field(artifact, :outer_design))
    utilities = _contest_artifact_field(artifact, :pathwise_utilities)
    utilities isa AbstractArray{<:Real,4} || throw(ArgumentError(
        "pathwise_utilities must be a four-dimensional real array"))
    run_audit = _contest_artifact_field(artifact, :run_audit)
    successful = _successful_contest_slots(run_audit, length(outer_design))
    expected_shape = (
        Int(_contest_artifact_field(metadata, :n_paths, "metadata")),
        length(methods), length(successful), length(panels))
    size(utilities) == expected_shape == Tuple(Int.(collect(
        _contest_artifact_field(metadata, :utility_payload_shape, "metadata")))) ||
        throw(DimensionMismatch("contest utility dimensions are inconsistent"))
    length(outer_design) == Int(_contest_artifact_field(
        metadata, :n_contests, "metadata")) || throw(DimensionMismatch(
            "outer design length is inconsistent"))
    [Int(_contest_artifact_field(entry, :id, "outer design entry"))
        for entry in outer_design] == collect(eachindex(outer_design)) ||
        throw(ArgumentError("outer design slot identifiers are not ordered"))
    configuration = _contest_artifact_field(
        artifact, :scientific_configuration)
    stable_fingerprint(configuration) == String(_contest_artifact_field(
        metadata, :scientific_fingerprint, "metadata")) || throw(ArgumentError(
            "scientific configuration fingerprint mismatch"))
    stable_fingerprint(outer_design) == String(_contest_artifact_field(
        metadata, :outer_design_fingerprint, "metadata")) || throw(ArgumentError(
            "outer design fingerprint mismatch"))
    utilities_fingerprint = _utility_payload_fingerprint(
        utilities, methods, panels)
    utilities_fingerprint == String(_contest_artifact_field(metadata,
        :utilities_fingerprint, "metadata")) || throw(ArgumentError(
            "utility payload fingerprint mismatch"))
    level = Float64(_contest_artifact_field(
        metadata, :confidence_level, "metadata"))
    summaries, aggregates = _contest_summaries(
        utilities, methods, panels, outer_design[successful], level)
    stored_summaries = _contest_artifact_field(artifact, :contest_summaries)
    stored_aggregates = _contest_artifact_field(artifact, :panel_aggregates)
    stable_fingerprint(stored_summaries) ==
        String(_contest_artifact_field(metadata,
            :summaries_fingerprint, "metadata")) || throw(ArgumentError(
                "contest summary fingerprint mismatch"))
    _contest_summary_reproduces(stored_summaries, summaries) ||
        throw(ArgumentError("stored contest summaries do not reproduce"))
    stable_fingerprint(stored_aggregates) == stable_fingerprint(aggregates) ==
        String(_contest_artifact_field(metadata,
            :aggregates_fingerprint, "metadata")) || throw(ArgumentError(
                "panel aggregate fingerprint mismatch"))
    timings = _contest_artifact_field(artifact, :timings)
    stable_fingerprint(timings) == String(_contest_artifact_field(metadata,
        :timings_fingerprint, "metadata")) || throw(ArgumentError(
            "timing fingerprint mismatch"))
    expected_timing_keys = Set((panel, method)
        for panel in panels for method in methods)
    Set((Symbol(_contest_artifact_field(row, :panel, "timing row")),
         Symbol(_contest_artifact_field(row, :method, "timing row")))
        for row in timings) == expected_timing_keys || throw(ArgumentError(
            "contest timings are incomplete"))
    run_audit = _contest_artifact_field(artifact, :run_audit)
    stable_fingerprint(run_audit) == String(_contest_artifact_field(metadata,
        :run_audit_fingerprint, "metadata")) || throw(ArgumentError(
            "run audit fingerprint mismatch"))
    simulation = _contest_artifact_field(artifact, :simulation_metadata)
    stable_fingerprint(simulation) == String(_contest_artifact_field(metadata,
        :simulation_metadata_fingerprint, "metadata")) || throw(ArgumentError(
            "simulation metadata fingerprint mismatch"))
    result_fingerprint = stable_fingerprint((;
        utilities_fingerprint,
        summaries_fingerprint=stable_fingerprint(stored_summaries),
        aggregates_fingerprint=stable_fingerprint(stored_aggregates),
        timings_fingerprint=stable_fingerprint(timings),
    ))
    result_fingerprint == String(_contest_artifact_field(metadata,
        :result_fingerprint, "metadata")) || throw(ArgumentError(
            "contest result fingerprint mismatch"))
    _artifact_authentication(metadata) == String(_contest_artifact_field(
        metadata, :authentication_fingerprint, "metadata")) ||
        throw(ArgumentError("contest artifact authentication mismatch"))
    return artifact
end

"""Load and fully authenticate one self-contained contest artifact."""
function load_contest_artifact(path::AbstractString; expected_schema=nothing)
    isfile(path) || throw(ArgumentError(
        "contest artifact does not exist: $path"))
    artifact = JLD2.jldopen(path, "r") do file
        Dict{String,Any}(String(name) => file[String(name)] for name in keys(file))
    end
    return _validate_contest_artifact(artifact; expected_schema)
end

"""
    publish_contest_artifact(path, artifact; compress=true)

Validate and atomically replace the requested final artifact. Only this final
file is published; interrupted runs leave no partial contest data behind.
"""
function publish_contest_artifact(path::AbstractString, artifact::AbstractDict;
                                  compress::Bool=true)
    _validate_contest_artifact(artifact)
    destination = _validate_data_output(path)
    mkpath(dirname(destination))
    isdir(destination) && throw(ArgumentError(
        "contest artifact destination is a directory: $destination"))
    temporary = _temporary_jld_path(destination)
    try
        JLD2.jldopen(temporary, "w"; compress) do file
            for (name, value) in artifact
                file[name] = value
            end
        end
        schema = String(_contest_artifact_field(
            _contest_artifact_field(artifact, :metadata), :artifact_schema))
        load_contest_artifact(temporary; expected_schema=schema)
        mv(temporary, destination; force=true)
        load_contest_artifact(destination; expected_schema=schema)
    finally
        isfile(temporary) && rm(temporary)
    end
    return destination
end
