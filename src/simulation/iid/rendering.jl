import ..Ptd: _validate_data_output
using Dates
using JLD2
using Printf
using SHA
using Statistics: quantile
using ..Ptd: stable_fingerprint, load_single_path_timing_summary,
    tables_dir


export RENDER_CONTEST_SCHEMA_VERSION,
       consumption_equivalent_bps,
       render_output_directory,
       load_render_manifest,
       render_iid_contest,
       render_iid_paper_table,
       render_iid_ce_gap_table

const RENDER_CONTEST_SCHEMA_VERSION = "ptd-iid-contest-render-v3"
const RENDER_AUTHENTICATION_SCHEMA_VERSION =
    "ptd-iid-contest-render-authentication-v3"
const RENDER_CONFIGURATION_SCHEMA_VERSION =
    "ptd-iid-contest-render-configuration-v3"
const CE_QUANTILE_PROBABILITIES = (0.05, 0.25, 0.50, 0.75, 0.95)
const RENDER_MANIFEST_FILENAME = "iid_contest_render_manifest.jld2"
const OUTCOME_TABLE_FILENAME = "iid_contest_outcome_shares.tex"
const CE_GAP_TABLE_FILENAME = "iid_contest_ce_gaps.tex"
const TIMING_TABLE_FILENAME = "iid_contest_timings.tex"
const _RENDER_UTC_FORMAT = dateformat"yyyy-mm-ddTHH:MM:SS.sss"

_render_utc_string(value::DateTime) = Dates.format(value, _RENDER_UTC_FORMAT) * "Z"
_render_file_mtime_utc(path::AbstractString) =
    _render_utc_string(Dates.unix2datetime(stat(path).mtime))
_sha256_file(path::AbstractString) = bytes2hex(SHA.sha256(read(path)))

function _render_field(container, name::Symbol, location::AbstractString)
    if container isa NamedTuple
        hasproperty(container, name) || throw(ArgumentError(
            "$location is missing required field $name"))
        return getproperty(container, name)
    elseif container isa AbstractDict
        haskey(container, name) && return container[name]
        haskey(container, String(name)) && return container[String(name)]
    end
    throw(ArgumentError("$location is missing required field $name"))
end

_render_field(container, name::Symbol) = _render_field(container, name, "artifact")

function _discount_mass(beta::Float64, horizon::Int)
    isfinite(beta) && 0.0 < beta <= 1.0 || throw(ArgumentError(
        "beta must be finite and lie in (0,1]"))
    horizon >= 1 || throw(ArgumentError("horizon must be positive"))
    mass = 0.0
    discount = 1.0
    for _ in 1:horizon
        mass += discount
        discount *= beta
    end
    isfinite(mass) && mass > 0.0 || throw(ArgumentError(
        "discount mass is not finite and positive"))
    return mass
end

"""
    consumption_equivalent_bps(value_a, value_b, beta, gamma, horizon)

Return the uniform proportional consumption scale, in basis points, that must
be applied to method B's complete consumption stream to match method A's
expected discounted utility. Flow utility is the shifted CRRA normalization
`(c^(1-gamma)-1)/(1-gamma)`. The exact logarithmic formula is used at
`gamma == 1`.
"""
function consumption_equivalent_bps(value_a::Real, value_b::Real,
                                    beta::Real, gamma::Real,
                                    horizon::Integer)
    utility_a = Float64(value_a)
    utility_b = Float64(value_b)
    discount = Float64(beta)
    curvature = Float64(gamma)
    dates = Int(horizon)
    all(isfinite, (utility_a, utility_b, discount, curvature)) ||
        throw(ArgumentError("CEG inputs must be finite"))
    curvature > 0.0 || throw(ArgumentError("gamma must be positive"))
    mass = _discount_mass(discount, dates)

    log_scale = if curvature == 1.0
        (utility_a - utility_b) / mass
    else
        exponent = 1.0 - curvature
        power_a = mass + exponent * utility_a
        power_b = mass + exponent * utility_b
        power_a > 0.0 && power_b > 0.0 || throw(ArgumentError(
            "shifted-CRRA welfare implies a nonpositive consumption-power index"))
        # This log1p form is stable when gamma is close to one.
        relative_change = exponent * (utility_a - utility_b) / power_b
        relative_change > -1.0 || throw(ArgumentError(
            "invalid shifted-CRRA consumption-power ratio"))
        log1p(relative_change) / exponent
    end
    result = 10_000.0 * expm1(log_scale)
    isfinite(result) || throw(ArgumentError(
        "consumption-equivalent transformation overflowed Float64"))
    return result
end

function _consumption_equivalent_interval_bps(
    value_b::Real, utility_ci_lower::Real, utility_ci_upper::Real,
    beta::Real, gamma::Real, horizon::Integer)

    baseline = Float64(value_b)
    lower_gap = Float64(utility_ci_lower)
    upper_gap = Float64(utility_ci_upper)
    all(isfinite, (baseline, lower_gap, upper_gap)) || throw(ArgumentError(
        "paired utility interval inputs must be finite"))
    lower_gap <= upper_gap || throw(ArgumentError(
        "paired utility interval endpoints are reversed"))
    lower_bps = consumption_equivalent_bps(
        baseline + lower_gap, baseline, beta, gamma, horizon)
    upper_bps = consumption_equivalent_bps(
        baseline + upper_gap, baseline, beta, gamma, horizon)
    all(isfinite, (lower_bps, upper_bps)) && lower_bps <= upper_bps ||
        throw(ArgumentError(
            "paired CEG interval endpoints are nonfinite or reversed"))
    return (lower_bps=lower_bps, upper_bps=upper_bps)
end

function _panel_specifications(records, panels::Vector{Symbol},
                               location::AbstractString)
    length(unique(panels)) == length(panels) && !isempty(panels) ||
        throw(ArgumentError("$location requires distinct panel identifiers"))
    by_panel = Dict{Symbol,Float64}()
    for (index, record) in enumerate(records)
        record_location = "$location[$index]"
        panel = Symbol(_render_field(record, :id, record_location))
        haskey(by_panel, panel) && throw(ArgumentError(
            "$location contains duplicate panel $panel"))
        sigma = Float64(_render_field(record, :sigma, record_location))
        isfinite(sigma) && sigma >= 0.0 || throw(ArgumentError(
            "$record_location has an invalid sigma"))
        by_panel[panel] = sigma
    end
    Set(keys(by_panel)) == Set(panels) || throw(ArgumentError(
        "$location does not cover the rendered panels exactly"))
    return Tuple((id=panel, sigma=by_panel[panel]) for panel in panels)
end

function _panel_sigma(panel::Symbol, specifications)
    for specification in specifications
        specification.id === panel && return specification.sigma
    end
    throw(ArgumentError("missing sigma for panel $panel"))
end

function _render_configuration(horizon::Int, methods::Vector{Symbol},
                               panels::Vector{Symbol}, panel_sigmas,
                               confidence_level::Float64,
                               timing_paths::Int)
    return (;
        schema=RENDER_CONFIGURATION_SCHEMA_VERSION,
        role=:processed_iid_contest_tables,
        source_role=:authenticated_processed_artifact_only,
        horizon,
        methods=Tuple(methods),
        panels=Tuple(panels),
        panel_sigmas,
        timing_paths,
        timing_average=:amortized_seconds_per_simulated_path,
        paired_confidence_level=confidence_level,
        crra_normalization=:shifted,
        ce_orientation=:scale_method_b_to_match_method_a,
        ce_unit=:basis_points,
        gamma_one_formula=:log,
        paired_interval_endpoint_reference=:method_b_expected_utility,
        paired_interval_endpoint_kind=
            :plugin_ce_scaled_paired_utility_gap_interval,
        paired_interval_has_nominal_ce_coverage=false,
        unattainable_plugin_endpoint_policy=:refuse_render,
        paired_interval_endpoint_summary=:median_across_contests,
        paired_interval_summary_is_ci_for_median=false,
        quantile_probabilities=CE_QUANTILE_PROBABILITIES,
        implementation_file_sha256=_sha256_file(@__FILE__),
        table_files=(
            outcome_shares=OUTCOME_TABLE_FILENAME,
            ce_gaps=CE_GAP_TABLE_FILENAME,
            timings=TIMING_TABLE_FILENAME,
        ),
    )
end

function _method_pairs(methods::Vector{Symbol})
    return [(methods[left], methods[right])
        for left in 1:length(methods)-1 for right in left+1:length(methods)]
end

"""Sample denominators and selection history, including legacy artifacts."""
function _completion_status(artifact)
    metadata = _render_field(artifact, :metadata)
    target = Int(_render_field(metadata, :n_contests))
    has_audit = (artifact isa AbstractDict &&
        (haskey(artifact, :run_audit) || haskey(artifact, "run_audit"))) ||
        hasproperty(artifact, :run_audit)
    if !has_audit
        return (; target_contests=target, completed_contests=target,
            failed_contests=nothing, original_failed_contests=nothing,
            replacement_count=nothing, conditional_on_success=true)
    end
    completion = _render_field(_render_field(artifact, :run_audit), :completion)
    completed = Int(_render_field(completion, :completed_contests))
    replacements = Int(_render_field(completion, :replacement_count))
    original_failed = if hasproperty(completion, :original_failed_contests)
        Int(completion.original_failed_contests)
    else
        rounds = _render_field(completion, :rounds)
        isempty(rounds) ? 0 : length(_render_field(first(rounds), :failed_slots))
    end
    failed = target - completed
    return (; target_contests=target, completed_contests=completed,
        failed_contests=failed, original_failed_contests=original_failed,
        replacement_count=replacements,
        conditional_on_success=original_failed > 0 || replacements > 0 || failed > 0)
end

function _render_sample_note(io, sample; timing=true)
    print(io, "\\par\\smallskip{\\footnotesize \\emph{Sample:} ",
        sample.completed_contests, " completed configurations; original design size ",
        sample.target_contests, ". ")
    if isnothing(sample.original_failed_contests)
        print(io, "The archived artifact has no completion audit; the original solver-failure count is unavailable. ")
    else
        print(io, "Original solver failures: ", sample.original_failed_contests,
            "/", sample.target_contests, ". ")
        sample.replacement_count > 0 && print(io,
            sample.replacement_count, " replacement configurations are included. ")
    end
    if sample.completed_contests == 0
        print(io, "No welfare comparisons or per-path timing estimates are available.")
    else
        print(io, "Welfare shares use the ", sample.completed_contests,
            " completed configurations as their denominator")
        sample.conditional_on_success && print(io,
            " and are conditional on successful solver completion")
        print(io, ". ")
        timing && print(io, "Reported timing covers these completed configurations only. ")
        print(io, "Unresolved comparisons do not establish equivalent accuracy.")
    end
    println(io, "}")
end

function _render_no_success_table(artifact, output_path::AbstractString)
    destination = _validate_data_output(output_path)
    mkpath(dirname(destination))
    open(destination, "w") do io
        println(io, "% Generated from authenticated contest failure audit.")
        _render_sample_note(io, _completion_status(artifact))
    end
    return destination
end

function _aggregate_by_panel(artifact, panels::Vector{Symbol})
    aggregates = Dict{Symbol,Any}()
    for (index, aggregate) in enumerate(_render_field(artifact, :panel_aggregates))
        panel = Symbol(_render_field(aggregate, :panel,
            "panel_aggregates[$index]"))
        haskey(aggregates, panel) && throw(ArgumentError(
            "processed artifact contains duplicate aggregate for panel $panel"))
        aggregates[panel] = aggregate
    end
    Set(keys(aggregates)) == Set(panels) || throw(ArgumentError(
        "processed aggregates do not cover the configured panels"))
    return aggregates
end

function _outcome_rows(artifact, methods::Vector{Symbol},
                       panels::Vector{Symbol})
    aggregates = _aggregate_by_panel(artifact, panels)
    rows = NamedTuple[]
    for panel in panels
        aggregate = aggregates[panel]
        n_contests = Int(_render_field(aggregate, :n_contests,
            "aggregate $panel"))
        n_contests > 0 || throw(ArgumentError(
            "aggregate $panel has no contests"))
        for method in methods
            point_counts = _render_field(aggregate, :point_counts,
                "aggregate $panel")
            point_shares = _render_field(aggregate, :point_shares,
                "aggregate $panel")
            resolved_counts = _render_field(aggregate, :resolved_counts,
                "aggregate $panel")
            resolved_shares = _render_field(aggregate, :resolved_win_shares,
                "aggregate $panel")
            loss_counts = _render_field(aggregate, :resolved_loss_counts,
                "aggregate $panel")
            loss_shares = _render_field(aggregate, :resolved_loss_shares,
                "aggregate $panel")
            push!(rows, (;
                panel,
                method,
                n_contests,
                sample=_completion_status(artifact),
                point_winner_count=Int(_render_field(point_counts, method,
                    "aggregate $panel point counts")),
                point_winner_share=Float64(_render_field(point_shares, method,
                    "aggregate $panel point shares")),
                resolved_win_count=Int(_render_field(resolved_counts, method,
                    "aggregate $panel resolved counts")),
                resolved_win_share=Float64(_render_field(resolved_shares, method,
                    "aggregate $panel resolved shares")),
                resolved_loss_count=Int(_render_field(loss_counts, method,
                    "aggregate $panel resolved loss counts")),
                resolved_loss_share=Float64(_render_field(loss_shares, method,
                    "aggregate $panel resolved loss shares")),
                mc_unresolved_count=Int(_render_field(aggregate, :mc_unresolved,
                    "aggregate $panel")),
                mc_unresolved_share=Float64(_render_field(
                    aggregate, :mc_unresolved_share, "aggregate $panel")),
                point_tie_count=Int(_render_field(aggregate, :point_ties,
                    "aggregate $panel")),
                point_tie_share=Float64(_render_field(
                    aggregate, :point_tie_share, "aggregate $panel")),
                unresolved_count=Int(_render_field(aggregate, :unresolved,
                    "aggregate $panel")),
                unresolved_share=Float64(_render_field(
                    aggregate, :unresolved_share, "aggregate $panel")),
            ))
        end
    end
    return rows
end

function _oriented_comparison(summary, method_a::Symbol, method_b::Symbol,
                              location::AbstractString)
    for comparison in _render_field(summary, :comparisons, location)
        stored_a = Symbol(_render_field(comparison, :method_a, location))
        stored_b = Symbol(_render_field(comparison, :method_b, location))
        if stored_a == method_a && stored_b == method_b
            return (;
                confidence_level=Float64(_render_field(
                    comparison, :confidence_level, location)),
                ci_lower=Float64(_render_field(comparison, :ci_lower, location)),
                ci_upper=Float64(_render_field(comparison, :ci_upper, location)),
            )
        elseif stored_a == method_b && stored_b == method_a
            return (;
                confidence_level=Float64(_render_field(
                    comparison, :confidence_level, location)),
                ci_lower=-Float64(_render_field(comparison, :ci_upper, location)),
                ci_upper=-Float64(_render_field(comparison, :ci_lower, location)),
            )
        end
    end
    throw(ArgumentError(
        "$location is missing comparison $method_a versus $method_b"))
end

function _summaries_by_panel(artifact, panels::Vector{Symbol})
    result = Dict(panel => Any[] for panel in panels)
    for (index, summary) in enumerate(_render_field(artifact, :contest_summaries))
        panel = Symbol(_render_field(summary, :panel, "contest_summaries[$index]"))
        haskey(result, panel) || throw(ArgumentError(
            "processed summary contains unknown panel $panel"))
        push!(result[panel], summary)
    end
    for panel in panels
        sort!(result[panel]; by=summary -> Int(_render_field(summary, :contest_id,
            "summary $panel")))
    end
    return result
end

function _ce_gap_rows(artifact, methods::Vector{Symbol},
                      panels::Vector{Symbol}, horizon::Int)
    summaries = _summaries_by_panel(artifact, panels)
    rows = NamedTuple[]
    for panel in panels, (method_a, method_b) in _method_pairs(methods)
        contest_ids = Int[]
        samples = Float64[]
        lower_samples = Float64[]
        upper_samples = Float64[]
        confidence_levels = Set{Float64}()
        resolved_a = 0
        resolved_b = 0
        unresolved = 0
        for (index, summary) in enumerate(summaries[panel])
            location = "summary $panel[$index]"
            summary_methods = Symbol.(collect(
                _render_field(summary, :method_ids, location)))
            summary_methods == methods || throw(ArgumentError(
                "$location method order differs from processed metadata"))
            expected = Float64.(collect(
                _render_field(summary, :expected_utility, location)))
            method_a_index = findfirst(==(method_a), summary_methods)
            method_b_index = findfirst(==(method_b), summary_methods)
            beta = Float64(_render_field(summary, :beta, location))
            gamma = Float64(_render_field(summary, :gamma, location))
            push!(contest_ids, Int(_render_field(summary, :contest_id, location)))
            point_bps = consumption_equivalent_bps(
                expected[method_a_index], expected[method_b_index], beta,
                gamma, horizon)
            push!(samples, point_bps)

            comparison = _oriented_comparison(
                summary, method_a, method_b, location)
            push!(confidence_levels, comparison.confidence_level)
            interval = _consumption_equivalent_interval_bps(
                expected[method_b_index], comparison.ci_lower,
                comparison.ci_upper, beta, gamma, horizon)
            push!(lower_samples, interval.lower_bps)
            push!(upper_samples, interval.upper_bps)
            tolerance = 64eps(max(1.0, abs(point_bps),
                abs(interval.lower_bps), abs(interval.upper_bps)))
            interval.lower_bps - tolerance <= point_bps <=
                interval.upper_bps + tolerance || throw(ArgumentError(
                    "$location paired CEG interval does not contain its point gap"))
            if interval.lower_bps > 0.0
                resolved_a += 1
            elseif interval.upper_bps < 0.0
                resolved_b += 1
            else
                unresolved += 1
            end
        end
        n_contests = length(samples)
        n_contests > 0 || throw(ArgumentError(
            "panel $panel has no CEG observations"))
        length(unique(contest_ids)) == n_contests || throw(ArgumentError(
            "panel $panel has duplicate contest identifiers"))
        length(confidence_levels) == 1 || throw(ArgumentError(
            "panel $panel pair $method_a/$method_b mixes confidence levels"))
        confidence_level = only(confidence_levels)
        quantiles = quantile(samples, collect(CE_QUANTILE_PROBABILITIES))
        push!(rows, (;
            panel,
            method_a,
            method_b,
            orientation=:scale_method_b_to_match_method_a,
            ci_endpoint_reference=:method_b_expected_utility,
            interval_endpoint_kind=
                :plugin_ce_scaled_paired_utility_gap_interval,
            interval_has_nominal_ce_coverage=false,
            confidence_level,
            n_contests,
            sample=_completion_status(artifact),
            contest_ids,
            ce_gap_samples_bps=samples,
            ce_ci_lower_samples_bps=lower_samples,
            ce_ci_upper_samples_bps=upper_samples,
            q05_bps=quantiles[1],
            q25_bps=quantiles[2],
            median_bps=quantiles[3],
            q75_bps=quantiles[4],
            q95_bps=quantiles[5],
            median_ci_lower_bps=quantile(lower_samples, 0.5),
            median_ci_upper_bps=quantile(upper_samples, 0.5),
            method_a_resolved_count=resolved_a,
            method_a_resolved_share=resolved_a / n_contests,
            method_b_resolved_count=resolved_b,
            method_b_resolved_share=resolved_b / n_contests,
            pairwise_unresolved_count=unresolved,
            pairwise_unresolved_share=unresolved / n_contests,
        ))
    end
    return rows
end

function _timing_rows(artifact, methods::Vector{Symbol},
                      panels::Vector{Symbol}, timing_paths::Int)
    timing_paths > 0 || throw(ArgumentError(
        "timing path count must be positive"))
    timings = Dict{Tuple{Symbol,Symbol},Any}()
    for (index, timing) in enumerate(_render_field(artifact, :timings))
        location = "timings[$index]"
        panel = Symbol(_render_field(timing, :panel, location))
        method = Symbol(_render_field(timing, :method, location))
        key = (panel, method)
        haskey(timings, key) && throw(ArgumentError(
            "processed timings contain a duplicate row for $panel/$method"))
        timings[key] = timing
    end
    expected = Set((panel, method) for panel in panels for method in methods)
    Set(keys(timings)) == expected || throw(ArgumentError(
        "processed timings do not contain exactly one row per panel/method"))

    rows = NamedTuple[]
    for panel in panels, method in methods
        timing = timings[(panel, method)]
        total_seconds = Float64(_render_field(
            timing, :total_seconds, "timing $panel/$method"))
        isfinite(total_seconds) && total_seconds >= 0.0 || throw(ArgumentError(
            "timing $panel/$method has invalid total_seconds"))
        push!(rows, (;
            panel,
            method,
            total_seconds,
            sample=_completion_status(artifact),
            timing_paths,
            amortized_seconds_per_path=total_seconds / timing_paths,
            thread_count=Int(_render_field(
                timing, :thread_count, "timing $panel/$method")),
            status=Symbol(_render_field(timing, :status,
                "timing $panel/$method")),
            completed_utc=String(_render_field(timing, :completed_utc,
                "timing $panel/$method")),
            source_relative_path=String(_render_field(timing, :relative_path,
                "timing $panel/$method")),
            source_authentication_fingerprint=String(_render_field(
                timing, :authentication_fingerprint,
                "timing $panel/$method")),
        ))
    end
    return rows
end

const _TEX_CHARACTER_ESCAPES = Dict{Char,String}(
    '\\' => "\\textbackslash{}",
    '{' => "\\{",
    '}' => "\\}",
    Char(0x24) => string('\\', Char(0x24)),
    '&' => "\\&",
    '#' => "\\#",
    '%' => "\\%",
    '_' => "\\_",
    '^' => "\\textasciicircum{}",
    '~' => "\\textasciitilde{}",
)

function _tex_escape(value::AbstractString)
    io = IOBuffer()
    for character in String(value)
        print(io, get(_TEX_CHARACTER_ESCAPES, character, string(character)))
    end
    return String(take!(io))
end

_panel_label(panel::Symbol) = panel === :baseline ? "Baseline" :
    panel === :high_variance ? "Higher variance" : _tex_escape(String(panel))

_panel_label(panel::Symbol, sigma::Real) = string(
    _panel_label(panel), " (\$\\sigma=", @sprintf("%.6g", Float64(sigma)), "\$)")

function _method_label(method::Symbol)
    method === :ptd_sceq && return "PTD-SCEQ"
    method === :ptd_osa && return "PTD-OSA"
    method === :dcegm && return "DC-EGM"
    return _tex_escape(String(method))
end

_percent(value::Real) = @sprintf("%.2f", 100.0 * Float64(value))
_bps(value::Real) = @sprintf("%.3f", Float64(value))
_seconds(value::Real) = @sprintf("%.6f", Float64(value))

function _trace_header(io, title::AbstractString, source,
                       row_fingerprint::AbstractString)
    println(io, "% Generated by Ptd.IID; do not edit by hand.")
    println(io, "% Table: ", title)
    println(io, "% Processed run fingerprint: ", source.run_fingerprint)
    println(io, "% Processed result fingerprint: ", source.result_fingerprint)
    println(io, "% Processed file SHA-256: ", source.file_sha256)
    println(io, "% Row-data fingerprint: ", row_fingerprint)
end

function _render_outcome_table(path::AbstractString, rows, panel_sigmas, source,
                               row_fingerprint::String)
    open(path, "w") do io
        _trace_header(io, "IID contest outcome shares", source,
            row_fingerprint)
        println(io, "\\begin{tabular}{llrrrrrr}")
        println(io, "\\toprule")
        println(io, "Panel & Method & Point win (\\%) & Resolved win (\\%) & Resolved loss (\\%) & MC unresolved (\\%) & Exact tie (\\%) & All unresolved (\\%) \\\\")
        println(io, "\\midrule")
        for row in rows
            println(io, _panel_label(
                    row.panel, _panel_sigma(row.panel, panel_sigmas)), " & ",
                _method_label(row.method), " & ",
                _percent(row.point_winner_share), " & ",
                _percent(row.resolved_win_share), " & ",
                _percent(row.resolved_loss_share), " & ",
                _percent(row.mc_unresolved_share), " & ",
                _percent(row.point_tie_share), " & ",
                _percent(row.unresolved_share), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        _render_sample_note(io, first(rows).sample)
    end
    return path
end


function _render_ce_gap_table(path::AbstractString, rows, panel_sigmas, source,
                              row_fingerprint::String)
    confidence_levels = unique(Float64(row.confidence_level) for row in rows)
    length(confidence_levels) == 1 || throw(ArgumentError(
        "CEG table rows mix paired confidence levels"))
    confidence_percent = @sprintf("%.0f", 100.0 * only(confidence_levels))
    open(path, "w") do io
        _trace_header(io, "IID contest CRRA consumption-equivalent gaps",
            source, row_fingerprint)
        println(io, "% Positive CEGs (in bps) favor method A; the scale is applied to method B to match method A.")
        panels = unique(row.panel for row in rows)
        println(io, "\\begin{tabular}{llrrrrrr}")
        println(io, "\\toprule")
        println(io, " & & \\multicolumn{3}{c}{CEG (basis points)} & \\multicolumn{3}{c}{Pairwise resolution (\\%)} \\\\")
        println(io, "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}")
        println(io, "Method A & Method B & P5 & Median & P95 & A resolved & B resolved & Unresolved \\\\")
        for panel in panels
            println(io, "\\midrule")
            println(io, "\\multicolumn{8}{l}{", _panel_label(
                panel, _panel_sigma(panel, panel_sigmas)), "} \\\\")
            for row in rows
                row.panel === panel || continue
                println(io, _method_label(row.method_a), " & ",
                    _method_label(row.method_b), " & ",
                    _bps(row.q05_bps), " & ",
                    _bps(row.median_bps), " & ",
                    _bps(row.q95_bps), " & ",
                    _percent(row.method_a_resolved_share), " & ",
                    _percent(row.method_b_resolved_share), " & ",
                    _percent(row.pairwise_unresolved_share), " \\\\")
            end
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        println(io, "\\fnote{Notes: CEG is the consumption-equivalent gap; positive values favor method A. One basis point equals 0.01\\%. A or B is resolved when its paired $confidence_percent\\% utility-difference interval favors that method.}")
    end
    return path
end

function _render_timing_table(path::AbstractString, rows, panel_sigmas, source,
                              row_fingerprint::String)
    open(path, "w") do io
        _trace_header(io, "IID contest total solve times", source,
            row_fingerprint)
        println(io, "% Each panel/method appears exactly once; no timing components or dispersion are reported.")
        println(io, "\\begin{tabular}{llrr}")
        println(io, "\\toprule")
        println(io, "Panel & Method & Total seconds & Amortized seconds/path \\\\")
        println(io, "\\midrule")
        for row in rows
            println(io, _panel_label(
                row.panel, _panel_sigma(row.panel, panel_sigmas)), " & ",
                _method_label(row.method), " & ",
                _seconds(row.total_seconds), " & ",
                _seconds(row.amortized_seconds_per_path), " \\\\")
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        _render_sample_note(io, first(rows).sample)
    end
    return path
end

function _paper_timing_comments(io, summary)
    isnothing(summary) && return println(io, "% Single-thread timing: not reported (no timing artifact supplied).")
    println(io, "% Single-thread timing artifact SHA-256: ", summary.artifact_sha256)
    println(io, "% Single-thread timing artifact fingerprint: ", summary.artifact_fingerprint)
    println(io, "% Single-thread timing source fingerprint: ", summary.source_fingerprint)
    println(io, "% Single-thread timing provenance fingerprint: ", summary.provenance_fingerprint)
end

function _render_single_path_note(io, summary, panels)
    if isnothing(summary)
        println(io, "\\par\\smallskip{\\footnotesize \\emph{Timing:} Single-thread timing is not reported; no authenticated timing artifact was supplied.}")
        return
    end
    repeats = [summary.configurations[panel].repeats for panel in panels]
    count_note = length(unique(repeats)) == 1 ? string(first(repeats)) : join(repeats, "/")
    println(io, "\\par\\smallskip{\\footnotesize \\emph{Timing:} Mean and median seconds for fresh policy construction plus one path, using one Julia thread and one BLAS thread; ",
        count_note, " measured repetitions per method", length(unique(repeats)) == 1 ? ". " : " in panel order. ",
        raw"Timing fixes $\beta=0.98$, $\gamma=1$, initial cash 16, and 50 periods; accuracy varies preferences and initial cash. ",
        "The complete backward solve is included; stochastic PTD plans are constructed during path evaluation. Full warmups are excluded.}")
end

function _render_iid_random_timing_note(io, summary)
    if isnothing(summary)
        println(io, "\\par\\smallskip{\\footnotesize \\emph{Timing:} Single-thread timing is not reported; no authenticated heterogeneous-agent timing artifact was supplied.}")
        return
    end
    println(io, "\\par\\smallskip{\\footnotesize \\emph{Timing:} Mean and median seconds across the same ",
        summary.configuration.n_agents,
        " agents in each accuracy panel, for fresh policy construction plus one path, using one Julia thread and one BLAS thread. Methods share the first saved accuracy path for each agent; full warmups are excluded.}")
end

"""
Render IID accuracy with single-thread means and medians across the same agents.
Missing heterogeneous-agent timing evidence is displayed as not reported.
Set `include_sample_note=false` when the surrounding paper supplies the notes.
"""
function render_iid_paper_table(processed_path::AbstractString;
    output_path::AbstractString=tables_dir("iid_contest_summary.tex"),
    include_sample_note::Bool=true,
    timing_artifact_path::Union{Nothing,AbstractString}=nothing,
    require_production_timing::Bool=true)
    artifact = load_iid_contest(processed_path)
    _completion_status(artifact).completed_contests == 0 &&
        return _render_no_success_table(artifact, output_path)
    methods = Symbol.(collect(_render_field(artifact, :methods)))
    panels = Symbol.(collect(_render_field(artifact, :panels)))
    metadata = _render_field(artifact, :metadata)
    scientific = _render_field(artifact, :scientific_configuration)
    model = _render_field(scientific, :model)
    panel_sigmas = _panel_specifications(
        _render_field(model, :panels), panels,
        "scientific_configuration.model.panels")
    outcome_rows = _outcome_rows(artifact, methods, panels)
    outcomes = Dict((row.panel, row.method) => row for row in outcome_rows)
    timing = isnothing(timing_artifact_path) ? nothing :
        load_iid_random_timing_summary(timing_artifact_path, processed_path;
            require_production=require_production_timing)

    destination = _validate_data_output(output_path)
    mkpath(dirname(destination))
    open(destination, "w") do io
        println(io, "% Generated by scripts/3_iid.jl; do not edit.")
        println(io, "% Source authentication fingerprint: ",
            _render_field(metadata, :authentication_fingerprint))
        _paper_timing_comments(io, timing)
        println(io, "\\begin{tabular}{lrrrr}")
        println(io, "\\toprule")
        println(io, "Method & \\makecell{Point win\\\\(\\%)} & \\makecell{Resolved win\\\\(\\%)} & \\makecell{Mean time\\\\(s)} & \\makecell{Median time\\\\(s)} \\\\")
        println(io, "\\midrule")
        for (panel_index, panel) in enumerate(panels)
            panel_index > 1 && println(io, "\\midrule")
            println(io, "\\multicolumn{5}{l}{",
                _panel_label(panel, _panel_sigma(panel, panel_sigmas)), "} \\\\")
            for method in methods
                outcome = outcomes[(panel, method)]
                row = isnothing(timing) ? nothing : timing.rows[(panel, method)]
                average = isnothing(row) ? "not reported" : @sprintf("%.3f", row.mean_seconds)
                middle = isnothing(row) ? "not reported" : @sprintf("%.3f", row.median_seconds)
                println(io, _method_label(method), " & ",
                    round(Int, 100 * outcome.point_winner_share), " & ",
                    round(Int, 100 * outcome.resolved_win_share), " & ",
                    average, " & ", middle, " \\\\")
            end
        end
        println(io, "\\bottomrule")
        println(io, "\\end{tabular}")
        if include_sample_note
            _render_sample_note(io, first(outcome_rows).sample; timing=false)
            _render_iid_random_timing_note(io, timing)
        end
    end
    return destination
end

"""Render the IID CE quantiles from the same authenticated contest as the summary."""
function render_iid_ce_gap_table(processed_path::AbstractString;
    output_path::AbstractString=tables_dir(CE_GAP_TABLE_FILENAME))
    artifact = load_iid_contest(processed_path)
    _completion_status(artifact).completed_contests == 0 &&
        return _render_no_success_table(artifact, output_path)
    methods = Symbol.(collect(_render_field(artifact, :methods)))
    panels = Symbol.(collect(_render_field(artifact, :panels)))
    metadata = _render_field(artifact, :metadata)
    model = _render_field(_render_field(artifact, :scientific_configuration), :model)
    panel_sigmas = _panel_specifications(_render_field(model, :panels), panels,
        "scientific_configuration.model.panels")
    rows = _ce_gap_rows(artifact, methods, panels, Int(_render_field(model, :horizon)))
    destination = _validate_data_output(output_path)
    mkpath(dirname(destination))
    return _render_ce_gap_table(destination, rows, panel_sigmas,
        _paper_table_source(processed_path, artifact), stable_fingerprint(rows))
end

# Compact tables need only the authenticated run/result identity. This also
# supports archived artifacts whose full render-manifest audit predates v3.
function _paper_table_source(path::AbstractString, artifact)
    metadata = _render_field(artifact, :metadata)
    return (run_fingerprint=String(_render_field(metadata, :run_fingerprint)),
        result_fingerprint=String(_render_field(metadata, :result_fingerprint)),
        file_sha256=_sha256_file(path))
end

function _source_provenance(processed_path::AbstractString, artifact)
    path = abspath(processed_path)
    metadata = _render_field(artifact, :metadata, "processed artifact")
    return (;
        status=:authenticated,
        artifact_path=path,
        file_bytes=filesize(path),
        file_mtime_utc=_render_file_mtime_utc(path),
        file_sha256=_sha256_file(path),
        artifact_schema=String(_render_field(metadata, :artifact_schema,
            "processed metadata")),
        completed_utc=String(_render_field(metadata, :completed_utc,
            "processed metadata")),
        scientific_fingerprint=String(_render_field(metadata,
            :scientific_fingerprint, "processed metadata")),
        run_fingerprint=String(_render_field(metadata, :run_fingerprint,
            "processed metadata")),
        source_commit=String(_render_field(metadata, :source_commit,
            "processed metadata")),
        input_tree_fingerprint=String(_render_field(metadata,
            :input_tree_fingerprint, "processed metadata")),
        environment_fingerprint=String(_render_field(metadata,
            :environment_fingerprint, "processed metadata")),
        simulation_metadata_fingerprint=String(_render_field(metadata,
            :simulation_metadata_fingerprint, "processed metadata")),
        run_audit_fingerprint=String(_render_field(metadata,
            :run_audit_fingerprint, "processed metadata")),
        utilities_fingerprint=String(_render_field(metadata,
            :utilities_fingerprint, "processed metadata")),
        summaries_fingerprint=String(_render_field(metadata,
            :summaries_fingerprint, "processed metadata")),
        aggregates_fingerprint=String(_render_field(metadata,
            :aggregates_fingerprint, "processed metadata")),
        result_fingerprint=String(_render_field(metadata, :result_fingerprint,
            "processed metadata")),
        authentication_fingerprint=String(_render_field(metadata,
            :authentication_fingerprint, "processed metadata")),
    )
end

function _source_identity(source)
    return stable_fingerprint((;
        artifact_schema=source.artifact_schema,
        file_sha256=source.file_sha256,
        scientific_fingerprint=source.scientific_fingerprint,
        run_fingerprint=source.run_fingerprint,
        result_fingerprint=source.result_fingerprint,
        authentication_fingerprint=source.authentication_fingerprint,
    ))
end

function _table_descriptor(id::Symbol, filename::String,
                           completed_utc::String, row_fingerprint::String,
                           directory::AbstractString)
    path = joinpath(directory, filename)
    return (;
        id,
        filename,
        status=:complete,
        completed_utc,
        row_fingerprint,
        bytes=filesize(path),
        sha256=_sha256_file(path),
    )
end

function _table_content_fingerprint(tables)
    identities = [(;
        id=Symbol(_render_field(table, :id, "table")),
        filename=String(_render_field(table, :filename, "table")),
        row_fingerprint=String(_render_field(table, :row_fingerprint, "table")),
        bytes=Int(_render_field(table, :bytes, "table")),
        sha256=String(_render_field(table, :sha256, "table")),
    ) for table in tables]
    return stable_fingerprint(identities)
end

function _render_authentication(metadata)
    return stable_fingerprint((;
        schema=RENDER_AUTHENTICATION_SCHEMA_VERSION,
        artifact_schema=RENDER_CONTEST_SCHEMA_VERSION,
        status=_render_field(metadata, :status, "metadata"),
        completed_utc=String(_render_field(metadata, :completed_utc, "metadata")),
        source_status=_render_field(metadata, :source_status, "metadata"),
        source_artifact_schema=String(_render_field(
            metadata, :source_artifact_schema, "metadata")),
        source_identity_fingerprint=String(_render_field(
            metadata, :source_identity_fingerprint, "metadata")),
        source_provenance_fingerprint=String(_render_field(
            metadata, :source_provenance_fingerprint, "metadata")),
        source_run_fingerprint=String(_render_field(
            metadata, :source_run_fingerprint, "metadata")),
        source_result_fingerprint=String(_render_field(
            metadata, :source_result_fingerprint, "metadata")),
        source_authentication_fingerprint=String(_render_field(
            metadata, :source_authentication_fingerprint, "metadata")),
        simulation_metadata_fingerprint=String(_render_field(
            metadata, :simulation_metadata_fingerprint, "metadata")),
        render_configuration_fingerprint=String(_render_field(
            metadata, :render_configuration_fingerprint, "metadata")),
        outcome_rows_fingerprint=String(_render_field(
            metadata, :outcome_rows_fingerprint, "metadata")),
        ce_gap_rows_fingerprint=String(_render_field(
            metadata, :ce_gap_rows_fingerprint, "metadata")),
        timing_rows_fingerprint=String(_render_field(
            metadata, :timing_rows_fingerprint, "metadata")),
        table_catalog_fingerprint=String(_render_field(
            metadata, :table_catalog_fingerprint, "metadata")),
        table_content_fingerprint=String(_render_field(
            metadata, :table_content_fingerprint, "metadata")),
        result_fingerprint=String(_render_field(
            metadata, :result_fingerprint, "metadata")),
    ))
end

function _write_manifest(path::AbstractString, artifact::AbstractDict;
                         compress::Bool)
    JLD2.jldopen(path, "w"; compress) do file
        for (name, value) in pairs(artifact)
            file[String(name)] = value
        end
    end
    return path
end

function _render_load_jld2(path::AbstractString)
    isfile(path) || throw(ArgumentError(
        "render manifest does not exist: $path"))
    return JLD2.jldopen(path, "r") do file
        Dict{String,Any}(String(name) => file[String(name)] for name in keys(file))
    end
end

function _validate_rows(outcome_rows, ce_rows, timing_rows,
                        methods::Vector{Symbol}, panels::Vector{Symbol},
                        confidence_level::Float64)
    !isempty(outcome_rows) && !isempty(ce_rows) && !isempty(timing_rows) ||
        throw(ArgumentError("render manifest contains an empty row collection"))
    length(unique(methods)) == length(methods) && length(methods) >= 2 ||
        throw(ArgumentError("render method identifiers are invalid"))
    length(unique(panels)) == length(panels) && !isempty(panels) ||
        throw(ArgumentError("render panel identifiers are invalid"))
    isfinite(confidence_level) && 0.0 < confidence_level < 1.0 ||
        throw(ArgumentError("render paired confidence level is invalid"))

    expected_method_keys = Set((panel, method)
        for panel in panels for method in methods)
    outcome_keys = [(row.panel, row.method) for row in outcome_rows]
    length(unique(outcome_keys)) == length(outcome_keys) &&
        Set(outcome_keys) == expected_method_keys || throw(ArgumentError(
            "outcome rows do not contain exactly one row per panel/method"))
    contests_by_panel = Dict{Symbol,Int}()
    panel_counts = Dict{Symbol,Tuple{Int,Int,Int}}()
    for row in outcome_rows
        row.n_contests > 0 || throw(ArgumentError(
            "outcome row has no contests"))
        get!(contests_by_panel, row.panel, row.n_contests) == row.n_contests ||
            throw(ArgumentError("outcome contest count changes within a panel"))
        counts = (row.mc_unresolved_count, row.point_tie_count,
            row.unresolved_count)
        get!(panel_counts, row.panel, counts) == counts ||
            throw(ArgumentError("panel-wide outcome counts change by method"))
        for (count, share) in (
            (row.point_winner_count, row.point_winner_share),
            (row.resolved_win_count, row.resolved_win_share),
            (row.resolved_loss_count, row.resolved_loss_share),
            (row.mc_unresolved_count, row.mc_unresolved_share),
            (row.point_tie_count, row.point_tie_share),
            (row.unresolved_count, row.unresolved_share),
        )
            0 <= count <= row.n_contests &&
                share == count / row.n_contests || throw(ArgumentError(
                    "outcome count/share pair is inconsistent"))
        end
    end

    expected_ce_keys = Set((panel, method_a, method_b)
        for panel in panels for (method_a, method_b) in _method_pairs(methods))
    ce_keys = [(row.panel, row.method_a, row.method_b) for row in ce_rows]
    length(unique(ce_keys)) == length(ce_keys) &&
        Set(ce_keys) == expected_ce_keys || throw(ArgumentError(
            "CEG rows do not contain exactly one row per panel/method pair"))
    for row in ce_rows
        row.orientation === :scale_method_b_to_match_method_a ||
            throw(ArgumentError("CEG row has an invalid orientation"))
        row.ci_endpoint_reference === :method_b_expected_utility ||
            throw(ArgumentError(
                "CEG row has an invalid paired-endpoint reference"))
        row.interval_endpoint_kind ===
            :plugin_ce_scaled_paired_utility_gap_interval &&
            row.interval_has_nominal_ce_coverage === false ||
            throw(ArgumentError(
                "CEG row overstates its plug-in endpoint interpretation"))
        row.confidence_level == confidence_level || throw(ArgumentError(
            "CEG row confidence level differs from render configuration"))
        row.n_contests == contests_by_panel[row.panel] ==
            length(row.contest_ids) == length(row.ce_gap_samples_bps) ==
            length(row.ce_ci_lower_samples_bps) ==
            length(row.ce_ci_upper_samples_bps) ||
            throw(ArgumentError("CEG row has an invalid contest count"))
        length(unique(row.contest_ids)) == row.n_contests &&
            all(isfinite, row.ce_gap_samples_bps) &&
            all(isfinite, row.ce_ci_lower_samples_bps) &&
            all(isfinite, row.ce_ci_upper_samples_bps) || throw(ArgumentError(
                "CEG samples are invalid"))
        for index in 1:row.n_contests
            lower = row.ce_ci_lower_samples_bps[index]
            point = row.ce_gap_samples_bps[index]
            upper = row.ce_ci_upper_samples_bps[index]
            lower <= upper || throw(ArgumentError(
                "paired CEG interval endpoints are reversed"))
            tolerance = 64eps(max(1.0, abs(lower), abs(point), abs(upper)))
            lower - tolerance <= point <= upper + tolerance ||
                throw(ArgumentError(
                    "paired CEG interval does not contain its point gap"))
        end
        quantiles = quantile(row.ce_gap_samples_bps,
            collect(CE_QUANTILE_PROBABILITIES))
        (row.q05_bps, row.q25_bps, row.median_bps, row.q75_bps,
            row.q95_bps) == Tuple(quantiles) || throw(ArgumentError(
                "stored CEG quantiles do not match their samples"))
        row.median_ci_lower_bps ==
            quantile(row.ce_ci_lower_samples_bps, 0.5) &&
            row.median_ci_upper_bps ==
                quantile(row.ce_ci_upper_samples_bps, 0.5) ||
            throw(ArgumentError(
                "stored median paired CEG endpoints do not match their samples"))
        resolved_a = count(>(0.0), row.ce_ci_lower_samples_bps)
        resolved_b = count(<(0.0), row.ce_ci_upper_samples_bps)
        unresolved = row.n_contests - resolved_a - resolved_b
        (row.method_a_resolved_count, row.method_b_resolved_count,
            row.pairwise_unresolved_count) ==
            (resolved_a, resolved_b, unresolved) || throw(ArgumentError(
                "pairwise resolution counts disagree with CEG endpoints"))
        row.method_a_resolved_count + row.method_b_resolved_count +
            row.pairwise_unresolved_count == row.n_contests ||
            throw(ArgumentError("pairwise resolution counts do not sum to contests"))
        for (count, share) in (
            (row.method_a_resolved_count, row.method_a_resolved_share),
            (row.method_b_resolved_count, row.method_b_resolved_share),
            (row.pairwise_unresolved_count, row.pairwise_unresolved_share),
        )
            0 <= count <= row.n_contests &&
                share == count / row.n_contests || throw(ArgumentError(
                    "pairwise resolution count/share is inconsistent"))
        end
    end

    timing_keys = [(row.panel, row.method) for row in timing_rows]
    length(unique(timing_keys)) == length(timing_keys) &&
        Set(timing_keys) == expected_method_keys || throw(ArgumentError(
            "timing rows do not contain exactly one row per panel/method"))
    all(row -> row.status === :authenticated &&
        isfinite(row.total_seconds) && row.total_seconds >= 0.0 &&
        row.timing_paths > 0 &&
        row.amortized_seconds_per_path ==
            row.total_seconds / row.timing_paths &&
        row.thread_count > 0 && endswith(row.completed_utc, 'Z'),
        timing_rows) || throw(ArgumentError(
            "render timing row has invalid status or total_seconds"))
    return nothing
end

function _validate_render_manifest(artifact, directory::AbstractString)
    artifact isa AbstractDict || throw(ArgumentError(
        "render manifest must be a dictionary"))
    metadata = _render_field(artifact, :metadata)
    String(_render_field(metadata, :artifact_schema, "metadata")) ==
        RENDER_CONTEST_SCHEMA_VERSION || throw(ArgumentError(
            "render manifest schema mismatch"))
    _render_field(metadata, :status, "metadata") === :complete || throw(ArgumentError(
        "render manifest is not complete"))
    completed = String(_render_field(metadata, :completed_utc, "metadata"))
    !isempty(completed) && endswith(completed, 'Z') || throw(ArgumentError(
        "render completion timestamp is invalid"))

    source = _render_field(artifact, :source_provenance)
    _render_field(source, :status, "source_provenance") === :authenticated ||
        throw(ArgumentError("render source is not authenticated"))
    String(_render_field(source, :artifact_schema, "source_provenance")) ==
        IID_CONTEST_ARTIFACT_SCHEMA_VERSION || throw(ArgumentError(
            "render source is not a supported processed contest artifact"))
    stable_fingerprint(source) == String(_render_field(metadata,
        :source_provenance_fingerprint, "metadata")) || throw(ArgumentError(
            "render source-provenance fingerprint mismatch"))
    _source_identity(source) == String(_render_field(metadata,
        :source_identity_fingerprint, "metadata")) || throw(ArgumentError(
            "render source-identity fingerprint mismatch"))
    _render_field(metadata, :source_status, "metadata") ===
        _render_field(source, :status, "source_provenance") &&
        String(_render_field(metadata, :source_artifact_schema, "metadata")) ==
            String(_render_field(source, :artifact_schema, "source_provenance")) &&
        String(_render_field(metadata, :source_run_fingerprint, "metadata")) ==
            String(_render_field(source, :run_fingerprint, "source_provenance")) &&
        String(_render_field(metadata, :source_result_fingerprint, "metadata")) ==
            String(_render_field(source, :result_fingerprint, "source_provenance")) &&
        String(_render_field(metadata, :source_authentication_fingerprint,
            "metadata")) == String(_render_field(source,
            :authentication_fingerprint, "source_provenance")) ||
        throw(ArgumentError(
            "render metadata disagrees with source provenance"))

    simulation = _render_field(artifact, :simulation_metadata)
    stable_fingerprint(simulation) == String(_render_field(metadata,
        :simulation_metadata_fingerprint, "metadata")) || throw(ArgumentError(
            "render simulation-metadata fingerprint mismatch"))
    stable_fingerprint(simulation) == String(_render_field(source,
        :simulation_metadata_fingerprint, "source_provenance")) ||
        throw(ArgumentError(
            "render simulation metadata differs from processed provenance"))
    configuration = _render_field(artifact, :render_configuration)
    String(_render_field(configuration, :schema, "render_configuration")) ==
        RENDER_CONFIGURATION_SCHEMA_VERSION || throw(ArgumentError(
            "render configuration schema mismatch"))
    stable_fingerprint(configuration) == String(_render_field(metadata,
        :render_configuration_fingerprint, "metadata")) || throw(ArgumentError(
            "render configuration fingerprint mismatch"))
    methods = Symbol.(collect(_render_field(configuration, :methods,
        "render_configuration")))
    panels = Symbol.(collect(_render_field(configuration, :panels,
        "render_configuration")))
    _panel_specifications(_render_field(configuration, :panel_sigmas,
        "render_configuration"), panels, "render_configuration.panel_sigmas")
    confidence_level = Float64(_render_field(configuration,
        :paired_confidence_level, "render_configuration"))

    outcome_rows = _render_field(artifact, :outcome_rows)
    ce_rows = _render_field(artifact, :ce_gap_rows)
    timing_rows = _render_field(artifact, :timing_rows)
    _validate_rows(outcome_rows, ce_rows, timing_rows, methods, panels,
        confidence_level)
    stable_fingerprint(outcome_rows) == String(_render_field(metadata,
        :outcome_rows_fingerprint, "metadata")) || throw(ArgumentError(
            "outcome-row fingerprint mismatch"))
    stable_fingerprint(ce_rows) == String(_render_field(metadata,
        :ce_gap_rows_fingerprint, "metadata")) || throw(ArgumentError(
            "CEG-row fingerprint mismatch"))
    stable_fingerprint(timing_rows) == String(_render_field(metadata,
        :timing_rows_fingerprint, "metadata")) || throw(ArgumentError(
            "timing-row fingerprint mismatch"))

    tables = _render_field(artifact, :tables)
    expected_ids = Set((:outcome_shares, :ce_gaps, :timings))
    Set(Symbol(_render_field(table, :id, "table")) for table in tables) ==
        expected_ids || throw(ArgumentError(
            "render manifest does not contain exactly the three expected tables"))
    length(tables) == 3 || throw(ArgumentError(
        "render manifest contains duplicate table descriptors"))
    filenames = String[_render_field(table, :filename, "table") for table in tables]
    length(unique(filenames)) == 3 || throw(ArgumentError(
        "render manifest table filenames are not unique"))
    expected_filenames = Dict(
        :outcome_shares => String(_render_field(_render_field(configuration, :table_files,
            "render_configuration"), :outcome_shares,
            "render_configuration.table_files")),
        :ce_gaps => String(_render_field(_render_field(configuration, :table_files,
            "render_configuration"), :ce_gaps,
            "render_configuration.table_files")),
        :timings => String(_render_field(_render_field(configuration, :table_files,
            "render_configuration"), :timings,
            "render_configuration.table_files")),
    )
    expected_row_fingerprints = Dict(
        :outcome_shares => stable_fingerprint(outcome_rows),
        :ce_gaps => stable_fingerprint(ce_rows),
        :timings => stable_fingerprint(timing_rows),
    )
    for (index, table) in enumerate(tables)
        location = "tables[$index]"
        _render_field(table, :status, location) === :complete || throw(ArgumentError(
            "$location is not complete"))
        filename = String(_render_field(table, :filename, location))
        table_id = Symbol(_render_field(table, :id, location))
        filename == expected_filenames[table_id] || throw(ArgumentError(
            "$location filename disagrees with the render configuration"))
        String(_render_field(table, :row_fingerprint, location)) ==
            expected_row_fingerprints[table_id] || throw(ArgumentError(
                "$location row fingerprint is inconsistent"))
        String(_render_field(table, :completed_utc, location)) == completed ||
            throw(ArgumentError("$location completion time is inconsistent"))
        basename(filename) == filename || throw(ArgumentError(
            "$location filename is not local to the render directory"))
        table_path = joinpath(directory, filename)
        isfile(table_path) || throw(ArgumentError(
            "rendered table is missing: $table_path"))
        filesize(table_path) == Int(_render_field(table, :bytes, location)) ||
            throw(ArgumentError("$location byte count mismatch"))
        _sha256_file(table_path) == String(_render_field(table, :sha256, location)) ||
            throw(ArgumentError("$location SHA-256 mismatch"))
    end
    stable_fingerprint(tables) == String(_render_field(metadata,
        :table_catalog_fingerprint, "metadata")) || throw(ArgumentError(
            "render table-catalog fingerprint mismatch"))
    table_content_fingerprint = _table_content_fingerprint(tables)
    table_content_fingerprint == String(_render_field(metadata,
        :table_content_fingerprint, "metadata")) || throw(ArgumentError(
            "render table-content fingerprint mismatch"))

    result_fingerprint = stable_fingerprint((;
        source_identity_fingerprint=String(_render_field(metadata,
            :source_identity_fingerprint, "metadata")),
        render_configuration_fingerprint=stable_fingerprint(configuration),
        outcome_rows_fingerprint=stable_fingerprint(outcome_rows),
        ce_gap_rows_fingerprint=stable_fingerprint(ce_rows),
        timing_rows_fingerprint=stable_fingerprint(timing_rows),
        table_content_fingerprint,
    ))
    result_fingerprint == String(_render_field(metadata,
        :result_fingerprint, "metadata")) || throw(ArgumentError(
            "render result fingerprint mismatch"))
    _render_authentication(metadata) == String(_render_field(metadata,
        :authentication_fingerprint, "metadata")) || throw(ArgumentError(
            "render manifest authentication mismatch"))
    return artifact
end

"""Load and authenticate a render manifest and all three adjacent TeX files."""
function load_render_manifest(path::AbstractString)
    manifest_path = abspath(path)
    return _validate_render_manifest(
        _render_load_jld2(manifest_path), dirname(manifest_path))
end

"""Deterministic output directory for one processed run fingerprint."""
render_output_directory(output_root::AbstractString,
                        run_fingerprint::AbstractString) =
    joinpath(abspath(output_root),
        "configuration-$(String(run_fingerprint))")

function _render_publication_identity(metadata)
    return stable_fingerprint((;
        source_identity_fingerprint=String(_render_field(metadata,
            :source_identity_fingerprint, "metadata")),
        render_configuration_fingerprint=String(_render_field(metadata,
            :render_configuration_fingerprint, "metadata")),
        result_fingerprint=String(_render_field(metadata,
            :result_fingerprint, "metadata")),
    ))
end

function _reuse_or_refuse(destination::AbstractString, proposed)
    manifest_path = joinpath(destination, RENDER_MANIFEST_FILENAME)
    existing = try
        load_render_manifest(manifest_path)
    catch error
        throw(ArgumentError(
            "refusing to replace invalid render output $destination: " *
            sprint(showerror, error)))
    end
    existing_identity = _render_publication_identity(
        _render_field(existing, :metadata, "existing render manifest"))
    proposed_identity = _render_publication_identity(
        _render_field(proposed, :metadata, "proposed render manifest"))
    existing_identity == proposed_identity || throw(ArgumentError(
        "refusing to replace a different authenticated render result at $destination"))
    return existing
end

function _publish_directory(destination::AbstractString,
                            temporary::AbstractString, proposed)
    if isdir(destination)
        return (:reused, _reuse_or_refuse(destination, proposed))
    elseif ispath(destination)
        throw(ArgumentError(
            "render destination exists and is not a directory: $destination"))
    end
    try
        Base.Filesystem.rename(temporary, destination)
    catch error
        if isdir(destination)
            return (:reused, _reuse_or_refuse(destination, proposed))
        end
        rethrow(error)
    end
    manifest = load_render_manifest(
        joinpath(destination, RENDER_MANIFEST_FILENAME))
    return (:computed, manifest)
end

"""
    render_iid_contest(processed_path; output_root, completed_at, compress)

Authenticate one compact processed IID contest artifact, derive CRRA
consumption-equivalent cross-contest summaries, and atomically publish exactly
three traceable TeX tables plus their authenticated render manifest. This stage
reads no timing source file, policy object, or solver output other
than the processed artifact supplied by `processed_path`.
"""
function render_iid_contest(
    processed_path::AbstractString;
    output_root::AbstractString=tables_dir("iid_contest"),
    completed_at::DateTime=now(UTC),
    compress::Bool=true)

    # Authentication happens before any output directory is created.
    processed = load_iid_contest(processed_path)
    processed_metadata = _render_field(processed, :metadata, "processed artifact")
    methods = Symbol.(collect(_render_field(processed, :methods,
        "processed artifact")))
    panels = Symbol.(collect(_render_field(processed, :panels,
        "processed artifact")))
    scientific = _render_field(processed, :scientific_configuration,
        "processed artifact")
    model = _render_field(scientific, :model, "scientific_configuration")
    horizon = Int(_render_field(model, :horizon,
        "scientific_configuration.model"))
    panel_sigmas = _panel_specifications(_render_field(model, :panels,
        "scientific_configuration.model"), panels,
        "scientific_configuration.model.panels")
    confidence_level = Float64(_render_field(processed_metadata,
        :confidence_level, "processed metadata"))
    n_contests = _completion_status(processed).completed_contests
    n_paths = Int(_render_field(processed_metadata,
        :n_paths, "processed metadata"))
    n_contests > 0 && n_paths > 0 || throw(ArgumentError(
        "no successful configurations: consult run_audit.failures; welfare inference and per-path timing are unavailable"))
    timing_paths = Base.checked_mul(n_contests, n_paths)
    configuration = _render_configuration(
        horizon, methods, panels, panel_sigmas, confidence_level,
        timing_paths)
    source = _source_provenance(processed_path, processed)
    simulation = _render_field(processed, :simulation_metadata,
        "processed artifact")

    outcome_rows = _outcome_rows(processed, methods, panels)
    ce_rows = _ce_gap_rows(processed, methods, panels, horizon)
    timing_rows = _timing_rows(processed, methods, panels, timing_paths)
    _validate_rows(outcome_rows, ce_rows, timing_rows, methods, panels,
        confidence_level)
    completed_utc = _render_utc_string(completed_at)
    outcome_fingerprint = stable_fingerprint(outcome_rows)
    ce_fingerprint = stable_fingerprint(ce_rows)
    timing_fingerprint = stable_fingerprint(timing_rows)

    root = abspath(output_root)
    _validate_data_output(joinpath(root, "render-output.tex"))
    ispath(root) && !isdir(root) && throw(ArgumentError(
        "render output root is not a directory: $root"))
    mkpath(root)
    destination = render_output_directory(root, String(_render_field(
        processed_metadata, :run_fingerprint, "processed metadata")))
    temporary = mktempdir(root;
        prefix=".iid-contest-render-write-", cleanup=false)
    published = false
    try
        _render_outcome_table(joinpath(temporary, OUTCOME_TABLE_FILENAME),
            outcome_rows, panel_sigmas, source, outcome_fingerprint)
        _render_ce_gap_table(joinpath(temporary, CE_GAP_TABLE_FILENAME),
            ce_rows, panel_sigmas, source, ce_fingerprint)
        _render_timing_table(joinpath(temporary, TIMING_TABLE_FILENAME),
            timing_rows, panel_sigmas, source, timing_fingerprint)
        tables = [
            _table_descriptor(:outcome_shares, OUTCOME_TABLE_FILENAME,
                completed_utc, outcome_fingerprint, temporary),
            _table_descriptor(:ce_gaps, CE_GAP_TABLE_FILENAME,
                completed_utc, ce_fingerprint, temporary),
            _table_descriptor(:timings, TIMING_TABLE_FILENAME,
                completed_utc, timing_fingerprint, temporary),
        ]
        source_identity_fingerprint = _source_identity(source)
        source_provenance_fingerprint = stable_fingerprint(source)
        render_configuration_fingerprint = stable_fingerprint(configuration)
        table_catalog_fingerprint = stable_fingerprint(tables)
        table_content_fingerprint = _table_content_fingerprint(tables)
        result_fingerprint = stable_fingerprint((;
            source_identity_fingerprint,
            render_configuration_fingerprint,
            outcome_rows_fingerprint=outcome_fingerprint,
            ce_gap_rows_fingerprint=ce_fingerprint,
            timing_rows_fingerprint=timing_fingerprint,
            table_content_fingerprint,
        ))
        metadata_without_authentication = (;
            artifact_schema=RENDER_CONTEST_SCHEMA_VERSION,
            status=:complete,
            completed_utc,
            source_status=:authenticated,
            source_artifact_schema=source.artifact_schema,
            source_identity_fingerprint,
            source_provenance_fingerprint,
            source_run_fingerprint=source.run_fingerprint,
            source_result_fingerprint=source.result_fingerprint,
            source_authentication_fingerprint=
                source.authentication_fingerprint,
            simulation_metadata_fingerprint=stable_fingerprint(simulation),
            render_configuration_fingerprint,
            outcome_rows_fingerprint=outcome_fingerprint,
            ce_gap_rows_fingerprint=ce_fingerprint,
            timing_rows_fingerprint=timing_fingerprint,
            table_catalog_fingerprint,
            table_content_fingerprint,
            result_fingerprint,
        )
        metadata = merge(metadata_without_authentication, (;
            authentication_fingerprint=
                _render_authentication(metadata_without_authentication),
        ))
        manifest_artifact = Dict{String,Any}(
            "metadata" => metadata,
            "source_provenance" => source,
            "simulation_metadata" => simulation,
            "render_configuration" => configuration,
            "outcome_rows" => outcome_rows,
            "ce_gap_rows" => ce_rows,
            "timing_rows" => timing_rows,
            "tables" => tables,
        )
        manifest_path = joinpath(temporary, RENDER_MANIFEST_FILENAME)
        _write_manifest(manifest_path, manifest_artifact; compress)
        load_render_manifest(manifest_path)
        status, restored = _publish_directory(
            destination, temporary, manifest_artifact)
        published = status === :computed
        table_paths = Dict(Symbol(table.id) =>
            joinpath(destination, String(table.filename))
            for table in restored["tables"])
        return (;
            status,
            output_directory=destination,
            manifest_path=joinpath(destination, RENDER_MANIFEST_FILENAME),
            table_paths,
            metadata=restored["metadata"],
            artifact=restored,
        )
    finally
        !published && ispath(temporary) && rm(temporary; recursive=true)
    end
end
