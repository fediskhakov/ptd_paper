import ..Ptd: PairedComparison, paired_comparison, stable_fingerprint

export HSZUtilityTrace, HSZUtilityArtifact, HSZUtilityRow,
       run_hsz_utility_comparison, load_hsz_utility_comparison,
       hsz_utility_summary, hsz_utility_quantiles, render_hsz_utility_comparison,
       default_hsz_utility_artifact_path, hsz_ce_percent

const HSZ_UTILITY_SCHEMA = "ptd-hsz-common-shock-utility-v1"

"""Realized actions, discounted log utilities, and floor transfers for one policy."""
struct HSZUtilityTrace
    cash::Matrix{Float64}
    consumption::Matrix{Float64}
    utility::Vector{Float64}
    floor_hits::Vector{Int}
    total_transfers::Vector{Float64}
end

"""Source identities and runtime information for a saved HSZ comparison."""
struct HSZUtilityProvenance
    source_artifact::String
    source_sha256::String
    source_scientific_fingerprint::String
    source_files::Dict{String,String}
    source_fingerprint::String
    created_utc::String
    julia_version::String
    default_threads::Int
    blas_threads::Int
end

"""
Reusable HSZ evidence with the same innovation rows at every initial cash.

`innovations[k,t,d]` stores each distinct shock path once. Within each cash
group, column `k` of every method uses that same row; `shock_row` records the
mapping for the complete cash-by-shock experiment. The example configuration
retains the original source figure's path allocation; the comparison's actual
allocation is specified by `initial_cash` and `shock_row`.
"""
struct HSZUtilityArtifact
    schema::String
    config::HSZExampleConfig
    scientific_fingerprint::String
    provenance::HSZUtilityProvenance
    innovations::Array{Float64,3}
    initial_cash::Vector{Float64}
    shock_row::Vector{Int}
    paths::Dict{Symbol,HSZUtilityTrace}
end

"""One initial-cash row: method means/counts and two paired G2EGM-minus-PTD intervals."""
struct HSZUtilityRow
    initial_cash::Float64
    n_paths::Int
    mean_utility::NTuple{3,Float64}
    floor_paths::NTuple{3,Int}
    floor_visits::NTuple{3,Int}
    comparisons::NTuple{2,PairedComparison}
end

"""Artifact and generated LaTeX table paths."""
struct HSZUtilityOutput
    artifact_path::String
    table_path::String
end

"""Location of the five-by-100 HSZ utility artifact, separate from the path figure."""
default_hsz_utility_artifact_path() =
    joinpath(PROJECT_ROOT, "data", "hsz-utility-medical-1p5.jld2")

# Explicitly convert our records to portable scalar/array dictionaries before
# canonical hashing. JLD2 stores the documented structs themselves.
_utility_record(value) = value
function _utility_record(value::Union{HSZExampleConfig,HSZUtilityTrace,
    HSZUtilityProvenance,HSZUtilityArtifact})
    return Dict(String(field) => _utility_record(getfield(value, field))
        for field in fieldnames(typeof(value)))
end
_utility_record(value::AbstractDict) =
    Dict(key => _utility_record(item) for (key, item) in value)

function _utility_scientific(config, n_shocks)
    return Dict("schema" => HSZ_UTILITY_SCHEMA,
        "model_and_original_figure" => _utility_record(config),
        "shock_paths_per_cash" => n_shocks,
        "total_paths_per_method" => n_shocks * length(config.initial_cash_support),
        "reuse_same_shocks_at_each_cash" => true,
        "utility" => "sum(beta^(t-1)*log(consumption[t]), t=1:T)",
        "floor_visit" => "positive transfer on a transition, dates 1 through T-1")
end

function _utility_source_files()
    files = String["Project.toml", "Manifest.toml", "src/Ptd.jl",
        "src/simulation/policies/hsz.jl", "src/simulation/hsz/example.jl",
        "src/simulation/hsz/utility_comparison.jl",
        "src/simulation/hsz/grid_refinement.jl",
        "src/simulation/common/contest_statistics.jl"]
    for directory in ("src/ptd", "src/g2egm")
        for (root, _, names) in walkdir(joinpath(PROJECT_ROOT, directory))
            append!(files, [relpath(joinpath(root, name), PROJECT_ROOT)
                for name in names if endswith(name, ".jl")])
        end
    end
    return Dict(file => bytes2hex(SHA.sha256(read(joinpath(PROJECT_ROOT, file))))
        for file in sort!(unique(files)))
end

_utility_require(condition, message) = condition || throw(ArgumentError(message))

"""Build each method once, then evaluate every cash-by-shock trajectory."""
function _utility_method_paths(config, innovations, initial_cash;
    policy_builder=_build_worker_policies)
    model, process = _model(config)
    rule = gauss_hermite_rule(process, config.quadrature_nodes_per_dimension)
    grid = _g2_grid(config, process)
    paths = Dict{Symbol,HSZUtilityTrace}()
    for method in HSZ_EXAMPLE_METHODS
        println("HSZ utility: constructing ", _method_label(method), " policies")
        flush(stdout)
        policies = policy_builder(method, model, process, rule, grid)
        validate_hsz_contest_policy_alignment(first(policies), method, model, process)
        println("HSZ utility: evaluating ", length(initial_cash), " paths for ", _method_label(method))
        flush(stdout)
        result = _evaluate_method(method, policies, model, process, innovations, initial_cash)
        paths[method] = HSZUtilityTrace(result.cash, result.consumption, result.utility,
            result.floor_hits, result.total_transfers)
    end
    return paths
end

"""Reconstruct every budget transition, floor transfer and utility from saved actions."""
function _validate_utility_trace(trace, initial_cash, shock_row, innovations, config)
    horizon, count = config.horizon, length(initial_cash)
    _utility_require(size(trace.cash) == size(trace.consumption) == (horizon, count),
        "HSZ utility action dimensions disagree")
    _utility_require(length(trace.utility) == length(trace.floor_hits) ==
        length(trace.total_transfers) == count, "HSZ utility path dimensions disagree")
    _utility_require(all(isfinite, trace.cash) && all(isfinite, trace.consumption) &&
        all(isfinite, trace.utility) && all(isfinite, trace.total_transfers),
        "HSZ utility paths must be finite")
    model, process = _model(config)
    initial_state = HSZState(process.unconditional_log_earnings,
        process.unconditional_log_medical)
    for path in 1:count
        observed = ObservedState(1, initial_cash[path], initial_state)
        utility, discount, total_transfer, floor_hits = 0.0, 1.0, 0.0, 0
        for date in 1:horizon
            cash, consumption = trace.cash[date, path], trace.consumption[date, path]
            _utility_require(isapprox(cash, observed.cash; atol=1e-10, rtol=1e-12),
                "HSZ utility stored cash violates the reconstructed budget")
            _utility_require(0.0 < consumption <= cash,
                "HSZ utility contains an infeasible consumption action")
            utility += discount * log(consumption)
            if date == horizon
                _utility_require(isapprox(consumption, cash; atol=1e-10, rtol=1e-12),
                    "HSZ utility terminal consumption must exhaust cash")
                continue
            end
            row = shock_row[path]
            transition = true_transition(model, process, observed, consumption,
                (innovations[row, date, 1], innovations[row, date, 2]))
            floor_hits += transition.transfer > 0.0
            total_transfer += transition.transfer
            observed = next_observed(observed, transition)
            discount *= model.beta
        end
        _utility_require(isapprox(utility, trace.utility[path]; atol=1e-10, rtol=1e-12),
            "HSZ utility differs from reconstructed discounted log utility")
        _utility_require(floor_hits == trace.floor_hits[path],
            "HSZ utility floor counts differ from reconstructed transfers")
        _utility_require(isapprox(total_transfer, trace.total_transfers[path];
            atol=1e-10, rtol=1e-12), "HSZ utility transfer totals disagree")
    end
    return nothing
end

function _validate_utility_artifact(artifact; require_current_source=false)
    _utility_require(artifact.schema == HSZ_UTILITY_SCHEMA, "HSZ utility schema mismatch")
    config = _validate(artifact.config)
    shocks = artifact.innovations
    n_shocks = size(shocks, 1)
    _utility_require(n_shocks >= 2 && size(shocks) == (n_shocks, config.horizon - 1, 2) &&
        all(isfinite, shocks), "HSZ utility requires at least two finite shock paths")
    _utility_require(n_shocks == config.n_paths, "HSZ utility must reuse every source shock row")
    _utility_require(all(isfinite, config.initial_cash_support) &&
        length(unique(config.initial_cash_support)) == length(config.initial_cash_support),
        "HSZ utility starting cash values must be distinct and finite")
    expected_cash = repeat(config.initial_cash_support; inner=n_shocks)
    expected_rows = repeat(collect(1:n_shocks), length(config.initial_cash_support))
    _utility_require(artifact.initial_cash == expected_cash && artifact.shock_row == expected_rows,
        "HSZ utility must reuse the same ordered shock rows at every starting cash")
    _utility_require(artifact.scientific_fingerprint ==
        stable_fingerprint(_utility_scientific(config, n_shocks)),
        "HSZ utility scientific fingerprint mismatch")
    provenance = artifact.provenance
    _utility_require(provenance.source_scientific_fingerprint == _scientific_fingerprint(config),
        "HSZ utility source configuration mismatch")
    _utility_require(provenance.source_fingerprint == stable_fingerprint(provenance.source_files),
        "HSZ utility recorded solver fingerprint mismatch")
    if require_current_source
        _utility_require(provenance.source_files == _utility_source_files(),
            "HSZ utility was generated with different computational source files")
    end
    _utility_require(Set(keys(artifact.paths)) == Set(HSZ_EXAMPLE_METHODS),
        "HSZ utility method set mismatch")
    for method in HSZ_EXAMPLE_METHODS
        _validate_utility_trace(artifact.paths[method], artifact.initial_cash,
            artifact.shock_row, shocks, config)
    end
    return artifact
end

"""
    run_hsz_utility_comparison(source_path; config, output_path, overwrite=false)

Evaluate every source innovation row at every initial cash using log utility.
Policies are constructed once per method (one PTD instance per logical worker)
and then reused for all cash groups. The source example is never overwritten.
The default production design is five initial cash values by 100 common shock
paths, hence 500 trajectories per method, with 50 dates and Q5-by-Q5 quadrature.
"""
function run_hsz_utility_comparison(source_path::AbstractString;
    config::HSZExampleConfig=production_hsz_example_config(),
    output_path::AbstractString=default_hsz_utility_artifact_path(),
    overwrite::Bool=false)
    _validate(config)
    input, output = abspath(source_path), _validate_data_output(output_path)
    _utility_require(input != output && (!isfile(output) || realpath(input) != realpath(output)),
        "HSZ utility output must not overwrite its source artifact")
    _utility_require(overwrite || !isfile(output), "HSZ utility output already exists")
    source = load_hsz_example(input)
    _utility_require(source.scientific_configuration == _scientific_configuration(config) &&
        source.scientific_fingerprint == _scientific_fingerprint(config),
        "HSZ utility source must match the requested scientific configuration")
    innovations = copy(source.innovations)
    _utility_require(size(innovations) == (config.n_paths, config.horizon - 1, 2) &&
        all(isfinite, innovations), "HSZ source innovation dimensions are invalid")
    _utility_require(source.initial_cash == _initial_cash(config),
        "HSZ source initial cash allocation disagrees with its configuration")
    # Authenticate the source's scientific content before borrowing its shocks.
    for method in HSZ_EXAMPLE_METHODS
        _validate_utility_trace(getproperty(source.paths, method), source.initial_cash,
            collect(1:config.n_paths), innovations, config)
    end
    initial_cash = repeat(config.initial_cash_support; inner=config.n_paths)
    shock_row = repeat(collect(1:config.n_paths), length(config.initial_cash_support))
    repeated_innovations = innovations[shock_row, :, :]
    files = _utility_source_files()
    provenance = HSZUtilityProvenance(basename(input), bytes2hex(SHA.sha256(read(input))),
        source.scientific_fingerprint, files, stable_fingerprint(files),
        string(Dates.now(Dates.UTC)), string(VERSION), Threads.nthreads(), BLAS.get_num_threads())
    paths = _utility_method_paths(config, repeated_innovations, initial_cash)
    artifact = HSZUtilityArtifact(HSZ_UTILITY_SCHEMA, config,
        stable_fingerprint(_utility_scientific(config, config.n_paths)),
        provenance, innovations, initial_cash, shock_row, paths)
    _validate_utility_artifact(artifact; require_current_source=true)
    payload_fingerprint = stable_fingerprint(_utility_record(artifact))
    mkpath(dirname(output))
    jldsave(output; artifact, payload_fingerprint)
    return artifact
end

"""
Load saved HSZ paths, authenticate their digest, and reconstruct all outcomes.

Rendering uses the recorded solver identity. Set `require_current_source=true`
to additionally require that today's computational source matches that identity.
No policy is constructed by this loader or the renderer.
"""
function load_hsz_utility_comparison(path::AbstractString; require_current_source::Bool=false)
    payload = JLD2.load(abspath(path))
    _utility_require(haskey(payload, "artifact") && haskey(payload, "payload_fingerprint"),
        "HSZ utility file is missing its payload or fingerprint")
    artifact = payload["artifact"]
    _utility_require(artifact isa HSZUtilityArtifact, "HSZ utility payload type mismatch")
    _utility_require(stable_fingerprint(_utility_record(artifact)) == payload["payload_fingerprint"],
        "HSZ utility payload fingerprint mismatch")
    return _validate_utility_artifact(artifact; require_current_source)
end

"""Log-utility consumption-equivalent gap in percent for the supplied utility difference."""
function hsz_ce_percent(config::HSZExampleConfig, utility_difference::Real)
    config.gamma == 1.0 || throw(ArgumentError("HSZ CEG conversion requires log utility"))
    annuity = sum(config.beta ^ t for t in 0:config.horizon-1)
    return 100.0 * expm1(Float64(utility_difference) / annuity)
end

"""
Paired Student-t intervals with Bonferroni coverage across both G2EGM-versus-PTD
comparisons at every starting cash. The nominal family coverage is `level`.
`adjustment=:none` is available for explicitly labeled pointwise diagnostics.
The monotone log-utility CE transform is applied to both endpoints when rendering.
"""
function hsz_utility_summary(artifact::HSZUtilityArtifact; kwargs...)
    return _hsz_utility_summary(artifact.config, artifact.paths, artifact.initial_cash; kwargs...)
end

function _hsz_utility_summary(config, paths, initial_cash; level::Real=0.95,
    adjustment::Symbol=:bonferroni, family_size::Int=2length(config.initial_cash_support))
    0.0 < level < 1.0 || throw(ArgumentError("confidence level must lie in (0,1)"))
    family_size >= 2length(config.initial_cash_support) || throw(ArgumentError(
        "confidence family must include both comparisons at every initial cash"))
    adjustment in (:bonferroni, :none) || throw(ArgumentError("unknown interval adjustment"))
    marginal_level = adjustment === :bonferroni ? 1.0 - (1.0 - level) / family_size : Float64(level)
    rows = HSZUtilityRow[]
    for cash in config.initial_cash_support
        columns = findall(==(cash), initial_cash)
        means = Tuple(mean(view(paths[method].utility, columns))
            for method in HSZ_EXAMPLE_METHODS)
        counts = Tuple(count(>(0), view(paths[method].floor_hits, columns))
            for method in HSZ_EXAMPLE_METHODS)
        visits = Tuple(sum(view(paths[method].floor_hits, columns))
            for method in HSZ_EXAMPLE_METHODS)
        comparisons = Tuple(paired_comparison(:g2egm, method,
            view(paths[:g2egm].utility, columns),
            view(paths[method].utility, columns); level=marginal_level)
            for method in (:ptd_sceq, :ptd_osa))
        push!(rows, HSZUtilityRow(cash, length(columns), means, counts, visits, comparisons))
    end
    return rows
end

"""
Empirical P5, median, and P95 of pathwise consumption-equivalent gaps.

Each gap pairs PTD and G2EGM utility on the same shock path, transforms PTD
minus G2EGM utility into percent, and only then takes quantiles across paths
at a given initial cash. Positive gaps favor PTD: they scale G2EGM consumption
to match PTD utility. These summarize realized path variation, not confidence
intervals for an expected utility difference.
"""
function hsz_utility_quantiles(artifact::HSZUtilityArtifact)
    return _hsz_utility_quantiles(artifact.config, artifact.paths, artifact.initial_cash)
end

function _hsz_utility_quantiles(config, paths, initial_cash)
    return map(config.initial_cash_support) do cash
        columns = findall(==(cash), initial_cash)
        isempty(columns) && throw(ArgumentError("HSZ quantiles require paths at every initial cash"))
        floor_paths = Tuple(count(>(0), view(paths[method].floor_hits, columns))
            for method in HSZ_EXAMPLE_METHODS)
        ce_quantiles = Tuple(map((:ptd_sceq, :ptd_osa)) do method
            gaps = [hsz_ce_percent(config,
                paths[method].utility[column] - paths[:g2egm].utility[column])
                for column in columns]
            values = quantile(gaps, [0.05, 0.5, 0.95])
            (; p5=values[1], median=values[2], p95=values[3])
        end)
        (; initial_cash=cash, n_paths=length(columns), floor_paths, ce_quantiles)
    end
end

"""Write separate three-decimal cells for pathwise CEG quantiles, in percent."""
function _write_utility_ce_quantiles(io, row)
    for comparison in row.ce_quantiles, value in values(comparison)
        @printf(io, " & %.3f", value)
    end
    return nothing
end

"""Preserve small paired differences using three significant digits in LaTeX math."""
function _utility_tex_number(value::Float64)
    iszero(value) && return "0"
    formatted = @sprintf("%.3g", value)
    occursin('e', formatted) || return formatted
    mantissa, exponent = split(formatted, 'e')
    return mantissa * "\\times10^{" * string(parse(Int, exponent)) * "}"
end

"""Format an estimate and endpoints to three significant digits with one small-value scale."""
function _utility_tex_mean_interval(estimate::Float64, lower::Float64, upper::Float64)
    magnitude = max(abs(estimate), abs(lower), abs(upper))
    exponent = 0 < magnitude < 0.01 ? floor(Int, log10(magnitude)) : 0
    scale = 10.0^exponent
    group = _utility_tex_number(estimate / scale) * raw"\;[" *
        _utility_tex_number(lower / scale) * raw",\," *
        _utility_tex_number(upper / scale) * "]"
    return exponent == 0 ? group :
        raw"\bigl(" * group * raw"\bigr)\times10^{" * string(exponent) * "}"
end

"""Render the CE estimate and transformed interval endpoints in percent."""
function _utility_tex_ce_comparison(config::HSZExampleConfig, comparison::PairedComparison)
    return _utility_tex_mean_interval(hsz_ce_percent(config, comparison.mean_difference),
        hsz_ce_percent(config, comparison.ci_lower), hsz_ce_percent(config, comparison.ci_upper))
end

"""Use a common power of ten for small paired confidence intervals."""
function _utility_tex_interval(comparison::PairedComparison)
    magnitude = max(abs(comparison.ci_lower), abs(comparison.ci_upper))
    exponent = 0 < magnitude < 0.01 ? floor(Int, log10(magnitude)) : 0
    scale = 10.0^exponent
    interval = "[" * _utility_tex_number(comparison.ci_lower / scale) * ",\\," *
        _utility_tex_number(comparison.ci_upper / scale) * "]"
    return exponent == 0 ? interval : interval * "\\times10^{" * string(exponent) * "}"
end

function _write_utility_table(path, rows)
    open(path, "w") do io
        println(io, "% Generated by scripts/5_hsz.jl; do not edit.")
        println(io, "% Quantiles are across paired shock paths after the pathwise CEG transform.")
        println(io, "% Positive CEGs favor PTD; the scale is applied to G2EGM consumption.")
        println(io, raw"\begin{tabular}{rrrrrrrrrr}")
        println(io, raw"\toprule")
        println(io, raw"& \multicolumn{3}{c}{Floor-transfer paths} & \multicolumn{3}{c}{\makecell{PTD-SCEQ vs G2EGM\\CEG (\%)}} & \multicolumn{3}{c}{\makecell{PTD-OSA vs G2EGM\\CEG (\%)}}", " \\\\")
        println(io, raw"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
        println(io, raw"\makecell{Initial\\cash} & \makecell{PTD-\\SCEQ} & \makecell{PTD-\\OSA} & G2EGM & P5 & Median & P95 & P5 & Median & P95", " \\\\")
        println(io, raw"\midrule")
        for row in rows
            @printf(io, "%.2f & %d/%d & %d/%d & %d/%d",
                row.initial_cash, row.floor_paths[1], row.n_paths,
                row.floor_paths[2], row.n_paths, row.floor_paths[3], row.n_paths)
            _write_utility_ce_quantiles(io, row)
            println(io, " \\\\")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
end

"""Reject aliased input/output paths before opening any rendered product."""
function _validate_utility_render_paths(artifact_path, output_paths)
    paths = abspath.([artifact_path; collect(output_paths)])
    for path in paths[2:end]
        _validate_data_output(path)
        mkpath(dirname(path))
        _utility_require(!(islink(path) && !ispath(path)),
            "HSZ rendering does not write through dangling symbolic links")
    end
    # Resolve directory aliases even when an output file does not exist yet;
    # device/inode checks additionally detect existing hard links.
    destinations = [ispath(path) ? realpath(path) :
        joinpath(realpath(dirname(path)), basename(path)) for path in paths]
    for first in 1:length(paths)-1, second in first+1:length(paths)
        aliased = destinations[first] == destinations[second] ||
            Base.samefile(paths[first], paths[second])
        _utility_require(!aliased,
            "HSZ rendering requires distinct input and output files")
    end
    return nothing
end

"""Render the LaTeX utility table directly from the saved comparison without solving."""
function render_hsz_utility_comparison(artifact_path::AbstractString;
    table_path::AbstractString=tables_dir("hsz_utility_comparison.tex"))
    artifact = load_hsz_utility_comparison(artifact_path)
    rows = hsz_utility_quantiles(artifact)
    _validate_utility_render_paths(artifact_path, (table_path,))
    _write_utility_table(table_path, rows)
    return HSZUtilityOutput(abspath(artifact_path), abspath(table_path))
end
