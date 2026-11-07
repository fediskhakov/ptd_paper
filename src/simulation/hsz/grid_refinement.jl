import ..Ptd: portable_path

export run_hsz_grid_refinement, load_hsz_grid_refinement, render_hsz_grid_refinement,
       default_hsz_grid_refinement_path, compose_hsz_grid_refinement,
       paper_hsz_grid_refinement_paths

const HSZ_GRID_REFINEMENT_SCHEMA = "ptd-hsz-grid-refinement-v2"
"""Default artifact path for the HSZ exogenous-grid refinement exercise."""
default_hsz_grid_refinement_path() = joinpath(PROJECT_ROOT, "data", "hsz-grid-refinement.jld2")

_utility_record(value::NamedTuple) =
    Dict(String(key) => _utility_record(item) for (key, item) in pairs(value))

const HSZ_REFINEMENT_DRIVER = "src/simulation/hsz/grid_refinement.jl"

"""Authenticate identical baseline policy inputs, allowing only the refinement driver to differ."""
function _refinement_source_compatibility(baseline_files, refinement_files)
    _utility_require(Set(keys(baseline_files)) == Set(keys(refinement_files)) &&
        haskey(baseline_files, HSZ_REFINEMENT_DRIVER),
        "HSZ refinement source file set differs from the baseline")
    changed = sort!([path for path in keys(baseline_files)
        if baseline_files[path] != refinement_files[path]])
    _utility_require(all(==(HSZ_REFINEMENT_DRIVER), changed),
        "HSZ refinement policy sources differ from the baseline: " * join(changed, ", "))
    return Dict("mode" => isempty(changed) ? "identical_sources" : "refinement_driver_only",
        "driver_path" => HSZ_REFINEMENT_DRIVER,
        "baseline_driver_sha256" => baseline_files[HSZ_REFINEMENT_DRIVER],
        "refinement_driver_sha256" => refinement_files[HSZ_REFINEMENT_DRIVER],
        "reason" => "The refinement driver does not construct baseline policies or paths; " *
            "its NamedTuple serialization helper is unchanged. All other baseline source " *
            "files, including policy solvers, must match exactly.")
end

"""Compile only G2EGM construction and replay on a small disposable grid."""
function _warm_hsz_refinement!(config)
    _, process = _model(config)
    rule = gauss_hermite_rule(process, config.quadrature_nodes_per_dimension)
    model = StochasticPTDConfig(beta=config.beta, gamma=config.gamma,
        return_rate=config.return_rate, cash_floor=config.cash_floor, terminal_date=3)
    grid = G2EGM.hsz_g2egm_grid(process; asset_maximum=30.0,
        asset_points=9, common_cash_points=9,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5,
        earnings_radius=config.state_radius, medical_radius=config.state_radius,
        cash_floor=config.cash_floor, gross_return=1.0 + config.return_rate)
    policies = _build_worker_policies(:g2egm, model, process, rule, grid)
    _evaluate_method(:g2egm, policies, model, process,
        zeros(Float64, 1, 2, 2), [first(config.initial_cash_support)])
    return nothing
end

"""Change only the four exogenous grid sizes, preserving calibration and shocks."""
function _refined_hsz_config(config::HSZExampleConfig, points::Int)
    fields = Dict(name => getfield(config, name) for name in fieldnames(HSZExampleConfig))
    for name in (:post_earnings_points, :post_medical_points,
                 :common_earnings_points, :common_medical_points)
        fields[name] = points
    end
    return HSZExampleConfig(; fields...)
end

"""
Compare the saved baseline G2EGM paths with newly solved finer exogenous grids.
The baseline G2EGM and all PTD paths are reused exactly. Only requested finer
resolutions are solved, using the baseline utility experiment's cash-by-shock
design. Compilation uses a disposable small G2EGM grid. Construction and replay
are timed separately; validation is outside the timers. Baseline timings are
unavailable rather than borrowed from the differently sized path-figure run.
The baseline artifact and its provenance are never modified.
"""
function run_hsz_grid_refinement(baseline_path::AbstractString;
    finer_points::AbstractVector{<:Integer}=[18],
    output_path::AbstractString=default_hsz_grid_refinement_path(),
    overwrite::Bool=false)
    _validate_utility_render_paths(baseline_path, (output_path,))
    _utility_require(overwrite || !isfile(output_path), "HSZ refinement output already exists")
    baseline = load_hsz_utility_comparison(baseline_path)
    files = _utility_source_files()
    compatibility = _refinement_source_compatibility(baseline.provenance.source_files, files)
    baseline_sha256 = bytes2hex(SHA.sha256(read(baseline_path)))
    config = baseline.config
    baseline_points = config.common_earnings_points
    _utility_require(config.post_earnings_points == config.post_medical_points ==
        config.common_earnings_points == config.common_medical_points,
        "refinement requires a common starting resolution for all four exogenous grids")
    _utility_require(!isempty(finer_points) && all(>(baseline_points), finer_points) &&
        length(unique(finer_points)) == length(finer_points),
        "supply distinct exogenous resolutions finer than the baseline")
    println("HSZ refinement: reusing saved ", baseline_points, "x", baseline_points,
        " G2EGM baseline and all PTD paths; baseline timing is not remeasured")
    flush(stdout)
    _warm_hsz_refinement!(config)
    model, process = _model(config)
    rule = gauss_hermite_rule(process, config.quadrature_nodes_per_dimension)
    innovations = baseline.innovations[baseline.shock_row, :, :]
    runs = Dict{String,Any}(string(baseline_points) => Dict(
        "config" => config, "trace" => baseline.paths[:g2egm],
        "reused_baseline" => true, "construction_seconds" => nothing,
        "evaluation_seconds" => nothing, "diagnostics" => nothing,
        "evaluated_paths" => length(baseline.initial_cash)))
    for points in sort!(Int.(finer_points))
        refined = _refined_hsz_config(config, points)
        grid = _g2_grid(refined, process)
        println("HSZ refinement: constructing G2EGM on ", points, "x", points, " exogenous nodes")
        flush(stdout)
        started = time_ns()
        policies = _build_worker_policies(:g2egm, model, process, rule, grid)
        construction_seconds = (time_ns() - started) / 1e9
        validate_hsz_contest_policy_alignment(first(policies), :g2egm, model, process)
        started = time_ns()
        result = _evaluate_method(:g2egm, policies, model, process,
            innovations, baseline.initial_cash)
        evaluation_seconds = (time_ns() - started) / 1e9
        trace = HSZUtilityTrace(result.cash, result.consumption, result.utility,
            result.floor_hits, result.total_transfers)
        _validate_utility_trace(trace, baseline.initial_cash, baseline.shock_row,
            baseline.innovations, refined)
        runs[string(points)] = Dict("config" => refined, "trace" => trace,
            "reused_baseline" => false,
            "construction_seconds" => construction_seconds,
            "evaluation_seconds" => evaluation_seconds,
            "evaluated_paths" => length(baseline.initial_cash),
            "diagnostics" => G2EGM.diagnostics(first(policies).solution))
    end
    _utility_require(files == _utility_source_files(),
        "computational source changed during the grid-refinement run")
    _utility_require(baseline_sha256 == bytes2hex(SHA.sha256(read(baseline_path))),
        "baseline artifact changed during the grid-refinement run")
    artifact = Dict("schema" => HSZ_GRID_REFINEMENT_SCHEMA,
        "baseline_artifact" => basename(baseline_path),
        "baseline_sha256" => baseline_sha256,
        "baseline_source_compatibility" => compatibility,
        "source_files" => files, "created_utc" => string(Dates.now(Dates.UTC)),
        "julia_version" => string(VERSION), "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(), "runs" => runs)
    payload_fingerprint = stable_fingerprint(_utility_record(artifact))
    mkpath(dirname(abspath(output_path)))
    jldsave(abspath(output_path); artifact, payload_fingerprint)
    return artifact
end

"""Load and authenticate a refinement artifact against its saved baseline experiment."""
function load_hsz_grid_refinement(path::AbstractString, baseline_path::AbstractString;
    require_current_source::Bool=false)
    payload = JLD2.load(path)
    artifact = payload["artifact"]
    _utility_require(artifact["schema"] == HSZ_GRID_REFINEMENT_SCHEMA,
        "HSZ refinement schema mismatch")
    _utility_require(stable_fingerprint(_utility_record(artifact)) == payload["payload_fingerprint"],
        "HSZ refinement payload fingerprint mismatch")
    _utility_require(artifact["baseline_sha256"] == bytes2hex(SHA.sha256(read(baseline_path))),
        "HSZ refinement baseline artifact differs")
    baseline = load_hsz_utility_comparison(baseline_path)
    expected_compatibility = _refinement_source_compatibility(
        baseline.provenance.source_files, artifact["source_files"])
    _utility_require(artifact["baseline_source_compatibility"] == expected_compatibility,
        "HSZ refinement baseline source compatibility record differs")
    if require_current_source
        current = _utility_source_files()
        _refinement_source_compatibility(baseline.provenance.source_files, current)
        _utility_require(artifact["source_files"] == current,
            "HSZ refinement computational source differs")
    end
    baseline_points = baseline.config.common_earnings_points
    _utility_require(haskey(artifact["runs"], string(baseline_points)) && length(artifact["runs"]) >= 2,
        "HSZ refinement requires the saved baseline and at least one finer grid")
    for (key, run) in artifact["runs"]
        points = parse(Int, key)
        _utility_require(points >= baseline_points, "HSZ refinement contains a coarser grid")
        expected = _refined_hsz_config(baseline.config, points)
        _utility_require(_utility_record(expected) == _utility_record(run["config"]),
            "HSZ refinement changed more than the exogenous grids")
        _validate_utility_trace(run["trace"], baseline.initial_cash, baseline.shock_row,
            baseline.innovations, run["config"])
        _utility_require(run["evaluated_paths"] == length(baseline.initial_cash),
            "HSZ refinement path count differs from the baseline utility design")
        reused = points == baseline_points
        _utility_require(run["reused_baseline"] === reused,
            "HSZ refinement baseline reuse flag is inconsistent")
        if reused
            _utility_require(_utility_record(run["trace"]) == _utility_record(baseline.paths[:g2egm]),
                "HSZ refinement did not preserve the saved baseline G2EGM trace")
            _utility_require(isnothing(run["construction_seconds"]) &&
                isnothing(run["evaluation_seconds"]) && isnothing(run["diagnostics"]),
                "HSZ refinement reused baseline timings and diagnostics must remain unavailable")
        else
            _utility_require(all(x -> x isa Real && isfinite(x) && x >= 0.0,
                (run["construction_seconds"], run["evaluation_seconds"])),
                "HSZ refinement timings must be finite and nonnegative")
        end
    end
    return (; artifact, baseline)
end

"""
    paper_hsz_grid_refinement_paths()

Return the saved refinement artifacts behind the paper's footnote.

The grid-refinement footnote in Section 4.4.3 of the paper compares the
saved 17-node G2EGM paths of the HSZ utility comparison with 18- and
19-node exogenous grids. Each finer grid was solved in a separate run, so
each has its own artifact, and both artifacts repeat the reused 17-node
run. `scripts/5_hsz.jl --refinement --render-only` composes the two.

# Returns
- `Vector{String}`: the absolute paths of `data/hsz-grid-refinement.jld2`
  (17 and 18 nodes) and `data/hsz-grid-refinement-19.jld2` (17 and 19
  nodes) in this package.
"""
function paper_hsz_grid_refinement_paths()

    paths = [default_hsz_grid_refinement_path(),
        joinpath(PROJECT_ROOT, "data", "hsz-grid-refinement-19.jld2")]

    return paths
end

"""
    compose_hsz_grid_refinement(baseline_path, refinement_paths;
        table_path=tables_dir("hsz_grid_refinement.tex"),
        include_timings=false)

Write one LaTeX table that combines saved HSZ grid-refinement artifacts.

Nothing is solved. `load_hsz_grid_refinement` authenticates each artifact
against the baseline file and rebuilds every saved path from its actions.
Each artifact repeats the reused baseline grid; those copies must agree
exactly, and the table shows that grid once. A finer grid may come from
only one artifact. The table has one row per grid and initial cash,
ordered by the number of nodes. Each row shows the initial G2EGM
consumption, the number of paths with a G2EGM floor transfer, and the
G2EGM-minus-PTD consumption-equivalent gaps in percent with paired
intervals. All intervals are recomputed from the saved utilities with one
Bonferroni family that covers both PTD comparisons at every initial cash
for every displayed grid. The header comments name each input by its path
inside this package, or by its file name outside it, with its SHA-256;
they never show an absolute path. The table is written only after every
check has passed, and only if no input file changed meanwhile.

# Arguments
- `baseline_path::AbstractString`: the saved HSZ utility comparison that
  every artifact was run from (`data/hsz-utility-medical-1p5.jld2` for
  the paper). It holds the baseline G2EGM paths and the PTD-SCEQ and
  PTD-OSA paths that all grids share; each artifact records its SHA-256.
- `refinement_paths::AbstractVector{<:AbstractString}`: one or more
  artifacts written by `run_hsz_grid_refinement` from that baseline, each
  with the reused baseline grid and at least one finer grid. The paper
  uses `paper_hsz_grid_refinement_paths()`.
- `table_path::AbstractString`: the LaTeX file to write; an existing file
  is replaced. Defaults to `tables_dir("hsz_grid_refinement.tex")`.
- `include_timings::Bool`: if `true`, add a row after each grid with its
  archived G2EGM construction and replay times in seconds; the reused
  baseline grid has none. Every artifact must then record the same Julia
  and BLAS thread counts. Defaults to `false`: accuracy rows only.

# Returns
A `NamedTuple` with fields
- `table_path::String`: the absolute path of the written table;
- `family_size::Int`: the Bonferroni family size of every interval,
  2 x (initial cash levels) x (displayed grids); 30 for the paper;
- `rows::Vector{NamedTuple}`: the table rows in order, each with
  - `nodes::Int`: nodes in each of the four exogenous grids;
  - `initial_cash::Float64`: cash on hand at date 1;
  - `initial_consumption::Float64`: G2EGM consumption at date 1, the
    same on every path from this cash;
  - `g2egm_floor_paths::Int`: paths with at least one G2EGM floor
    transfer;
  - `n_paths::Int`: paths that start from this cash;
  - `comparisons::NTuple{2,PairedComparison}`: G2EGM minus PTD-SCEQ and
    G2EGM minus PTD-OSA discounted log utility, with paired intervals at
    the Bonferroni-adjusted level for 95% family coverage.
"""
function compose_hsz_grid_refinement(baseline_path::AbstractString,
    refinement_paths::AbstractVector{<:AbstractString};
    table_path::AbstractString=tables_dir("hsz_grid_refinement.tex"),
    include_timings::Bool=false)

    _utility_require(!isempty(refinement_paths),
        "an HSZ refinement table needs at least one refinement artifact")
    inputs = abspath.([baseline_path; collect(refinement_paths)])
    _utility_require(all(isfile, inputs),
        "every HSZ refinement table input must be an existing file")
    destination = abspath(table_path)
    _validate_utility_render_paths(first(inputs),
        (inputs[2:end]..., destination))
    digests = _refinement_input_sha256(inputs)
    loaded = [load_hsz_grid_refinement(path, first(inputs))
        for path in inputs[2:end]]
    baseline = first(loaded).baseline
    artifacts = [entry.artifact for entry in loaded]
    _utility_require(all(artifact -> artifact["baseline_sha256"] ==
            first(digests), artifacts),
        "every refinement artifact must authenticate the same baseline")
    if include_timings
        thread_counts = unique((artifact["julia_threads"],
            artifact["blas_threads"]) for artifact in artifacts)
        _utility_require(length(thread_counts) == 1,
            "archived refinement timings are comparable only with the " *
            "same Julia and BLAS thread counts")
    end
    grids = _refinement_grids(baseline, artifacts)
    family_size = 2 * length(baseline.config.initial_cash_support) *
        length(grids)
    rows = _refinement_rows(baseline, grids, family_size)
    composition = (; baseline, artifacts, grids, inputs, digests)
    buffer = IOBuffer()
    _write_refinement_table(buffer, composition, rows, family_size;
        include_timings)
    _utility_require(_refinement_input_sha256(inputs) == digests,
        "an HSZ refinement input changed while the table was composed")
    write(destination, take!(buffer))

    return (; table_path=destination, family_size, rows)
end

"""
    render_hsz_grid_refinement(path, baseline_path;
        table_path=tables_dir("hsz_grid_refinement.tex"),
        include_timings=false)

Write the grid-refinement table of one saved refinement artifact.

This is `compose_hsz_grid_refinement(baseline_path, [path]; ...)`: one
Bonferroni family covers both PTD comparisons at every initial cash for
the reused baseline grid and each finer grid in `path`.

# Arguments
- `path::AbstractString`: an artifact written by `run_hsz_grid_refinement`.
- `baseline_path::AbstractString`: the saved HSZ utility comparison that
  the artifact was run from.
- `table_path::AbstractString`: the LaTeX file to write; an existing file
  is replaced. Defaults to `tables_dir("hsz_grid_refinement.tex")`.
- `include_timings::Bool`: if `true`, add the archived timing rows.
  Defaults to `false`.

# Returns
- `String`: the absolute path of the written table.
"""
function render_hsz_grid_refinement(path::AbstractString,
    baseline_path::AbstractString;
    table_path::AbstractString=tables_dir("hsz_grid_refinement.tex"),
    include_timings::Bool=false)

    composed = compose_hsz_grid_refinement(baseline_path, [path];
        table_path, include_timings)

    return composed.table_path
end

"""
    _refinement_input_sha256(paths)

Hash the input files of a grid-refinement table.

# Arguments
- `paths::AbstractVector{<:AbstractString}`: the files to hash.

# Returns
- `Vector{String}`: the lowercase hexadecimal SHA-256 digest of each
  file's bytes, in the order of `paths`.
"""
function _refinement_input_sha256(paths)

    digests = [bytes2hex(SHA.sha256(read(path))) for path in paths]

    return digests
end

"""
    _refinement_comment_text(value)

Turn a value into text that fits on one LaTeX comment line.

# Arguments
- `value`: any value, usually a string read from an artifact.

# Returns
- `String`: `string(value)` with every line break replaced by a space.
"""
function _refinement_comment_text(value)

    text = replace(string(value), '\n' => ' ', '\r' => ' ')

    return text
end

"""
    _refinement_display_path(path)

Name an input file in a table comment without its absolute path.

The name is the one that `portable_path` saves in artifacts.

# Arguments
- `path::AbstractString`: an existing input file.

# Returns
- `String`: the file's path relative to this package, with `/`
  separators, when the file lies inside the package (for example
  `data/hsz-grid-refinement.jld2`); otherwise only its file name. Line
  breaks become spaces, so the name fits on one comment line.
"""
function _refinement_display_path(path::AbstractString)

    name = portable_path(path)

    return _refinement_comment_text(name)
end

"""
    _refinement_grids(baseline, artifacts)

Collect the grids that a grid-refinement table displays.

Every artifact repeats the reused baseline run. Those copies must agree
exactly, and the table shows the run once. Each finer grid may come from
only one artifact.

# Arguments
- `baseline::HSZUtilityArtifact`: the authenticated baseline comparison.
- `artifacts::AbstractVector`: the authenticated refinement artifacts,
  the dictionaries returned by `load_hsz_grid_refinement`.

# Returns
- `Vector{NamedTuple}`: one entry per displayed grid, ordered by nodes,
  with fields `nodes::Int` (nodes in each exogenous grid), `run` (the
  saved run dictionary described in `run_hsz_grid_refinement`) and
  `source::Int` (the index in `artifacts` of the artifact that holds the
  run; 1 for the reused baseline grid).
"""
function _refinement_grids(baseline, artifacts)

    baseline_key = string(baseline.config.common_earnings_points)
    reused_run = first(artifacts)["runs"][baseline_key]
    reused_record = _utility_record(reused_run)
    grids = NamedTuple[(; nodes=parse(Int, baseline_key), run=reused_run,
        source=1)]
    for (source, artifact) in enumerate(artifacts)
        _utility_require(isequal(
                _utility_record(artifact["runs"][baseline_key]),
                reused_record),
            "the reused baseline runs of the refinement artifacts differ")
        for (key, run) in artifact["runs"]
            key == baseline_key && continue
            nodes = parse(Int, key)
            _utility_require(all(grid -> grid.nodes != nodes, grids),
                "the $nodes-node grid appears in more than one " *
                "refinement artifact")
            push!(grids, (; nodes, run, source))
        end
    end
    sort!(grids; by=grid -> grid.nodes)

    return grids
end

"""
    _refinement_rows(baseline, grids, family_size)

Summarize every displayed grid at every initial cash.

The PTD-SCEQ and PTD-OSA paths come from the baseline; only the G2EGM
paths change with the grid. Every path from one initial cash starts in the
same state, so its saved date-1 G2EGM consumption must be the same.

# Arguments
- `baseline::HSZUtilityArtifact`: the authenticated baseline comparison.
- `grids::AbstractVector`: the displayed grids from `_refinement_grids`.
- `family_size::Int`: the Bonferroni family size of every interval.

# Returns
- `Vector{NamedTuple}`: the table rows in order; their fields are listed
  under `compose_hsz_grid_refinement`.
"""
function _refinement_rows(baseline, grids, family_size)

    g2egm = findfirst(==(:g2egm), HSZ_EXAMPLE_METHODS)
    rows = NamedTuple[]
    for grid in grids
        trace = grid.run["trace"]
        paths = copy(baseline.paths)
        paths[:g2egm] = trace
        summaries = _hsz_utility_summary(grid.run["config"], paths,
            baseline.initial_cash; level=0.95, adjustment=:bonferroni,
            family_size)
        for summary in summaries
            columns = findall(==(summary.initial_cash),
                baseline.initial_cash)
            initial = trace.consumption[1, columns]
            _utility_require(allequal(initial),
                "G2EGM date-1 consumption differs across paths " *
                "from the same initial cash")
            push!(rows, (; nodes=grid.nodes,
                initial_cash=summary.initial_cash,
                initial_consumption=first(initial),
                g2egm_floor_paths=summary.floor_paths[g2egm],
                n_paths=summary.n_paths,
                comparisons=summary.comparisons))
        end
    end

    return rows
end

"""
    _write_refinement_table(io, composition, rows, family_size;
        include_timings)

Write the grid-refinement table and its provenance comments to `io`.

# Arguments
- `io::IO`: the destination stream.
- `composition::NamedTuple`: the authenticated inputs, with fields
  `baseline` (the `HSZUtilityArtifact`), `artifacts` (the refinement
  artifact dictionaries), `grids` (from `_refinement_grids`), `inputs`
  (absolute paths: the baseline, then each artifact) and `digests`
  (their SHA-256 digests, in the same order).
- `rows::AbstractVector`: the rows from `_refinement_rows`.
- `family_size::Int`: the Bonferroni family size of the intervals.
- `include_timings::Bool`: whether to add the archived timing rows.

# Returns
- `Nothing`.
"""
function _write_refinement_table(io, composition, rows, family_size;
    include_timings::Bool)

    (; baseline, artifacts, grids, inputs, digests) = composition
    println(io, "% Generated by scripts/5_hsz.jl --refinement; ",
        "do not edit.")
    println(io, "% Baseline artifact: ",
        _refinement_display_path(first(inputs)))
    println(io, "% Baseline SHA-256: ", first(digests))
    println(io, "% Baseline policy source fingerprint: ",
        baseline.provenance.source_fingerprint)
    for grid in grids
        grid.run["reused_baseline"] && continue
        artifact = artifacts[grid.source]
        label = "% Grid " * string(grid.nodes)
        println(io, label, " artifact: ",
            _refinement_display_path(inputs[grid.source + 1]))
        println(io, label, " artifact SHA-256: ", digests[grid.source + 1])
        println(io, label, " authenticated source fingerprint: ",
            stable_fingerprint(artifact["source_files"]))
        println(io, label, " created UTC: ",
            _refinement_comment_text(artifact["created_utc"]))
        println(io, label, " Julia version/threads/BLAS threads: ",
            _refinement_comment_text(artifact["julia_version"]), "/",
            artifact["julia_threads"], "/", artifact["blas_threads"])
    end
    println(io, "% All intervals recomputed from saved traces; original ",
        "means, counts, and measured times retained.")
    println(io, "% Initial G2EGM consumption is date-1 consumption, ",
        "the same on every path from that initial cash.")
    println(io, raw"\begin{tabular}{rrrrcc}")
    println(io, raw"\toprule")
    println(io, raw"Nodes & \makecell{Initial\\cash} & ",
        raw"\makecell{Initial G2EGM\\consumption} & ",
        raw"\makecell{G2EGM\\floor paths} & ",
        raw"G2EGM vs SCEQ (\%) & G2EGM vs OSA (\%)", " \\\\")
    println(io, raw"\midrule")
    for grid in grids
        for row in rows
            row.nodes == grid.nodes &&
                _write_refinement_row(io, baseline.config, row)
        end
        include_timings && _write_refinement_timing(io, grid)
    end
    println(io, raw"\bottomrule")
    println(io, raw"\end{tabular}")
    println(io, "% Bonferroni family size: ", family_size,
        "; 95% family coverage; all displayed grids, cash levels, ",
        "and both PTD comparisons.")

    return nothing
end

"""
    _write_refinement_row(io, config, row)

Write one grid-and-cash row of the grid-refinement table.

# Arguments
- `io::IO`: the destination stream.
- `config::HSZExampleConfig`: the baseline configuration, whose discount
  factor and horizon turn utility differences into consumption-equivalent
  percentages.
- `row::NamedTuple`: one row from `_refinement_rows`.

# Returns
- `Nothing`.
"""
function _write_refinement_row(io, config, row)

    @printf(io, "%d & %.2f & %.3f & %d/%d", row.nodes, row.initial_cash,
        row.initial_consumption, row.g2egm_floor_paths, row.n_paths)
    for comparison in row.comparisons
        print(io, " & \$", _utility_tex_ce_comparison(config, comparison),
            "\$")
    end
    println(io, " \\\\")

    return nothing
end

"""
    _write_refinement_timing(io, grid)

Write the archived timing row of one displayed grid.

# Arguments
- `io::IO`: the destination stream.
- `grid::NamedTuple`: one entry from `_refinement_grids`. The reused
  baseline run has no timings, so its row says so.

# Returns
- `Nothing`.
"""
function _write_refinement_timing(io, grid)

    run = grid.run
    if run["reused_baseline"]
        print(io, raw"\multicolumn{6}{l}{Baseline paths reused; ",
            "timing not remeasured.}")
    else
        construction = @sprintf("%.3f", run["construction_seconds"])
        replay = @sprintf("%.3f", run["evaluation_seconds"])
        print(io, raw"\multicolumn{6}{l}{Saved ", grid.nodes,
            "-node run: construction ", construction, " s; replay (",
            run["evaluated_paths"], " paths) ", replay, " s.}")
    end
    println(io, " \\\\")

    return nothing
end
