"""Declared tensor grids for the AR(1) G2EGM solve."""
struct AR1G2EGMGrid
    asset::Vector{Float64}
    post_log_income::Vector{Float64}
    cash::Vector{Float64}
    log_income::Vector{Float64}

    function AR1G2EGMGrid(asset, post_log_income, cash, log_income)
        a = collect(Float64, asset)
        post_ell = collect(Float64, post_log_income)
        x = collect(Float64, cash)
        ell = collect(Float64, log_income)
        for (grid, name) in ((a, "asset"), (post_ell, "post-decision log-income"),
                             (x, "cash"), (ell, "common log-income"))
            length(grid) >= 2 || throw(ArgumentError(
                "the $name grid needs at least two nodes"))
            all(isfinite, grid) && all(diff(grid) .> 0.0) ||
                throw(ArgumentError(
                    "the $name grid must be finite and strictly increasing"))
        end
        first(a) == 0.0 || throw(ArgumentError(
            "the post-decision asset grid must begin at zero"))
        first(post_ell) == first(ell) && last(post_ell) == last(ell) ||
            throw(ArgumentError(
                "post-decision and common log-income grids must share endpoints"))
        return new(a, post_ell, x, ell)
    end
end

"""Numerical controls for local-triangle reconstruction and upper envelopes."""
Base.@kwdef struct AR1G2EGMOptions
    bounding_box_padding::Int = 2
    minimum_barycentric_weight::Float64 = -0.25
    hole_search_radius::Int = 2
    value_atol::Float64 = 1e-11
    value_rtol::Float64 = 1e-10
    determinant_tolerance::Float64 = 1e-14
    reject_mixed_floor_regimes::Bool = false
    enforce_monotone_saving::Bool = true
    monotonicity_atol::Float64 = 1e-10
    monotonicity_rtol::Float64 = 1e-10
    direct_iterations::Int = 80
end

function _validate_options(options::AR1G2EGMOptions)
    options.bounding_box_padding >= 0 || throw(ArgumentError(
        "bounding_box_padding must be nonnegative"))
    -1.0 <= options.minimum_barycentric_weight <= 0.0 ||
        throw(ArgumentError(
            "minimum_barycentric_weight must lie in [-1,0]"))
    options.hole_search_radius >= 0 || throw(ArgumentError(
        "hole_search_radius must be nonnegative"))
    all(isfinite, (options.value_atol, options.value_rtol,
                   options.determinant_tolerance,
                   options.monotonicity_atol,
                   options.monotonicity_rtol)) &&
        options.value_atol >= 0.0 && options.value_rtol >= 0.0 &&
        options.determinant_tolerance >= 0.0 &&
        options.monotonicity_atol >= 0.0 &&
        options.monotonicity_rtol >= 0.0 || throw(ArgumentError(
            "all G2EGM tolerances must be finite and nonnegative"))
    options.direct_iterations >= 16 || throw(ArgumentError(
        "direct_iterations must be at least 16"))
    return options
end

"""Counters accumulated during a complete backward G2EGM solve."""
Base.@kwdef mutable struct G2EGMSolveDiagnostics
    quadrature_expectations::Int = 0
    continuation_queries::Int = 0
    triangles_considered::Int = 0
    triangles_accepted::Int = 0
    triangles_degenerate::Int = 0
    triangles_rejected_regime::Int = 0
    triangles_mixed_floor_regime::Int = 0
    triangle_candidates::Int = 0
    triangle_extrapolated_candidates::Int = 0
    triangle_winners::Int = 0
    triangle_extrapolated_winners::Int = 0
    minimum_winning_barycentric_weight::Float64 = Inf
    interior_holes_before_fill::Int = 0
    interior_holes_filled::Int = 0
    constraint_winners::Int = 0
    floor_kink_candidates::Int = 0
    floor_kink_winners::Int = 0
    final_q5_candidate_revaluations::Int = 0
    monotonicity_violations::Int = 0
    monotonicity_repairs::Int = 0
    direct_repair_winners::Int = 0
    direct_objective_evaluations::Int = 0
    monotonicity_violations_remaining::Int = 0
    asset_ceiling_candidates::Int = 0
    asset_ceiling_winners::Int = 0
    policy_domain_asset_ceiling_winners::Int = 0
    continuation_padding_asset_ceiling_winners::Int = 0
    minimum_solve_cash_margin::Float64 = Inf
    minimum_solve_log_income_margin::Float64 = Inf
end

"""Fail-closed query-domain and boundary diagnostics."""
mutable struct G2EGMQueryDiagnostics
    consumption_queries::Int
    value_queries::Int
    domain_errors::Int
    lower_cash_contacts::Int
    upper_cash_contacts::Int
    lower_log_income_contacts::Int
    upper_log_income_contacts::Int
    asset_ceiling_choices::Int
    minimum_cash_margin::Float64
    minimum_log_income_margin::Float64
    lock::ReentrantLock
end

G2EGMQueryDiagnostics() = G2EGMQueryDiagnostics(
    0, 0, 0, 0, 0, 0, 0, 0, Inf, Inf, ReentrantLock())

"""Solved policy and value arrays on the declared common tensor grid."""
struct AR1G2EGMSolution{R<:GaussHermiteRule{1,5}}
    config::StochasticPTDConfig
    process::LogAR1Process
    base_rule::R
    grid::AR1G2EGMGrid
    options::AR1G2EGMOptions
    policy::Vector{Matrix{Float64}}
    value::Vector{Matrix{Float64}}
    solve_diagnostics::G2EGMSolveDiagnostics
    query_diagnostics::G2EGMQueryDiagnostics
    fingerprint::String
end

function _single_log_grid(lower::Float64, upper::Float64, points::Int)
    points >= 2 || throw(ArgumentError("a single-log grid needs at least two points"))
    upper > lower || throw(ArgumentError("a grid upper bound must exceed its lower bound"))
    width = upper - lower
    grid = lower .+ exp.(range(0.0, log1p(width); length=points)) .- 1.0
    grid[1] = lower
    grid[end] = upper
    all(diff(grid) .> 0.0) || throw(ArgumentError(
        "the requested single-log grid is not strictly increasing in Float64"))
    return grid
end

function _sinh_grid(lower::Float64, upper::Float64, points::Int,
                    curvature::Float64)
    points >= 2 || throw(ArgumentError("a sinh grid needs at least two points"))
    upper > lower || throw(ArgumentError("a grid upper bound must exceed its lower bound"))
    isfinite(curvature) && curvature >= 0.0 || throw(ArgumentError(
        "sinh-grid curvature must be finite and nonnegative"))
    coordinate = range(-1.0, 1.0; length=points)
    transformed = curvature == 0.0 ? collect(coordinate) :
        sinh.(curvature .* coordinate) ./ sinh(curvature)
    midpoint = (lower + upper) / 2.0
    halfwidth = (upper - lower) / 2.0
    grid = midpoint .+ halfwidth .* transformed
    grid[1] = lower
    grid[end] = upper
    return grid
end

"""
    ar1_g2egm_grid(; ...)

Build an AR(1) G2EGM grid without inspecting evaluator shocks. The defaults
define the inexpensive pilot grid; the paper-facing runner passes the larger
grid declared in the manuscript explicitly.
"""
function ar1_g2egm_grid(; asset_maximum::Real=30_000.0,
    asset_points::Integer=401, common_cash_points::Integer=301,
    post_log_income_points::Integer=31,
    common_log_income_points::Integer=21,
    cash_floor::Real=3.0, gross_return::Real=1.10,
    log_income_bounds::Tuple{<:Real,<:Real}=(-3.0, 3.0),
    post_log_income_curvature::Real=3.0,
    common_log_income_curvature::Real=2.5)

    amax = Float64(asset_maximum)
    floor = Float64(cash_floor)
    gross = Float64(gross_return)
    ell_lower, ell_upper = Float64.(log_income_bounds)
    all(isfinite, (amax, floor, gross, ell_lower, ell_upper)) ||
        throw(ArgumentError("all G2EGM grid bounds must be finite"))
    amax > 0.0 || throw(ArgumentError("asset_maximum must be positive"))
    floor >= 0.0 || throw(ArgumentError("cash_floor must be nonnegative"))
    gross > 0.0 || throw(ArgumentError("gross_return must be positive"))
    ell_lower < ell_upper || throw(ArgumentError(
        "log-income bounds must be ordered"))
    cash_maximum = gross * amax + exp(ell_upper)
    asset = _single_log_grid(0.0, amax, Int(asset_points))
    cash = _single_log_grid(floor, cash_maximum, Int(common_cash_points))
    post_ell = _sinh_grid(ell_lower, ell_upper,
        Int(post_log_income_points), Float64(post_log_income_curvature))
    ell = _sinh_grid(ell_lower, ell_upper,
        Int(common_log_income_points), Float64(common_log_income_curvature))
    return AR1G2EGMGrid(asset, post_ell, cash, ell)
end

_bits(value::Float64) = string(reinterpret(UInt64, value); base=16, pad=16)
_grid_token(grid::Vector{Float64}) = join(_bits.(grid), ',')

function _options_token(options::AR1G2EGMOptions)
    return join((options.bounding_box_padding,
                 _bits(options.minimum_barycentric_weight),
                 options.hole_search_radius,
                 _bits(options.value_atol), _bits(options.value_rtol),
                 _bits(options.determinant_tolerance),
                 options.reject_mixed_floor_regimes,
                 options.enforce_monotone_saving,
                 _bits(options.monotonicity_atol),
                 _bits(options.monotonicity_rtol),
                 options.direct_iterations), '|')
end

function _solution_input_fingerprint(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid, options::AR1G2EGMOptions)
    tokens = String[
        G2EGM_SOLVER_SCHEMA,
        G2EGM_REFERENCE_COMMIT,
        _bits(config.beta), _bits(config.gamma), _bits(config.return_rate),
        _bits(config.cash_floor), string(config.terminal.final_date),
        _bits(process.rho), _bits(process.log_std),
        _bits(process.unconditional_log_mean),
        _grid_token(grid.asset), _grid_token(grid.post_log_income),
        _grid_token(grid.cash), _grid_token(grid.log_income),
        _options_token(options),
    ]
    append!(tokens, (_bits(Float64(only(node))) for node in rule.nodes))
    append!(tokens, (_bits(Float64(weight)) for weight in rule.weights))
    return bytes2hex(SHA.sha256(join(tokens, '\n')))
end

function _update_solution_array_hash!(context, label::AbstractString,
    arrays::Vector{Matrix{Float64}})
    SHA.update!(context, collect(codeunits(
        "$label|$(length(arrays))|Float64|network-order\n")))
    buffer = Vector{UInt64}(undef, 8_192)
    for (date, values) in enumerate(arrays)
        SHA.update!(context, collect(codeunits(
            "$date|$(size(values, 1))|$(size(values, 2))\n")))
        used = 0
        for value in values
            used += 1
            buffer[used] = hton(reinterpret(UInt64, value))
            if used == length(buffer)
                SHA.update!(context, reinterpret(UInt8, buffer))
                used = 0
            end
        end
        used > 0 && SHA.update!(context,
            reinterpret(UInt8, view(buffer, 1:used)))
    end
    return context
end

function _solution_output_fingerprint(policy::Vector{Matrix{Float64}},
    value::Vector{Matrix{Float64}})
    length(policy) == length(value) || throw(DimensionMismatch(
        "G2EGM policy and value horizons differ"))
    context = SHA.SHA2_256_CTX()
    SHA.update!(context, collect(codeunits(
        "ptd-ar1-g2egm-solution-arrays-v1\n")))
    _update_solution_array_hash!(context, "policy", policy)
    _update_solution_array_hash!(context, "value", value)
    return bytes2hex(SHA.digest!(context))
end

function _complete_solution_fingerprint(input_fingerprint::String,
    policy::Vector{Matrix{Float64}}, value::Vector{Matrix{Float64}})
    output_fingerprint = _solution_output_fingerprint(policy, value)
    return bytes2hex(SHA.sha256(join((
        "ptd-ar1-g2egm-complete-solution-v1",
        input_fingerprint,
        output_fingerprint,
    ), '\n')))
end

"""SHA-256 identity binding solver inputs and every computed policy/value bit."""
solution_fingerprint(solution::AR1G2EGMSolution) = solution.fingerprint

function _search_interval(grid::Vector{Float64}, value::Float64)
    index = searchsortedlast(grid, value)
    index == 0 && return 0
    index >= length(grid) && return value == last(grid) ? length(grid) - 1 : length(grid)
    return index
end

function _bilinear(xgrid::Vector{Float64}, ygrid::Vector{Float64},
    values::Matrix{Float64}, x::Float64, y::Float64)
    ix = _search_interval(xgrid, x)
    iy = _search_interval(ygrid, y)
    1 <= ix < length(xgrid) || throw(DomainError(x,
        "first interpolation coordinate is outside the declared grid"))
    1 <= iy < length(ygrid) || throw(DomainError(y,
        "second interpolation coordinate is outside the declared grid"))
    wx = (x - xgrid[ix]) / (xgrid[ix + 1] - xgrid[ix])
    wy = (y - ygrid[iy]) / (ygrid[iy + 1] - ygrid[iy])
    return (1.0 - wx) * ((1.0 - wy) * values[ix, iy] +
                         wy * values[ix, iy + 1]) +
           wx * ((1.0 - wy) * values[ix + 1, iy] +
                 wy * values[ix + 1, iy + 1])
end

function _query_tolerance(value::Float64, lower::Float64, upper::Float64)
    return 64eps(max(1.0, abs(value), abs(lower), abs(upper)))
end

function _checked_coordinate(value::Float64, lower::Float64, upper::Float64,
    query_diagnostics::G2EGMQueryDiagnostics, kind::Symbol)
    tolerance = _query_tolerance(value, lower, upper)
    if value < lower - tolerance || value > upper + tolerance
        lock(query_diagnostics.lock) do
            query_diagnostics.domain_errors += 1
        end
        throw(DomainError(value,
            "$kind query is outside the solved G2EGM domain [$lower,$upper]"))
    end
    return clamp(value, lower, upper)
end

function _record_query!(solution::AR1G2EGMSolution, cash::Float64,
    log_income::Float64; value_query::Bool, terminal::Bool)
    qd = solution.query_diagnostics
    xlower = first(solution.grid.cash)
    # The part of the common cash grid above the asset-grid endpoint is
    # continuation padding: it is needed to evaluate x' = R a + y', but its
    # current-period policy can be mechanically capped by a_max.  Only expose
    # that padding at the analytic terminal date, where there is no saving
    # choice. Nonterminal action/value queries therefore fail closed above the
    # largest common-cash node at or below a_max; this also prevents the final
    # interpolation interval from borrowing an unaudited padding policy.
    policy_last = searchsortedlast(
        solution.grid.cash, last(solution.grid.asset))
    xupper = terminal ? last(solution.grid.cash) :
             solution.grid.cash[policy_last]
    elllower, ellupper = extrema(solution.grid.log_income)
    x = _checked_coordinate(cash, xlower, xupper, qd, :cash)
    ell = _checked_coordinate(log_income, elllower, ellupper, qd, :log_income)
    xtol = _query_tolerance(x, xlower, xupper)
    elltol = _query_tolerance(ell, elllower, ellupper)
    lock(qd.lock) do
        value_query ? (qd.value_queries += 1) : (qd.consumption_queries += 1)
        abs(x - xlower) <= xtol && (qd.lower_cash_contacts += 1)
        abs(x - xupper) <= xtol && (qd.upper_cash_contacts += 1)
        abs(ell - elllower) <= elltol && (qd.lower_log_income_contacts += 1)
        abs(ell - ellupper) <= elltol && (qd.upper_log_income_contacts += 1)
        qd.minimum_cash_margin = min(qd.minimum_cash_margin,
            x - xlower, xupper - x)
        qd.minimum_log_income_margin = min(qd.minimum_log_income_margin,
            ell - elllower, ellupper - ell)
    end
    return x, ell
end

"""Interpolate feasible consumption from a solved G2EGM policy."""
function consumption(solution::AR1G2EGMSolution, cash::Real,
    log_income::Real; date::Integer)
    t = Int(date)
    1 <= t <= solution.config.terminal.final_date || throw(ArgumentError(
        "date is outside the solved finite horizon"))
    xraw, ellraw = Float64(cash), Float64(log_income)
    all(isfinite, (xraw, ellraw)) || throw(ArgumentError(
        "G2EGM query coordinates must be finite"))
    x, ell = _record_query!(solution, xraw, ellraw;
        value_query=false, terminal=t == solution.config.terminal.final_date)
    t == solution.config.terminal.final_date && return x
    c = _bilinear(solution.grid.cash, solution.grid.log_income,
        solution.policy[t], x, ell)
    c = clamp(c, 0.0, x)
    saving = x - c
    tolerance = _query_tolerance(saving, 0.0, last(solution.grid.asset))
    if saving >= last(solution.grid.asset) - tolerance
        lock(solution.query_diagnostics.lock) do
            solution.query_diagnostics.asset_ceiling_choices += 1
        end
    end
    return c
end

"""Interpolate the value function from a solved G2EGM policy."""
function value_at(solution::AR1G2EGMSolution, cash::Real,
    log_income::Real; date::Integer)
    t = Int(date)
    1 <= t <= solution.config.terminal.final_date || throw(ArgumentError(
        "date is outside the solved finite horizon"))
    xraw, ellraw = Float64(cash), Float64(log_income)
    all(isfinite, (xraw, ellraw)) || throw(ArgumentError(
        "G2EGM query coordinates must be finite"))
    x, ell = _record_query!(solution, xraw, ellraw;
        value_query=true, terminal=t == solution.config.terminal.final_date)
    if t == solution.config.terminal.final_date
        return _g2egm_utility(solution.config.gamma, x)
    end
    return _bilinear(solution.grid.cash, solution.grid.log_income,
        solution.value[t], x, ell)
end

"""Return the G2EGM action for an observed stochastic state."""
function action(solution::AR1G2EGMSolution,
    observed::ObservedState{LogAR1State})
    observed.date <= solution.config.terminal.final_date || throw(ArgumentError(
        "observed date is beyond the solved horizon"))
    return consumption(solution, observed.cash, observed.exogenous.log_income;
        date=observed.date)
end

"""Reset query diagnostics stored on a solved G2EGM model."""
function reset_query_diagnostics!(solution::AR1G2EGMSolution)
    qd = solution.query_diagnostics
    lock(qd.lock) do
        qd.consumption_queries = 0
        qd.value_queries = 0
        qd.domain_errors = 0
        qd.lower_cash_contacts = 0
        qd.upper_cash_contacts = 0
        qd.lower_log_income_contacts = 0
        qd.upper_log_income_contacts = 0
        qd.asset_ceiling_choices = 0
        qd.minimum_cash_margin = Inf
        qd.minimum_log_income_margin = Inf
    end
    return qd
end

"""Return serializable construction and query diagnostics for G2EGM."""
function diagnostics(solution::AR1G2EGMSolution)
    sd = solution.solve_diagnostics
    qd = solution.query_diagnostics
    query = lock(qd.lock) do
        (; consumption_queries=qd.consumption_queries,
           value_queries=qd.value_queries,
           domain_errors=qd.domain_errors,
           lower_cash_contacts=qd.lower_cash_contacts,
           upper_cash_contacts=qd.upper_cash_contacts,
           lower_log_income_contacts=qd.lower_log_income_contacts,
           upper_log_income_contacts=qd.upper_log_income_contacts,
           asset_ceiling_choices=qd.asset_ceiling_choices,
           minimum_cash_margin=qd.minimum_cash_margin,
           minimum_log_income_margin=qd.minimum_log_income_margin)
    end
    solve = (; (name => getfield(sd, name) for name in fieldnames(G2EGMSolveDiagnostics))...)
    return (; solve, query)
end

"""Describe the quadrature rule embedded in a G2EGM solution."""
function quadrature_audit(solution::AR1G2EGMSolution)
    return (;
        family=:positive_gauss_hermite,
        nodes_per_dimension=5,
        standard_normal_nodes=Tuple(only(node) for node in solution.base_rule.nodes),
        normalized_weights=Tuple(solution.base_rule.weights),
        weight_sum=sum(solution.base_rule.weights),
    )
end
