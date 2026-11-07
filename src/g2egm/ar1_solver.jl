function _g2egm_utility(gamma::Float64, consumption::Float64)
    consumption > 0.0 || return -Inf
    return gamma == 1.0 ? log(consumption) :
           (consumption^(1.0 - gamma) - 1.0) / (1.0 - gamma)
end

_g2egm_marginal_utility(gamma::Float64, consumption::Float64) =
    consumption^(-gamma)

function _g2egm_inverse_marginal(gamma::Float64, marginal::Float64)
    isfinite(marginal) && marginal > 0.0 || return NaN
    return marginal^(-1.0 / gamma)
end

function _rules_equal(left::GaussHermiteRule{1,5},
    right::GaussHermiteRule{1,5})
    return left.nodes == right.nodes && left.weights == right.weights &&
           left.nodes_per_dimension == right.nodes_per_dimension
end

function _validate_grid_for_model(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid)

    first(grid.cash) == config.cash_floor || throw(ArgumentError(
        "the common cash grid must begin at the post-transfer cash floor"))
    gross = 1.0 + config.return_rate
    required_upper = gross * last(grid.asset) + exp(last(grid.post_log_income))
    last(grid.cash) >= required_upper - 64eps(required_upper) ||
        throw(ArgumentError(
            "the common cash grid does not contain the declared post-decision transition domain"))

    ell_lower, ell_upper = extrema(grid.log_income)
    mean = process.unconditional_log_mean
    for ell in (first(grid.post_log_income), last(grid.post_log_income)),
        node in rule.nodes
        next_ell = mean + process.rho * (ell - mean) +
                   process.log_std * only(node)
        ell_lower <= next_ell <= ell_upper || throw(ArgumentError(
            "the Q5 AR(1) transition leaves the declared log-income domain"))
    end
    return grid
end

function _record_solve_query!(diagnostics::G2EGMSolveDiagnostics,
    grid::AR1G2EGMGrid, cash::Float64, log_income::Float64)
    cash_lower, cash_upper = extrema(grid.cash)
    ell_lower, ell_upper = extrema(grid.log_income)
    cash_lower <= cash <= cash_upper || throw(DomainError(cash,
        "a continuation cash query left the common G2EGM grid"))
    ell_lower <= log_income <= ell_upper || throw(DomainError(log_income,
        "a continuation log-income query left the common G2EGM grid"))
    diagnostics.continuation_queries += 1
    diagnostics.minimum_solve_cash_margin = min(
        diagnostics.minimum_solve_cash_margin,
        cash - cash_lower, cash_upper - cash)
    diagnostics.minimum_solve_log_income_margin = min(
        diagnostics.minimum_solve_log_income_margin,
        log_income - ell_lower, ell_upper - log_income)
    return nothing
end

function _expected_continuation(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid, child_value::Matrix{Float64},
    child_policy::Matrix{Float64}, saving::Float64, log_income::Float64,
    diagnostics::G2EGMSolveDiagnostics; marginal::Bool)

    gross = 1.0 + config.return_rate
    mean = process.unconditional_log_mean
    expected_value = 0.0
    expected_marginal = 0.0
    signature = UInt8(0)
    diagnostics.quadrature_expectations += 1
    @inbounds for q in eachindex(rule.weights)
        next_log_income = mean + process.rho * (log_income - mean) +
                          process.log_std * only(rule.nodes[q])
        next_income = exp(next_log_income)
        pretransfer = gross * saving + next_income
        nonbinding = pretransfer > config.cash_floor
        nonbinding && (signature |= UInt8(1) << (q - 1))
        next_cash = max(pretransfer, config.cash_floor)
        _record_solve_query!(
            diagnostics, grid, next_cash, next_log_income)
        child_v = _bilinear(grid.cash, grid.log_income,
            child_value, next_cash, next_log_income)
        expected_value += rule.weights[q] * child_v
        if marginal && nonbinding
            child_c = _bilinear(grid.cash, grid.log_income,
                child_policy, next_cash, next_log_income)
            child_c > 0.0 || throw(ErrorException(
                "the child G2EGM policy has nonpositive consumption"))
            expected_marginal += rule.weights[q] * gross *
                _g2egm_marginal_utility(config.gamma, child_c)
        end
    end
    return expected_value, expected_marginal, signature
end

function _postdecision_tables(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid, child_value::Matrix{Float64},
    child_policy::Matrix{Float64}, diagnostics::G2EGMSolveDiagnostics)

    shape = (length(grid.asset), length(grid.post_log_income))
    continuation = Matrix{Float64}(undef, shape)
    marginal = Matrix{Float64}(undef, shape)
    mapped_cash = Matrix{Float64}(undef, shape)
    node_consumption = Matrix{Float64}(undef, shape)
    regimes = Matrix{UInt8}(undef, shape)
    valid = trues(shape)

    @inbounds for j in eachindex(grid.post_log_income)
        ell = grid.post_log_income[j]
        for i in eachindex(grid.asset)
            saving = grid.asset[i]
            expected_value, expected_marginal, signature =
                _expected_continuation(config, process, rule, grid,
                    child_value, child_policy, saving, ell, diagnostics;
                    marginal=true)
            continuation[i, j] = expected_value
            marginal[i, j] = expected_marginal
            regimes[i, j] = signature
            c = _g2egm_inverse_marginal(
                config.gamma, config.beta * expected_marginal)
            x = saving + c
            node_consumption[i, j] = c
            mapped_cash[i, j] = x
            valid[i, j] = isfinite(expected_value) && isfinite(c) && c > 0.0 &&
                          isfinite(x)
        end
    end
    mapped_log_income = repeat(
        reshape(grid.post_log_income, 1, :), length(grid.asset), 1)
    return (; continuation, marginal, mapped_cash, mapped_log_income,
            node_consumption, regimes, valid)
end

function _envelope_options(options::AR1G2EGMOptions)
    return EnvelopeOptions(
        index_padding=options.bounding_box_padding,
        minimum_weight=options.minimum_barycentric_weight,
        degenerate_atol=options.determinant_tolerance,
        degenerate_rtol=64eps(Float64),
        value_atol=options.value_atol,
        value_rtol=options.value_rtol,
        hole_radius=options.hole_search_radius,
        fill_holes=true,
    )
end

function _prefer_g2egm_candidate(candidate_value::Float64,
    candidate_consumption::Float64, incumbent_value::Float64,
    incumbent_consumption::Float64, options::AR1G2EGMOptions)
    # Never let an infinite endpoint masquerade as a numerical tie: with an
    # infinite scale the usual `abs(gap) <= tolerance` test becomes
    # `Inf <= Inf`. Bellman candidates admitted to the envelope must be finite.
    isfinite(candidate_value) || return false
    isfinite(incumbent_value) || return true
    tolerance = options.value_atol + options.value_rtol *
        max(abs(candidate_value), abs(incumbent_value), 1.0)
    if candidate_value > incumbent_value + tolerance
        return true
    elseif abs(candidate_value - incumbent_value) <= tolerance
        return candidate_consumption < incumbent_consumption
    end
    return false
end

function _merge_envelope_audit!(diagnostics::G2EGMSolveDiagnostics,
    audit::EnvelopeAudit)
    diagnostics.triangles_considered += audit.triangles_total
    diagnostics.triangles_accepted += audit.triangles_processed
    diagnostics.triangles_degenerate += audit.triangles_degenerate
    diagnostics.triangles_rejected_regime += audit.triangles_regime_rejected
    diagnostics.triangle_candidates += audit.finite_candidates
    diagnostics.triangle_extrapolated_candidates +=
        audit.negative_barycentric_candidates
    diagnostics.interior_holes_before_fill += audit.direct_holes_before_fill
    diagnostics.interior_holes_filled += audit.holes_filled
    return nothing
end

function _count_mixed_regime_triangles(regimes::Matrix{UInt8})
    mixed = 0
    for j in 1:size(regimes, 2)-1, i in 1:size(regimes, 1)-1
        for triangle in tensor_cell_triangles(i, j)
            signature = regimes[triangle[1]]
            if regimes[triangle[2]] != signature ||
               regimes[triangle[3]] != signature
                mixed += 1
            end
        end
    end
    return mixed
end

function _direct_objective(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid, child_value::Matrix{Float64},
    child_policy::Matrix{Float64}, cash::Float64, ell::Float64,
    saving::Float64, diagnostics::G2EGMSolveDiagnostics)
    diagnostics.direct_objective_evaluations += 1
    0.0 <= saving <= min(cash, last(grid.asset)) || return -Inf
    consumption = cash - saving
    consumption > 0.0 || return -Inf
    continuation_value, _, _ = _expected_continuation(
        config, process, rule, grid, child_value, child_policy,
        saving, ell, diagnostics; marginal=false)
    return _g2egm_utility(config.gamma, consumption) +
           config.beta * continuation_value
end

function _golden_direct_maximum(objective, lower::Float64, upper::Float64,
    iterations::Int)
    upper > lower || return (saving=lower, value=objective(lower))
    ratio = (sqrt(5.0) - 1.0) / 2.0
    left = upper - ratio * (upper - lower)
    right = lower + ratio * (upper - lower)
    left_value, right_value = objective(left), objective(right)
    for _ in 1:iterations
        if left_value < right_value
            lower = left
            left = right
            left_value = right_value
            right = lower + ratio * (upper - lower)
            right_value = objective(right)
        else
            upper = right
            right = left
            right_value = left_value
            left = upper - ratio * (upper - lower)
            left_value = objective(left)
        end
    end
    candidates = (lower, left, right, upper)
    values = objective.(candidates)
    best = maximum(values)
    tied = findall(value -> value == best, values)
    index = tied[argmax(getindex.(Ref(candidates), tied))]
    return (saving=candidates[index], value=values[index])
end

"""A quadrature branch with fixed next income and log-grid interpolation cell."""
struct DirectContinuationBranch
    log_income::Float64
    income::Float64
    log_index::Int
    probability::Float64
end

"""One interpolation cell and its analytically maximizing saving candidate."""
struct DirectCellCandidate
    lower::Float64
    upper::Float64
    saving::Float64
    value::Float64
end

"""Feasible action and fresh-quadrature value returned by direct Bellman repair."""
struct DirectActionResult
    consumption::Float64
    saving::Float64
    value::Float64
end

function _direct_branches(process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid, log_income::Float64)
    mean = process.unconditional_log_mean
    return [begin
        next_log = mean + process.rho * (log_income - mean) +
                   process.log_std * only(rule.nodes[q])
        index = _search_interval(grid.log_income, next_log)
        1 <= index < length(grid.log_income) || throw(DomainError(next_log,
            "a direct-repair log-income query left the common G2EGM grid"))
        DirectContinuationBranch(next_log, exp(next_log), index, rule.weights[q])
    end for q in eachindex(rule.weights)]
end

"""Split saving at every Q5 preimage of a child cash knot, including the floor."""
function _direct_cell_boundaries(config::StochasticPTDConfig,
    grid::AR1G2EGMGrid, branches::Vector{DirectContinuationBranch}, upper::Float64)
    gross = 1.0 + config.return_rate
    boundaries = Float64[0.0, upper]
    for branch in branches, next_cash in grid.cash
        saving = (next_cash - branch.income) / gross
        isfinite(saving) && 0.0 < saving < upper && push!(boundaries, saving)
    end
    return sort!(unique!(boundaries))
end

"""
Slope of the expected child-value interpolant on a single open saving cell.

For fixed next log income, bilinear interpolation is affine in cash. Compute
its slope from child-node differences, rather than subtracting two nearly equal
expected values. `T=BigFloat` also recomputes these differences and weights at
high precision when Float64 candidate values cannot be ranked reliably.
"""
function _direct_cell_slope(::Type{T}, config::StochasticPTDConfig,
    grid::AR1G2EGMGrid, child_value::Matrix{Float64},
    branches::Vector{DirectContinuationBranch}, saving) where {T<:AbstractFloat}
    gross = T(1.0 + config.return_rate)
    slope = zero(T)
    for branch in branches
        next_cash = gross * T(saving) + T(branch.income)
        next_cash > T(config.cash_floor) || continue
        index = searchsortedlast(grid.cash, next_cash)
        index = min(index, length(grid.cash) - 1)
        1 <= index < length(grid.cash) || throw(DomainError(next_cash,
            "a direct-repair slope query left the common cash grid"))
        j = branch.log_index
        weight = (T(branch.log_income) - T(grid.log_income[j])) /
                 (T(grid.log_income[j + 1]) - T(grid.log_income[j]))
        left_slope = T(child_value[index + 1, j]) - T(child_value[index, j])
        right_slope = T(child_value[index + 1, j + 1]) - T(child_value[index, j + 1])
        slope += T(branch.probability) * gross *
                 ((one(T) - weight) * left_slope + weight * right_slope) /
                 (T(grid.cash[index + 1]) - T(grid.cash[index]))
    end
    return slope
end

"""Analytic maximizer of u(cash-a)+beta*(intercept+slope*a) on one cell."""
function _direct_cell_saving(::Type{T}, cash::Float64, beta::Float64,
    gamma::Float64, lower::Float64, upper::Float64, slope::T) where {T<:AbstractFloat}
    slope > zero(T) || return T(lower)
    consumption = (T(beta) * slope)^(-one(T) / T(gamma))
    return clamp(T(cash) - consumption, T(lower), T(upper))
end

"""Evaluate the same stored interpolant with 256-bit arithmetic for close rankings."""
function _precise_direct_value(config::StochasticPTDConfig, grid::AR1G2EGMGrid,
    child_value::Matrix{Float64}, branches::Vector{DirectContinuationBranch},
    cash::Float64, saving::BigFloat)
    consumption = BigFloat(cash) - saving
    consumption > 0 || return BigFloat(-Inf)
    gamma = BigFloat(config.gamma)
    utility = config.gamma == 1.0 ? log(consumption) :
              (consumption^(1 - gamma) - 1) / (1 - gamma)
    continuation = BigFloat(0)
    gross = BigFloat(1.0 + config.return_rate)
    for branch in branches
        next_cash = max(gross * saving + BigFloat(branch.income),
            BigFloat(config.cash_floor))
        index = min(searchsortedlast(grid.cash, next_cash), length(grid.cash) - 1)
        1 <= index < length(grid.cash) || throw(DomainError(next_cash,
            "a precise direct-repair query left the common cash grid"))
        j = branch.log_index
        wx = (next_cash - BigFloat(grid.cash[index])) /
             (BigFloat(grid.cash[index + 1]) - BigFloat(grid.cash[index]))
        wy = (BigFloat(branch.log_income) - BigFloat(grid.log_income[j])) /
             (BigFloat(grid.log_income[j + 1]) - BigFloat(grid.log_income[j]))
        left = (1 - wy) * BigFloat(child_value[index, j]) +
               wy * BigFloat(child_value[index, j + 1])
        right = (1 - wy) * BigFloat(child_value[index + 1, j]) +
                wy * BigFloat(child_value[index + 1, j + 1])
        continuation += BigFloat(branch.probability) * ((1 - wx) * left + wx * right)
    end
    return utility + BigFloat(config.beta) * continuation
end

"""
    _direct_global_action(config, process, rule, grid, options, child_value,
                          child_policy, cash, ell, diagnostics)

Globally maximize the finite-Q5 Bellman objective on the stored child-value
interpolant, subject to `0 <= saving <= min(cash, last(grid.asset))`.
Every preimage of a child cash knot is included. Between consecutive preimages
the continuation value is affine, so CRRA gives its exact interior candidate;
clamping that candidate also considers the cell's constraint endpoints.

The welfare tolerance used to screen G2EGM triangles is deliberately not used
here: a utility near-tie can hide a large saving error at high wealth. First
anchor comparisons to the largest computed value. Re-optimize and compare all
rounding-ambiguous cells at 256-bit precision, selecting greater saving only at
a high-precision rounding tie. Julia 1.12's do-block precision/rounding scopes
are task-local, so concurrent repairs do not change another task's arithmetic.
"""
function _direct_global_action(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid, options::AR1G2EGMOptions,
    child_value::Matrix{Float64}, child_policy::Matrix{Float64},
    cash::Float64, ell::Float64, diagnostics::G2EGMSolveDiagnostics)

    isfinite(cash) && cash > 0.0 || throw(ArgumentError(
        "direct Bellman repair requires finite positive cash"))
    upper = min(cash, last(grid.asset))
    branches = _direct_branches(process, rule, grid, ell)
    boundaries = _direct_cell_boundaries(config, grid, branches, upper)
    objective = saving -> _direct_objective(config, process, rule, grid,
        child_value, child_policy, cash, ell, saving, diagnostics)
    candidates = DirectCellCandidate[]
    for index in 1:length(boundaries)-1
        lower, cell_upper = boundaries[index], boundaries[index + 1]
        midpoint = lower + (cell_upper - lower) / 2.0
        slope = _direct_cell_slope(Float64, config, grid, child_value, branches, midpoint)
        saving = _direct_cell_saving(Float64, cash, config.beta, config.gamma,
            lower, cell_upper, slope)
        # Zero consumption is never a finite maximizer. Keep its closest
        # feasible floating-point action if an analytic root rounds to cash.
        saving == cash && (saving = prevfloat(cash))
        push!(candidates, DirectCellCandidate(lower, cell_upper, saving, objective(saving)))
    end
    best_index = argmax(getfield.(candidates, :value))
    best_value = candidates[best_index].value
    isfinite(best_value) || throw(ErrorException(
        "direct Bellman repair found no finite candidate at cash=$cash, log_income=$ell"))
    rounding_band = 64eps(Float64) * max(abs(best_value), 1.0)
    ambiguous = findall(candidate -> candidate.value >= best_value - rounding_band, candidates)
    best_saving = candidates[best_index].saving
    if length(ambiguous) > 1
        best_saving = setprecision(BigFloat, 256) do
            setrounding(BigFloat, RoundNearest) do
                savings = BigFloat[]
                values = BigFloat[]
                for index in ambiguous
                    candidate = candidates[index]
                    midpoint = (BigFloat(candidate.lower) + BigFloat(candidate.upper)) / 2
                    slope = _direct_cell_slope(BigFloat, config, grid,
                        child_value, branches, midpoint)
                    saving = _direct_cell_saving(BigFloat, cash, config.beta, config.gamma,
                        candidate.lower, candidate.upper, slope)
                    push!(savings, saving)
                    push!(values, _precise_direct_value(config, grid, child_value,
                        branches, cash, saving))
                end
                maximum_value = maximum(values)
                tolerance = 64eps(BigFloat) * max(abs(maximum_value), BigFloat(1))
                # Anchor once to the maximum: pairwise approximate comparisons
                # are nontransitive and can walk through successively worse actions.
                tied = findall(value -> value >= maximum_value - tolerance, values)
                Float64(maximum(savings[tied]))
            end
        end
        best_saving = clamp(best_saving, 0.0, upper)
        best_saving == cash && (best_saving = prevfloat(cash))
    end
    return DirectActionResult(cash - best_saving, best_saving, objective(best_saving))
end

function _saving_violation(first_saving::Float64,
    second_saving::Float64, cash::Float64, options::AR1G2EGMOptions)
    tolerance = options.monotonicity_atol + options.monotonicity_rtol *
        max(abs(first_saving), abs(second_saving), cash, 1.0)
    return second_saving < first_saving - tolerance
end

function _repair_monotone_saving!(policy::Matrix{Float64},
    value::Matrix{Float64}, winner_source::Matrix{UInt8},
    config::StochasticPTDConfig, process::LogAR1Process,
    rule::GaussHermiteRule{1,5}, grid::AR1G2EGMGrid,
    options::AR1G2EGMOptions, child_value::Matrix{Float64},
    child_policy::Matrix{Float64}, diagnostics::G2EGMSolveDiagnostics)

    # Common-grid nodes above a_max are continuation padding.  At sufficiently
    # high cash the finite asset grid necessarily forces saving back to a_max,
    # so they cannot be part of a global monotonicity assertion.  Audit and
    # repair only through the largest common-cash node at or below a_max.
    last_policy_index = searchsortedlast(grid.cash, last(grid.asset))
    last_policy_index >= 2 || throw(ArgumentError(
        "the G2EGM policy domain needs at least two common-cash nodes"))
    for j in eachindex(grid.log_income)
        ell = grid.log_income[j]
        # A newly repaired node can expose an adjacent reconstruction error.
        # Each direct solve depends only on the fixed child arrays, so repeating
        # it cannot change the answer. Solve every targeted node at most once;
        # a remaining violation between two repaired nodes is a hard diagnostic.
        directly_solved = falses(last_policy_index)
        for _ in 1:last_policy_index
            targets = Int[]
            for i in 1:last_policy_index-1
                first_saving = grid.cash[i] - policy[i, j]
                second_saving = grid.cash[i + 1] - policy[i + 1, j]
                if _saving_violation(
                    first_saving, second_saving, grid.cash[i + 1], options)
                    diagnostics.monotonicity_violations += 1
                    if options.enforce_monotone_saving &&
                       directly_solved[i] && directly_solved[i + 1]
                        tolerance = options.monotonicity_atol + options.monotonicity_rtol *
                            max(abs(first_saving), abs(second_saving), grid.cash[i + 1], 1.0)
                        diagnostics.monotonicity_violations_remaining += 1
                        throw(ErrorException(
                            "G2EGM monotone saving failed after exact interpolation-cell repair: " *
                            "log_income=$ell, cash=($(grid.cash[i]), $(grid.cash[i + 1])), " *
                            "saving=($first_saving, $second_saving), tolerance=$tolerance. " *
                            "Both states were already solved directly; repeating them cannot help."))
                    end
                    directly_solved[i] || push!(targets, i)
                    directly_solved[i + 1] || push!(targets, i + 1)
                end
            end
            isempty(targets) && break
            options.enforce_monotone_saving || break
            sort!(unique!(targets))
            for i in targets
                direct = _direct_global_action(
                    config, process, rule, grid, options,
                    child_value, child_policy, grid.cash[i], ell, diagnostics)
                policy[i, j] = direct.consumption
                value[i, j] = direct.value
                winner_source[i, j] = UInt8(3)
                directly_solved[i] = true
                diagnostics.monotonicity_repairs += 1
            end
        end
        for i in 1:last_policy_index-1
            first_saving = grid.cash[i] - policy[i, j]
            second_saving = grid.cash[i + 1] - policy[i + 1, j]
            _saving_violation(
                first_saving, second_saving, grid.cash[i + 1], options) &&
                (diagnostics.monotonicity_violations_remaining += 1)
        end
    end
    !options.enforce_monotone_saving ||
        diagnostics.monotonicity_violations_remaining == 0 ||
        throw(ErrorException(
            "the G2EGM common-grid policy violates monotone saving after direct repair"))
    return nothing
end

function _floor_kink_savings(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    log_income::Float64, asset_maximum::Float64)
    gross = 1.0 + config.return_rate
    mean = process.unconditional_log_mean
    savings = Float64[]
    for node in rule.nodes
        next_log = mean + process.rho * (log_income - mean) +
                   process.log_std * only(node)
        saving = (config.cash_floor - exp(next_log)) / gross
        0.0 < saving < asset_maximum && push!(savings, saving)
    end
    sort!(unique!(savings))
    return savings
end

function _solve_period(config::StochasticPTDConfig,
    process::LogAR1Process, rule::GaussHermiteRule{1,5},
    grid::AR1G2EGMGrid, options::AR1G2EGMOptions,
    child_value::Matrix{Float64}, child_policy::Matrix{Float64},
    diagnostics::G2EGMSolveDiagnostics)

    post = _postdecision_tables(config, process, rule, grid,
        child_value, child_policy, diagnostics)
    diagnostics.triangles_mixed_floor_regime +=
        _count_mixed_regime_triangles(post.regimes)
    continuation = post.continuation
    asset_maximum = last(grid.asset)
    feasible = function (cash::Float64, ::Float64, c::Float64)
        saving = cash - c
        tolerance = 64eps(max(cash, asset_maximum, 1.0))
        return isfinite(c) && c > 0.0 &&
               saving > tolerance && saving <= asset_maximum + tolerance
    end
    value_at_candidate = function (cash::Float64, ell::Float64, c::Float64)
        saving = clamp(cash - c, 0.0, asset_maximum)
        continuation_value = _bilinear(grid.asset,
            grid.post_log_income, continuation, saving, ell)
        return _g2egm_utility(config.gamma, c) +
               config.beta * continuation_value
    end
    regimes = options.reject_mixed_floor_regimes ? post.regimes : nothing
    interior = project_local_envelope(
        post.mapped_cash, post.mapped_log_income, post.node_consumption,
        grid.cash, grid.log_income;
        feasible, value_at=value_at_candidate,
        valid_vertices=post.valid, regimes,
        options=_envelope_options(options))
    _merge_envelope_audit!(diagnostics, interior.audit)

    common_shape = (length(grid.cash), length(grid.log_income))
    policy = Matrix{Float64}(undef, common_shape)
    value = Matrix{Float64}(undef, common_shape)
    # Source 0 covers either explicit asset constraint.  The final saving-based
    # audit below distinguishes upper-bound winners from the borrowing corner.
    winner_source = zeros(UInt8, common_shape) # 0=constraint, 1=triangle, 2=kink

    # Current borrowing constraint a=0. The local G2EGM projection above uses
    # the postdecision interpolant to screen triangle candidates. The final
    # triangle winner at every common node, this constraint, every exact kink,
    # and every direct repair are all revalued with the same fresh Q5 operator
    # on the child common grid.
    @inbounds for j in eachindex(grid.log_income)
        ell = grid.log_income[j]
        continuation_zero, _, _ = _expected_continuation(
            config, process, rule, grid, child_value, child_policy,
            0.0, ell, diagnostics; marginal=false)
        diagnostics.final_q5_candidate_revaluations += 1
        for i in eachindex(grid.cash)
            cash = grid.cash[i]
            policy[i, j] = cash
            value[i, j] = _g2egm_utility(config.gamma, cash) +
                          config.beta * continuation_zero
        end
    end

    # First envelope: the reconstructed interior segment against a=0.
    @inbounds for index in CartesianIndices(policy)
        interior.coverage[index] || continue
        cash = grid.cash[index[1]]
        ell = grid.log_income[index[2]]
        candidate_c = interior.consumption[index]
        saving = clamp(cash - candidate_c, 0.0, asset_maximum)
        continuation_value, _, _ = _expected_continuation(
            config, process, rule, grid, child_value, child_policy,
            saving, ell, diagnostics; marginal=false)
        diagnostics.final_q5_candidate_revaluations += 1
        candidate_v = _g2egm_utility(config.gamma, candidate_c) +
                      config.beta * continuation_value
        if _prefer_g2egm_candidate(candidate_v, candidate_c,
            value[index], policy[index], options)
            policy[index] = candidate_c
            value[index] = candidate_v
            winner_source[index] = UInt8(1)
        end
    end

    # Upper numerical asset constraint a=a_max.  Common-grid cash above a_max is
    # continuation padding required by x'=R*a+y'.  The endogenous triangle
    # surface need not cover that padding, so omitting this explicit candidate
    # can incorrectly leave consume-all in control even when a_max is the
    # constrained Bellman optimum.  As for the lower constraint and the final
    # triangle winner, use the child common grid and a fresh Q5 evaluation.
    @inbounds for j in eachindex(grid.log_income)
        ell = grid.log_income[j]
        continuation_ceiling, _, _ = _expected_continuation(
            config, process, rule, grid, child_value, child_policy,
            asset_maximum, ell, diagnostics; marginal=false)
        diagnostics.final_q5_candidate_revaluations += 1
        for i in eachindex(grid.cash)
            cash = grid.cash[i]
            cash > asset_maximum || continue
            c = cash - asset_maximum
            candidate_v = _g2egm_utility(config.gamma, c) +
                          config.beta * continuation_ceiling
            diagnostics.asset_ceiling_candidates += 1
            if _prefer_g2egm_candidate(candidate_v, c,
                value[i, j], policy[i, j], options)
                policy[i, j] = c
                value[i, j] = candidate_v
                winner_source[i, j] = UInt8(0)
            end
        end
    end

    # Exact finite-Q5 floor-kink saving candidates. These are the known curves
    # on which W is continuous but W_a changes regime.
    @inbounds for j in eachindex(grid.log_income)
        ell = grid.log_income[j]
        kink_data = Tuple{Float64,Float64}[]
        for saving in _floor_kink_savings(
            config, process, rule, ell, asset_maximum)
            continuation_value, _, _ = _expected_continuation(
                config, process, rule, grid, child_value, child_policy,
                saving, ell, diagnostics; marginal=false)
            diagnostics.final_q5_candidate_revaluations += 1
            push!(kink_data, (saving, continuation_value))
        end
        for i in eachindex(grid.cash)
            cash = grid.cash[i]
            for (saving, continuation_value) in kink_data
                saving < cash || continue
                c = cash - saving
                candidate_v = _g2egm_utility(config.gamma, c) +
                              config.beta * continuation_value
                diagnostics.floor_kink_candidates += 1
                if _prefer_g2egm_candidate(candidate_v, c,
                    value[i, j], policy[i, j], options)
                    policy[i, j] = c
                    value[i, j] = candidate_v
                    winner_source[i, j] = UInt8(2)
                end
            end
        end
    end

    _repair_monotone_saving!(policy, value, winner_source,
        config, process, rule, grid, options, child_value, child_policy,
        diagnostics)

    if !all(isfinite, policy) || !all(isfinite, value)
        bad = findfirst(index ->
            !isfinite(policy[index]) || !isfinite(value[index]),
            CartesianIndices(policy))
        throw(ErrorException(
            "the G2EGM period solve left a nonfinite common-grid node " *
            "at $bad (cash=$(grid.cash[bad[1]]), " *
            "log_income=$(grid.log_income[bad[2]]), " *
            "consumption=$(policy[bad]), value=$(value[bad]), " *
            "source=$(winner_source[bad]))"))
    end
    @inbounds for index in CartesianIndices(policy)
        cash = grid.cash[index[1]]
        0.0 < policy[index] <= cash || throw(ErrorException(
            "the G2EGM upper envelope selected an infeasible consumption"))
        saving = cash - policy[index]
        tolerance = 64eps(max(cash, asset_maximum, 1.0))
        if saving >= asset_maximum - tolerance
            diagnostics.asset_ceiling_winners += 1
            if cash <= asset_maximum
                diagnostics.policy_domain_asset_ceiling_winners += 1
            else
                diagnostics.continuation_padding_asset_ceiling_winners += 1
            end
        end
        source = winner_source[index]
        if source == 0
            diagnostics.constraint_winners += 1
        elseif source == 1
            diagnostics.triangle_winners += 1
            winning_weight =
                interior.winner_minimum_barycentric_weight[index]
            diagnostics.minimum_winning_barycentric_weight = min(
                diagnostics.minimum_winning_barycentric_weight,
                winning_weight)
            winning_weight < 0.0 &&
                (diagnostics.triangle_extrapolated_winners += 1)
        else
            if source == 2
                diagnostics.floor_kink_winners += 1
            else
                diagnostics.direct_repair_winners += 1
            end
        end
    end
    return policy, value
end

"""
    build_ar1_solution(config, process; base_rule, grid, options)

Solve the finite-horizon scalar AR(1) model using G2EGM local triangles and
a common grid.
"""
function build_ar1_solution(config::StochasticPTDConfig,
    process::LogAR1Process; base_rule=gauss_hermite_rule(process, 5),
    grid::Union{Nothing,AR1G2EGMGrid}=nothing,
    options::AR1G2EGMOptions=AR1G2EGMOptions())

    rule = base_rule
    rule isa GaussHermiteRule{1,5} || throw(ArgumentError(
        "the AR(1) G2EGM comparator requires exactly five positive Gauss-Hermite nodes"))
    canonical = gauss_hermite_rule(process, 5)
    _rules_equal(rule, canonical) || throw(ArgumentError(
        "the supplied G2EGM rule is not the canonical Q5 rule for the AR(1) process"))
    _validate_options(options)
    resolved_grid = isnothing(grid) ? ar1_g2egm_grid(
        cash_floor=config.cash_floor,
        gross_return=1.0 + config.return_rate) : grid
    _validate_grid_for_model(config, process, rule, resolved_grid)

    horizon = config.terminal.final_date
    policy = Vector{Matrix{Float64}}(undef, horizon)
    value = Vector{Matrix{Float64}}(undef, horizon)
    terminal_policy = repeat(reshape(resolved_grid.cash, :, 1),
        1, length(resolved_grid.log_income))
    terminal_value = similar(terminal_policy)
    @inbounds for index in CartesianIndices(terminal_value)
        terminal_value[index] = _g2egm_utility(
            config.gamma, terminal_policy[index])
    end
    policy[horizon] = terminal_policy
    value[horizon] = terminal_value

    solve_diagnostics = G2EGMSolveDiagnostics()
    for date in horizon-1:-1:1
        policy[date], value[date] = _solve_period(
            config, process, rule, resolved_grid, options,
            value[date + 1], policy[date + 1], solve_diagnostics)
    end
    input_fingerprint = _solution_input_fingerprint(
        config, process, rule, resolved_grid, options)
    fingerprint = _complete_solution_fingerprint(
        input_fingerprint, policy, value)
    return AR1G2EGMSolution(config, process, rule, resolved_grid, options,
        policy, value, solve_diagnostics, G2EGMQueryDiagnostics(), fingerprint)
end

"""Validate a G2EGM solution against its declared model, process, and grids."""
function validate_alignment(solution::AR1G2EGMSolution,
    config::StochasticPTDConfig, process::LogAR1Process,
    base_rule::GaussHermiteRule{1,5}, grid::AR1G2EGMGrid=solution.grid;
    options::AR1G2EGMOptions=solution.options)
    solution.config == config || throw(ArgumentError(
        "the G2EGM solution configuration does not match the contest"))
    solution.process == process || throw(ArgumentError(
        "the G2EGM AR(1) process does not match the contest"))
    _rules_equal(solution.base_rule, base_rule) || throw(ArgumentError(
        "the G2EGM quadrature does not match the contest Q5 rule"))
    solution.grid.asset == grid.asset &&
        solution.grid.post_log_income == grid.post_log_income &&
        solution.grid.cash == grid.cash &&
        solution.grid.log_income == grid.log_income || throw(ArgumentError(
            "the G2EGM solution grids do not match the declared contest grids"))
    solution.options == options || throw(ArgumentError(
        "the G2EGM solution options do not match the declared contest options"))
    input_fingerprint = _solution_input_fingerprint(
        config, process, base_rule, grid, options)
    expected = _complete_solution_fingerprint(
        input_fingerprint, solution.policy, solution.value)
    solution.fingerprint == expected || throw(ArgumentError(
        "the G2EGM solution fingerprint does not match its scientific inputs"))
    occursin(r"^[0-9a-f]{64}$", solution.fingerprint) || throw(ErrorException(
        "the G2EGM solution fingerprint is malformed"))
    return nothing
end
