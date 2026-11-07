function _validate_hsz_options(options::HSZG2EGMOptions)
    options.bounding_box_padding >= 0 || throw(ArgumentError(
        "HSZ bounding-box padding must be nonnegative"))
    -1.0 <= options.minimum_barycentric_weight <= 0.0 ||
        throw(ArgumentError("HSZ minimum barycentric weight must lie in [-1,0]"))
    options.direct_iterations >= 16 || throw(ArgumentError(
        "HSZ direct_iterations must be at least 16"))
    return options
end

function _hsz_rules_equal(left::GaussHermiteRule{2,25},
    right::GaussHermiteRule{2,25})
    return left.nodes == right.nodes && left.weights == right.weights &&
           left.nodes_per_dimension == right.nodes_per_dimension
end

function _validate_hsz_grid(config::StochasticPTDConfig,
    process::HSZProcess, rule::GaussHermiteRule{2,25}, grid::HSZG2EGMGrid)
    first(grid.cash) == config.cash_floor || throw(ArgumentError(
        "the HSZ common cash grid must begin at the cash floor"))
    required_cash = (1.0 + config.return_rate) * last(grid.asset) +
        exp(last(grid.post_log_earnings)) - exp(first(grid.post_log_medical))
    last(grid.cash) >= required_cash - 128eps(max(required_cash, 1.0)) ||
        throw(ArgumentError("the HSZ common cash grid omits transition padding"))
    for earnings in extrema(grid.post_log_earnings),
        medical in extrema(grid.post_log_medical), node in rule.nodes
        next_e = process.unconditional_log_earnings +
            process.rho_earnings *
            (earnings - process.unconditional_log_earnings) +
            process.std_earnings * node[1]
        next_m = process.unconditional_log_medical +
            process.rho_medical *
            (medical - process.unconditional_log_medical) +
            process.std_medical * node[2]
        first(grid.log_earnings) <= next_e <= last(grid.log_earnings) ||
            throw(ArgumentError("Q25 earnings transition leaves the HSZ grid"))
        first(grid.log_medical) <= next_m <= last(grid.log_medical) ||
            throw(ArgumentError("Q25 medical transition leaves the HSZ grid"))
    end
    return grid
end

function _record_hsz_solve_query!(diagnostics::HSZG2EGMSolveDiagnostics,
    grid::HSZG2EGMGrid, cash::Float64, earnings::Float64, medical::Float64)
    first(grid.cash) <= cash <= last(grid.cash) || throw(DomainError(cash,
        "HSZ continuation cash left the common grid"))
    first(grid.log_earnings) <= earnings <= last(grid.log_earnings) ||
        throw(DomainError(earnings, "HSZ earnings left the common grid"))
    first(grid.log_medical) <= medical <= last(grid.log_medical) ||
        throw(DomainError(medical, "HSZ medical state left the common grid"))
    diagnostics.continuation_queries += 1
    diagnostics.minimum_cash_margin = min(diagnostics.minimum_cash_margin,
        cash - first(grid.cash), last(grid.cash) - cash)
    diagnostics.minimum_earnings_margin = min(
        diagnostics.minimum_earnings_margin,
        earnings - first(grid.log_earnings), last(grid.log_earnings) - earnings)
    diagnostics.minimum_medical_margin = min(
        diagnostics.minimum_medical_margin,
        medical - first(grid.log_medical), last(grid.log_medical) - medical)
    return nothing
end

function _hsz_expected_continuation(config::StochasticPTDConfig,
    process::HSZProcess, rule::GaussHermiteRule{2,25}, grid::HSZG2EGMGrid,
    child_value::Array{Float64,3}, child_policy::Array{Float64,3},
    saving::Float64, earnings::Float64, medical::Float64,
    diagnostics::HSZG2EGMSolveDiagnostics; marginal::Bool)
    gross = 1.0 + config.return_rate
    expected_value, expected_marginal = 0.0, 0.0
    signature = UInt32(0)
    diagnostics.quadrature_expectations += 1
    @inbounds for q in eachindex(rule.weights)
        node = rule.nodes[q]
        next_e = process.unconditional_log_earnings +
            process.rho_earnings *
            (earnings - process.unconditional_log_earnings) +
            process.std_earnings * node[1]
        next_m = process.unconditional_log_medical +
            process.rho_medical *
            (medical - process.unconditional_log_medical) +
            process.std_medical * node[2]
        net_income = exp(next_e) - exp(next_m)
        pretransfer = gross * saving + net_income
        nonbinding = pretransfer > config.cash_floor
        nonbinding && (signature |= UInt32(1) << (q - 1))
        next_cash = max(pretransfer, config.cash_floor)
        _record_hsz_solve_query!(
            diagnostics, grid, next_cash, next_e, next_m)
        child_v = _trilinear(grid.cash, grid.log_earnings,
            grid.log_medical, child_value, next_cash, next_e, next_m)
        expected_value += rule.weights[q] * child_v
        if marginal && nonbinding
            child_c = _trilinear(grid.cash, grid.log_earnings,
                grid.log_medical, child_policy, next_cash, next_e, next_m)
            child_c > 0.0 || throw(ErrorException(
                "the HSZ child policy has nonpositive consumption"))
            expected_marginal += rule.weights[q] * gross *
                _g2egm_marginal_utility(config.gamma, child_c)
        end
    end
    return expected_value, expected_marginal, signature
end

function _hsz_postdecision_tables(config, process, rule, grid,
    child_value, child_policy, diagnostics)
    shape = (length(grid.asset), length(grid.post_log_earnings),
             length(grid.post_log_medical))
    continuation = Array{Float64,3}(undef, shape)
    mapped_cash = Array{Float64,3}(undef, shape)
    mapped_e = Array{Float64,3}(undef, shape)
    mapped_m = Array{Float64,3}(undef, shape)
    node_consumption = Array{Float64,3}(undef, shape)
    regimes = Array{UInt32,3}(undef, shape)
    valid = trues(shape)
    @inbounds for k in eachindex(grid.post_log_medical),
                  j in eachindex(grid.post_log_earnings),
                  i in eachindex(grid.asset)
        saving = grid.asset[i]
        earnings, medical = grid.post_log_earnings[j], grid.post_log_medical[k]
        expected_value, expected_marginal, signature =
            _hsz_expected_continuation(config, process, rule, grid,
                child_value, child_policy, saving, earnings, medical,
                diagnostics; marginal=true)
        c = _g2egm_inverse_marginal(
            config.gamma, config.beta * expected_marginal)
        continuation[i, j, k] = expected_value
        mapped_cash[i, j, k] = saving + c
        mapped_e[i, j, k] = earnings
        mapped_m[i, j, k] = medical
        node_consumption[i, j, k] = c
        regimes[i, j, k] = signature
        valid[i, j, k] = isfinite(expected_value) && isfinite(c) && c > 0.0
    end
    return (; continuation, mapped_cash, mapped_e, mapped_m,
            node_consumption, regimes, valid)
end

function _hsz_floor_kink_savings(config, process, rule,
    earnings::Float64, medical::Float64, asset_maximum::Float64)
    gross = 1.0 + config.return_rate
    savings = Float64[]
    for node in rule.nodes
        next_e = process.unconditional_log_earnings +
            process.rho_earnings *
            (earnings - process.unconditional_log_earnings) +
            process.std_earnings * node[1]
        next_m = process.unconditional_log_medical +
            process.rho_medical *
            (medical - process.unconditional_log_medical) +
            process.std_medical * node[2]
        saving = (config.cash_floor - (exp(next_e) - exp(next_m))) / gross
        0.0 < saving < asset_maximum && push!(savings, saving)
    end
    sort!(unique!(savings))
    return savings
end

function _hsz_prefer(candidate_v, candidate_c, incumbent_v, incumbent_c,
    options::HSZG2EGMOptions)
    isfinite(candidate_v) || return false
    isfinite(incumbent_v) || return true
    tolerance = options.value_atol + options.value_rtol *
        max(abs(candidate_v), abs(incumbent_v), 1.0)
    return candidate_v > incumbent_v + tolerance ||
        (abs(candidate_v - incumbent_v) <= tolerance &&
         candidate_c < incumbent_c)
end

"""One Q25 branch with fixed exogenous interpolation cells and net income."""
struct HSZDirectContinuationBranch
    earnings::Float64
    medical::Float64
    income::Float64
    earnings_index::Int
    medical_index::Int
    probability::Float64
end

function _hsz_direct_branches(process::HSZProcess, rule::GaussHermiteRule{2,25},
    grid::HSZG2EGMGrid, earnings::Float64, medical::Float64)
    return [begin
        node = rule.nodes[q]
        next_e = process.unconditional_log_earnings + process.rho_earnings *
            (earnings - process.unconditional_log_earnings) + process.std_earnings * node[1]
        next_m = process.unconditional_log_medical + process.rho_medical *
            (medical - process.unconditional_log_medical) + process.std_medical * node[2]
        j, k = _search_interval(grid.log_earnings, next_e),
               _search_interval(grid.log_medical, next_m)
        1 <= j < length(grid.log_earnings) || throw(DomainError(next_e,
            "an HSZ direct-repair earnings query left the common grid"))
        1 <= k < length(grid.log_medical) || throw(DomainError(next_m,
            "an HSZ direct-repair medical query left the common grid"))
        HSZDirectContinuationBranch(next_e, next_m, exp(next_e) - exp(next_m),
            j, k, rule.weights[q])
    end for q in eachindex(rule.weights)]
end

"""All saving preimages of child cash knots, including each Q25 floor boundary."""
function _hsz_direct_cell_boundaries(config, grid, branches, upper::Float64)
    gross = 1.0 + config.return_rate
    boundaries = Float64[0.0, upper]
    for branch in branches, next_cash in grid.cash
        saving = (next_cash - branch.income) / gross
        isfinite(saving) && 0.0 < saving < upper && push!(boundaries, saving)
    end
    return sort!(unique!(boundaries))
end

"""Cash slope of the trilinear child interpolant on one open saving cell.

Fixed earnings and medical coordinates leave a bilinear weighted average of
cash-node differences. Form the differences before summing to retain small
slopes even when the continuation level is large.
"""
function _hsz_direct_cell_slope(::Type{T}, config, grid, child_value,
    branches, saving) where {T<:AbstractFloat}
    gross = T(1.0 + config.return_rate)
    slope = zero(T)
    for branch in branches
        next_cash = gross * T(saving) + T(branch.income)
        next_cash > T(config.cash_floor) || continue
        i = min(searchsortedlast(grid.cash, next_cash), length(grid.cash) - 1)
        1 <= i < length(grid.cash) || throw(DomainError(next_cash,
            "an HSZ direct-repair slope query left the common cash grid"))
        j, k = branch.earnings_index, branch.medical_index
        we = (T(branch.earnings) - T(grid.log_earnings[j])) /
             (T(grid.log_earnings[j + 1]) - T(grid.log_earnings[j]))
        wm = (T(branch.medical) - T(grid.log_medical[k])) /
             (T(grid.log_medical[k + 1]) - T(grid.log_medical[k]))
        delta = zero(T)
        for dm in 0:1, de in 0:1
            weight = (de == 0 ? one(T) - we : we) *
                     (dm == 0 ? one(T) - wm : wm)
            delta += weight * (T(child_value[i + 1, j + de, k + dm]) -
                               T(child_value[i, j + de, k + dm]))
        end
        slope += T(branch.probability) * gross * delta /
                 (T(grid.cash[i + 1]) - T(grid.cash[i]))
    end
    return slope
end

"""Evaluate the same stored Q25 interpolant at high precision for close rankings."""
function _precise_hsz_direct_value(config, grid, child_value, branches,
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
        i = min(searchsortedlast(grid.cash, next_cash), length(grid.cash) - 1)
        1 <= i < length(grid.cash) || throw(DomainError(next_cash,
            "a precise HSZ direct-repair query left the common cash grid"))
        j, k = branch.earnings_index, branch.medical_index
        wx = (next_cash - BigFloat(grid.cash[i])) /
             (BigFloat(grid.cash[i + 1]) - BigFloat(grid.cash[i]))
        we = (BigFloat(branch.earnings) - BigFloat(grid.log_earnings[j])) /
             (BigFloat(grid.log_earnings[j + 1]) - BigFloat(grid.log_earnings[j]))
        wm = (BigFloat(branch.medical) - BigFloat(grid.log_medical[k])) /
             (BigFloat(grid.log_medical[k + 1]) - BigFloat(grid.log_medical[k]))
        child = BigFloat(0)
        for dm in 0:1, de in 0:1
            weight = (de == 0 ? 1 - we : we) * (dm == 0 ? 1 - wm : wm)
            child += weight * ((1 - wx) * BigFloat(child_value[i, j + de, k + dm]) +
                wx * BigFloat(child_value[i + 1, j + de, k + dm]))
        end
        continuation += BigFloat(branch.probability) * child
    end
    return utility + BigFloat(config.beta) * continuation
end

"""Globally maximize the stored finite-Q25 Bellman interpolant.

Split at every child cash-knot preimage, maximize each affine-continuation cell
analytically, and compare all candidates. As in the AR(1) repair, triangle/value
screening tolerances must not turn small utility gaps into large saving errors:
rounding-ambiguous cells are re-optimized and ranked with 256-bit arithmetic.
"""
function _hsz_direct_action(config, process, rule, grid, options,
    child_value, child_policy, cash::Float64, earnings::Float64, medical::Float64,
    diagnostics)
    isfinite(cash) && cash > 0.0 || throw(ArgumentError(
        "HSZ direct Bellman repair requires finite positive cash"))
    upper = min(cash, last(grid.asset))
    branches = _hsz_direct_branches(process, rule, grid, earnings, medical)
    boundaries = _hsz_direct_cell_boundaries(config, grid, branches, upper)
    objective = function (saving)
        0.0 <= saving <= upper || return -Inf
        c = cash - saving
        c > 0.0 || return -Inf
        continuation, _, _ = _hsz_expected_continuation(
            config, process, rule, grid, child_value, child_policy,
            saving, earnings, medical, diagnostics; marginal=false)
        return _g2egm_utility(config.gamma, c) + config.beta * continuation
    end
    candidates = DirectCellCandidate[]
    for index in 1:length(boundaries)-1
        lower, cell_upper = boundaries[index], boundaries[index + 1]
        midpoint = lower + (cell_upper - lower) / 2.0
        slope = _hsz_direct_cell_slope(Float64, config, grid, child_value, branches, midpoint)
        saving = _direct_cell_saving(Float64, cash, config.beta, config.gamma,
            lower, cell_upper, slope)
        saving == cash && (saving = prevfloat(cash))
        push!(candidates, DirectCellCandidate(lower, cell_upper, saving, objective(saving)))
    end
    best_index = argmax(getfield.(candidates, :value))
    best_value = candidates[best_index].value
    isfinite(best_value) || throw(ErrorException(
        "HSZ direct repair found no finite candidate at cash=$cash, " *
        "earnings=$earnings, medical=$medical"))
    rounding_band = 256eps(Float64) * max(abs(best_value), 1.0)
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
                    slope = _hsz_direct_cell_slope(BigFloat, config, grid,
                        child_value, branches, midpoint)
                    saving = _direct_cell_saving(BigFloat, cash, config.beta, config.gamma,
                        candidate.lower, candidate.upper, slope)
                    push!(savings, saving)
                    push!(values, _precise_hsz_direct_value(config, grid, child_value,
                        branches, cash, saving))
                end
                maximum_value = maximum(values)
                tolerance = 256eps(BigFloat) * max(abs(maximum_value), BigFloat(1))
                tied = findall(value -> value >= maximum_value - tolerance, values)
                Float64(maximum(savings[tied]))
            end
        end
        best_saving = clamp(best_saving, 0.0, upper)
        best_saving == cash && (best_saving = prevfloat(cash))
    end
    return (consumption=cash - best_saving, saving=best_saving, value=objective(best_saving))
end

function _hsz_saving_violation(first_saving, second_saving, cash, options)
    tolerance = options.monotonicity_atol + options.monotonicity_rtol *
        max(abs(first_saving), abs(second_saving), cash, 1.0)
    return second_saving < first_saving - tolerance
end

function _repair_hsz_monotonicity!(policy, value, config, process, rule,
    grid, options, child_value, child_policy, diagnostics; winner_source=nothing)
    # Nodes above a_max are continuation padding; only the supported policy
    # domain has a saving-monotonicity certificate.
    last_policy = searchsortedlast(grid.cash, last(grid.asset))
    last_policy >= 2 || throw(ArgumentError(
        "the HSZ policy domain needs at least two common-cash nodes"))
    @inbounds for k in eachindex(grid.log_medical), j in eachindex(grid.log_earnings)
        directly_solved = falses(last_policy)
        for _ in 1:last_policy
            targets = Int[]
            for i in 1:last_policy-1
                first_saving = grid.cash[i] - policy[i, j, k]
                second_saving = grid.cash[i + 1] - policy[i + 1, j, k]
                if _hsz_saving_violation(first_saving, second_saving, grid.cash[i + 1], options)
                    diagnostics.monotonicity_violations += 1
                    if options.enforce_monotone_saving && directly_solved[i] && directly_solved[i + 1]
                        diagnostics.monotonicity_violations_remaining += 1
                        throw(ErrorException(
                            "HSZ monotone saving failed after exact interpolation-cell repair: " *
                            "earnings=$(grid.log_earnings[j]), medical=$(grid.log_medical[k]), " *
                            "cash=($(grid.cash[i]), $(grid.cash[i + 1])), " *
                            "saving=($first_saving, $second_saving). " *
                            "Both states were already solved directly; repeating them cannot help."))
                    end
                    directly_solved[i] || push!(targets, i)
                    directly_solved[i + 1] || push!(targets, i + 1)
                end
            end
            isempty(targets) && break
            options.enforce_monotone_saving || break
            for i in sort!(unique!(targets))
                direct = _hsz_direct_action(config, process, rule, grid, options,
                    child_value, child_policy, grid.cash[i],
                    grid.log_earnings[j], grid.log_medical[k], diagnostics)
                policy[i, j, k] = direct.consumption
                value[i, j, k] = direct.value
                isnothing(winner_source) || (winner_source[i, j, k] = UInt8(3))
                directly_solved[i] = true
                diagnostics.monotonicity_repairs += 1
            end
        end
        for i in 1:last_policy-1
            _hsz_saving_violation(grid.cash[i] - policy[i, j, k],
                grid.cash[i + 1] - policy[i + 1, j, k], grid.cash[i + 1], options) &&
                (diagnostics.monotonicity_violations_remaining += 1)
        end
    end
    !options.enforce_monotone_saving || diagnostics.monotonicity_violations_remaining == 0 ||
        throw(ErrorException("the HSZ common-grid policy violates monotone saving after direct repair"))
    return nothing
end

function _solve_hsz_period(config, process, rule, grid, options,
    child_value, child_policy, diagnostics)
    post = _hsz_postdecision_tables(config, process, rule, grid,
        child_value, child_policy, diagnostics)
    asset_maximum = last(grid.asset)
    feasible = (cash, earnings, medical, c) -> begin
        saving = cash - c
        tolerance = 64eps(max(cash, asset_maximum, 1.0))
        isfinite(c) && c > 0.0 && saving > tolerance &&
            saving <= asset_maximum + tolerance
    end
    value_at_candidate = (cash, earnings, medical, c) -> begin
        saving = clamp(cash - c, 0.0, asset_maximum)
        continuation = _trilinear(grid.asset, grid.post_log_earnings,
            grid.post_log_medical, post.continuation,
            saving, earnings, medical)
        _g2egm_utility(config.gamma, c) + config.beta * continuation
    end
    tetra_options = TetraEnvelopeOptions(
        index_padding=options.bounding_box_padding,
        minimum_weight=options.minimum_barycentric_weight,
        determinant_tolerance=options.determinant_tolerance,
        value_atol=options.value_atol, value_rtol=options.value_rtol)
    interior = project_tetra_envelope(
        post.mapped_cash, post.mapped_e, post.mapped_m,
        post.node_consumption, grid.cash, grid.log_earnings,
        grid.log_medical; feasible, value_at=value_at_candidate,
        valid_vertices=post.valid, options=tetra_options)
    diagnostics.tetrahedra_considered +=
        interior.audit.tetrahedra_considered
    diagnostics.tetrahedra_processed += interior.audit.tetrahedra_processed
    diagnostics.tetrahedra_degenerate += interior.audit.tetrahedra_degenerate
    diagnostics.tetrahedron_candidates += interior.audit.finite_candidates

    shape = (length(grid.cash), length(grid.log_earnings),
             length(grid.log_medical))
    policy = Array{Float64,3}(undef, shape)
    value = Array{Float64,3}(undef, shape)
    source = zeros(UInt8, shape) # 0 constraint, 1 tetra, 2 floor kink, 3 repair

    @inbounds for k in eachindex(grid.log_medical),
                  j in eachindex(grid.log_earnings)
        earnings, medical = grid.log_earnings[j], grid.log_medical[k]
        continuation_zero, _, _ = _hsz_expected_continuation(
            config, process, rule, grid, child_value, child_policy,
            0.0, earnings, medical, diagnostics; marginal=false)
        diagnostics.q25_candidate_revaluations += 1
        for i in eachindex(grid.cash)
            cash = grid.cash[i]
            policy[i, j, k] = cash
            value[i, j, k] = _g2egm_utility(config.gamma, cash) +
                config.beta * continuation_zero
        end
    end

    @inbounds for index in CartesianIndices(policy)
        interior.coverage[index] || continue
        cash = grid.cash[index[1]]
        earnings = grid.log_earnings[index[2]]
        medical = grid.log_medical[index[3]]
        c = interior.consumption[index]
        saving = clamp(cash - c, 0.0, asset_maximum)
        continuation, _, _ = _hsz_expected_continuation(
            config, process, rule, grid, child_value, child_policy,
            saving, earnings, medical, diagnostics; marginal=false)
        diagnostics.q25_candidate_revaluations += 1
        candidate_v = _g2egm_utility(config.gamma, c) +
            config.beta * continuation
        if _hsz_prefer(candidate_v, c, value[index], policy[index], options)
            policy[index], value[index], source[index] = c, candidate_v, 1
        end
    end

    @inbounds for k in eachindex(grid.log_medical),
                  j in eachindex(grid.log_earnings)
        earnings, medical = grid.log_earnings[j], grid.log_medical[k]
        continuation_ceiling, _, _ = _hsz_expected_continuation(
            config, process, rule, grid, child_value, child_policy,
            asset_maximum, earnings, medical, diagnostics; marginal=false)
        diagnostics.q25_candidate_revaluations += 1
        for i in eachindex(grid.cash)
            cash = grid.cash[i]
            cash > asset_maximum || continue
            c = cash - asset_maximum
            candidate_v = _g2egm_utility(config.gamma, c) +
                config.beta * continuation_ceiling
            if _hsz_prefer(candidate_v, c, value[i, j, k],
                policy[i, j, k], options)
                policy[i, j, k], value[i, j, k], source[i, j, k] =
                    c, candidate_v, 0
            end
        end

        kink_data = Tuple{Float64,Float64}[]
        for saving in _hsz_floor_kink_savings(config, process, rule,
            earnings, medical, asset_maximum)
            continuation, _, _ = _hsz_expected_continuation(
                config, process, rule, grid, child_value, child_policy,
                saving, earnings, medical, diagnostics; marginal=false)
            diagnostics.q25_candidate_revaluations += 1
            push!(kink_data, (saving, continuation))
        end
        for i in eachindex(grid.cash), (saving, continuation) in kink_data
            saving < grid.cash[i] || continue
            c = grid.cash[i] - saving
            candidate_v = _g2egm_utility(config.gamma, c) +
                config.beta * continuation
            diagnostics.floor_kink_candidates += 1
            if _hsz_prefer(candidate_v, c, value[i, j, k],
                policy[i, j, k], options)
                policy[i, j, k], value[i, j, k], source[i, j, k] =
                    c, candidate_v, 2
            end
        end
    end

    _repair_hsz_monotonicity!(policy, value, config, process, rule,
        grid, options, child_value, child_policy, diagnostics; winner_source=source)
    all(isfinite, policy) && all(isfinite, value) || throw(ErrorException(
        "the HSZ G2EGM solve produced a nonfinite common-grid value"))
    @inbounds for index in CartesianIndices(policy)
        cash = grid.cash[index[1]]
        0.0 < policy[index] <= cash || throw(ErrorException(
            "the HSZ G2EGM envelope selected an infeasible action"))
        saving = cash - policy[index]
        saving >= asset_maximum - 64eps(max(cash, asset_maximum, 1.0)) &&
            (diagnostics.asset_ceiling_winners += 1)
        if source[index] == 0
            diagnostics.constraint_winners += 1
        elseif source[index] == 1
            diagnostics.tetrahedron_winners += 1
            interior.winner_minimum_weight[index] < 0.0 &&
                (diagnostics.extrapolated_winners += 1)
        elseif source[index] == 2
            diagnostics.floor_kink_winners += 1
        elseif source[index] == 3
            diagnostics.direct_repair_winners += 1
        end
    end
    return policy, value
end

"""Solve HSZ with a pure-Julia three-dimensional tetrahedral G2EGM."""
function build_hsz_solution(config::StochasticPTDConfig, process::HSZProcess;
    base_rule=gauss_hermite_rule(process, 5),
    grid::Union{Nothing,HSZG2EGMGrid}=nothing,
    options::HSZG2EGMOptions=HSZG2EGMOptions())
    rule = base_rule
    rule isa GaussHermiteRule{2,25} || throw(ArgumentError(
        "HSZ G2EGM requires exactly 5x5=25 Gauss-Hermite nodes"))
    _hsz_rules_equal(rule, gauss_hermite_rule(process, 5)) ||
        throw(ArgumentError("the supplied HSZ rule is not canonical Q5xQ5"))
    _validate_hsz_options(options)
    resolved_grid = isnothing(grid) ? hsz_g2egm_grid(process;
        cash_floor=config.cash_floor,
        gross_return=1.0 + config.return_rate) : grid
    _validate_hsz_grid(config, process, rule, resolved_grid)
    horizon = config.terminal.final_date
    policy = Vector{Array{Float64,3}}(undef, horizon)
    value = Vector{Array{Float64,3}}(undef, horizon)
    terminal_policy = repeat(reshape(resolved_grid.cash, :, 1, 1),
        1, length(resolved_grid.log_earnings),
        length(resolved_grid.log_medical))
    terminal_value = _g2egm_utility.(config.gamma, terminal_policy)
    policy[horizon], value[horizon] = terminal_policy, terminal_value
    diagnostics = HSZG2EGMSolveDiagnostics()
    for date in horizon-1:-1:1
        policy[date], value[date] = _solve_hsz_period(
            config, process, rule, resolved_grid, options,
            value[date + 1], policy[date + 1], diagnostics)
    end
    fingerprint = _hsz_array_fingerprint(policy, value)
    return HSZG2EGMSolution(config, process, rule, resolved_grid, options,
        policy, value, diagnostics, HSZG2EGMQueryDiagnostics(), fingerprint)
end

function _checked_hsz_query(solution::HSZG2EGMSolution,
    cash::Float64, earnings::Float64, medical::Float64; value_query::Bool,
    terminal::Bool)
    grid, qd = solution.grid, solution.query_diagnostics
    policy_last = searchsortedlast(grid.cash, last(grid.asset))
    cash_upper = terminal ? last(grid.cash) : grid.cash[policy_last]
    values = (cash, earnings, medical)
    lowers = (first(grid.cash), first(grid.log_earnings),
              first(grid.log_medical))
    uppers = (cash_upper, last(grid.log_earnings), last(grid.log_medical))
    for index in 1:3
        tolerance = 64eps(max(abs(values[index]), abs(lowers[index]),
            abs(uppers[index]), 1.0))
        if values[index] < lowers[index] - tolerance ||
           values[index] > uppers[index] + tolerance
            lock(qd.lock) do
                qd.domain_errors += 1
            end
            throw(DomainError(values[index],
                "HSZ query coordinate is outside the solved domain"))
        end
    end
    clamped = ntuple(i -> clamp(values[i], lowers[i], uppers[i]), 3)
    lock(qd.lock) do
        value_query ? (qd.value_queries += 1) : (qd.consumption_queries += 1)
        qd.minimum_cash_margin = min(qd.minimum_cash_margin,
            clamped[1] - lowers[1], uppers[1] - clamped[1])
        qd.minimum_earnings_margin = min(qd.minimum_earnings_margin,
            clamped[2] - lowers[2], uppers[2] - clamped[2])
        qd.minimum_medical_margin = min(qd.minimum_medical_margin,
            clamped[3] - lowers[3], uppers[3] - clamped[3])
    end
    return clamped
end

function consumption(solution::HSZG2EGMSolution, cash::Real,
    log_earnings::Real, log_medical::Real; date::Integer)
    t = Int(date)
    1 <= t <= solution.config.terminal.final_date || throw(ArgumentError(
        "date is outside the HSZ solution horizon"))
    x, e, m = _checked_hsz_query(solution, Float64(cash),
        Float64(log_earnings), Float64(log_medical);
        value_query=false, terminal=t == solution.config.terminal.final_date)
    t == solution.config.terminal.final_date && return x
    c = clamp(_trilinear(solution.grid.cash,
        solution.grid.log_earnings, solution.grid.log_medical,
        solution.policy[t], x, e, m), 0.0, x)
    if x - c >= last(solution.grid.asset) - 64eps(max(x, 1.0))
        lock(solution.query_diagnostics.lock) do
            solution.query_diagnostics.asset_ceiling_choices += 1
        end
    end
    return c
end

function value_at(solution::HSZG2EGMSolution, cash::Real,
    log_earnings::Real, log_medical::Real; date::Integer)
    t = Int(date)
    x, e, m = _checked_hsz_query(solution, Float64(cash),
        Float64(log_earnings), Float64(log_medical);
        value_query=true, terminal=t == solution.config.terminal.final_date)
    t == solution.config.terminal.final_date &&
        return _g2egm_utility(solution.config.gamma, x)
    return _trilinear(solution.grid.cash, solution.grid.log_earnings,
        solution.grid.log_medical, solution.value[t], x, e, m)
end

function action(solution::HSZG2EGMSolution,
    observed::ObservedState{HSZState})
    return consumption(solution, observed.cash,
        observed.exogenous.log_earnings, observed.exogenous.log_medical;
        date=observed.date)
end

function validate_alignment(solution::HSZG2EGMSolution,
    config::StochasticPTDConfig, process::HSZProcess,
    base_rule::GaussHermiteRule{2,25}, grid::HSZG2EGMGrid=solution.grid;
    options::HSZG2EGMOptions=solution.options)
    solution.config == config && solution.process == process ||
        throw(ArgumentError("HSZ G2EGM solution does not match the model"))
    _hsz_rules_equal(solution.base_rule, base_rule) || throw(ArgumentError(
        "HSZ G2EGM solution does not match the Q25 rule"))
    solution.grid == grid && solution.options == options ||
        throw(ArgumentError("HSZ G2EGM grid/options do not match"))
    solution.fingerprint == _hsz_array_fingerprint(
        solution.policy, solution.value) || throw(ArgumentError(
            "HSZ G2EGM solution fingerprint is invalid"))
    return nothing
end

function reset_query_diagnostics!(solution::HSZG2EGMSolution)
    qd = solution.query_diagnostics
    lock(qd.lock) do
        qd.consumption_queries = qd.value_queries = qd.domain_errors = 0
        qd.asset_ceiling_choices = 0
        qd.minimum_cash_margin = qd.minimum_earnings_margin =
            qd.minimum_medical_margin = Inf
    end
    return qd
end

function diagnostics(solution::HSZG2EGMSolution)
    solve = (; (name => getfield(solution.solve_diagnostics, name)
                for name in fieldnames(HSZG2EGMSolveDiagnostics))...)
    qd = solution.query_diagnostics
    query = lock(qd.lock) do
        (; consumption_queries=qd.consumption_queries,
           value_queries=qd.value_queries, domain_errors=qd.domain_errors,
           asset_ceiling_choices=qd.asset_ceiling_choices,
           minimum_cash_margin=qd.minimum_cash_margin,
           minimum_earnings_margin=qd.minimum_earnings_margin,
           minimum_medical_margin=qd.minimum_medical_margin)
    end
    return (; solve, query)
end

function quadrature_audit(solution::HSZG2EGMSolution)
    return (; family=:positive_gauss_hermite_tensor_product,
        dimensions=2, nodes_per_dimension=5, total_nodes=25,
        standard_normal_nodes=solution.base_rule.nodes,
        normalized_weights=solution.base_rule.weights,
        weight_sum=sum(solution.base_rule.weights))
end
