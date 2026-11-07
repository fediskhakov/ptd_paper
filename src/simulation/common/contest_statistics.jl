using Distributions: TDist, quantile
using Statistics: mean, std

"""Paired Monte Carlo comparison of two policies inside one outer contest."""
struct PairedComparison
    method_a::Symbol
    method_b::Symbol
    n_paths::Int
    confidence_level::Float64
    mean_difference::Float64
    standard_error::Float64
    ci_lower::Float64
    ci_upper::Float64
end

"""
Processed statistics for one expected-utility contest.

`expected_utility` contains within-contest column means only. `point_winners`
contains every method attaining the exact largest sample mean, so an exact tie
is never broken by method order or by an arbitrary numerical tolerance.
`classification` is one of:

  * `:resolved` -- one method's paired interval is strictly favorable against
    every competitor;
  * `:mc_unresolved` -- there is a unique sample-mean leader, but at least one
    paired interval includes zero; or
  * `:point_tie` -- two or more methods have exactly equal largest sample means.

The singular `point_winner` is `nothing` for a point tie, and
`resolved_winner` is `nothing` for either unresolved classification.
"""
struct ContestSummary
    contest_id::Int
    panel::Symbol
    method_ids::Vector{Symbol}
    n_paths::Int
    confidence_level::Float64
    expected_utility::Vector{Float64}
    point_winners::Vector{Symbol}
    point_winner::Union{Nothing,Symbol}
    resolved_winner::Union{Nothing,Symbol}
    classification::Symbol
    comparisons::Vector{PairedComparison}
end

function _confidence_level(level::Real)
    value = Float64(level)
    isfinite(value) && 0.0 < value < 1.0 ||
        throw(ArgumentError("level must be finite and lie in (0,1)"))
    return value
end

function _validate_utilities(method_ids::AbstractVector{Symbol},
                             utilities::AbstractMatrix{<:Real})
    size(utilities, 1) >= 2 ||
        throw(ArgumentError("at least two paths are required for uncertainty estimates"))
    size(utilities, 2) == length(method_ids) || throw(DimensionMismatch(
        "utilities must have one column per method"))
    length(method_ids) >= 2 ||
        throw(ArgumentError("at least two methods are required for a contest"))
    length(unique(method_ids)) == length(method_ids) ||
        throw(ArgumentError("method identifiers must be unique"))
    all(isfinite, utilities) ||
        throw(ArgumentError("pathwise utilities must all be finite"))
    return nothing
end

"""
    paired_comparison(method_a, method_b, utility_a, utility_b; level=0.95)

Compare expected utilities using common-path differences. The standard error is
the sample standard deviation of the paired differences divided by `sqrt(K)`;
the interval uses the two-sided Student-t critical value with `K-1` degrees of
freedom. No additional model solve or simulation is performed.
"""
function paired_comparison(method_a::Symbol,
                           method_b::Symbol,
                           utility_a::AbstractVector{<:Real},
                           utility_b::AbstractVector{<:Real};
                           level::Real=0.95)
    method_a != method_b ||
        throw(ArgumentError("a paired comparison requires two distinct methods"))
    length(utility_a) == length(utility_b) ||
        throw(DimensionMismatch("paired utility vectors must have equal length"))
    n = length(utility_a)
    n >= 2 || throw(ArgumentError("at least two paired paths are required"))
    confidence_level = _confidence_level(level)

    values_a = Float64.(utility_a)
    values_b = Float64.(utility_b)
    all(isfinite, values_a) && all(isfinite, values_b) ||
        throw(ArgumentError("paired utility values must be finite Float64 values"))
    differences = values_a .- values_b
    all(isfinite, differences) ||
        throw(ArgumentError("paired utility differences must be finite"))

    average = mean(differences)
    standard_error = std(differences; corrected=true) / sqrt(n)
    critical = quantile(TDist(n - 1), 0.5 + confidence_level / 2)
    half_width = critical * standard_error
    ci_lower = average - half_width
    ci_upper = average + half_width
    all(isfinite, (average, standard_error, half_width, ci_lower, ci_upper)) ||
        throw(ArgumentError("paired Monte Carlo statistics overflowed Float64"))
    return PairedComparison(
        method_a,
        method_b,
        n,
        confidence_level,
        average,
        standard_error,
        ci_lower,
        ci_upper,
    )
end

function _favorable(comparison::PairedComparison, candidate::Symbol)
    if comparison.method_a == candidate
        return comparison.ci_lower > 0.0
    elseif comparison.method_b == candidate
        return comparison.ci_upper < 0.0
    end
    return true
end

"""
    summarize_contest(contest_id, panel, method_ids, utilities; level=0.95)

`utilities[k,m]` is the realized discounted payoff of method `m` on hidden
path `k`. Column means estimate contest-specific expected utility. A resolved
winner must have a strictly favorable paired confidence interval relative to
every other method; intervals touching zero remain unresolved.
"""
function summarize_contest(contest_id::Int,
                           panel::Symbol,
                           method_ids::AbstractVector{Symbol},
                           utilities::AbstractMatrix{<:Real};
                           level::Real=0.95)
    contest_id > 0 || throw(ArgumentError("contest_id must be positive"))
    panel in (:baseline, :high_variance, :ar1) ||
        throw(ArgumentError("unknown risk panel"))
    _validate_utilities(method_ids, utilities)
    confidence_level = _confidence_level(level)
    ids = collect(method_ids)
    values = Float64.(utilities)
    expected = vec(mean(values; dims=1))
    all(isfinite, expected) ||
        throw(ArgumentError("within-contest expected utilities overflowed Float64"))

    largest = maximum(expected)
    point_indices = findall(==(largest), expected)
    point_winners = ids[point_indices]
    point_winner = length(point_winners) == 1 ? only(point_winners) : nothing

    # Store every unordered pair, not only comparisons against a sample leader.
    # This keeps processed output sufficient for pairwise diagnostics without
    # rerunning any policy or simulation.
    comparisons = PairedComparison[]
    for method_a in 1:length(ids)-1, method_b in method_a+1:length(ids)
        push!(comparisons, paired_comparison(
            ids[method_a],
            ids[method_b],
            view(values, :, method_a),
            view(values, :, method_b);
            level=confidence_level,
        ))
    end

    # Evaluate every method symmetrically. This makes the result invariant to
    # method ordering and avoids selecting a putative winner before uncertainty
    # is assessed. At most one method can have intervals strictly favorable
    # against every alternative.
    resolved_candidates = [method for method in ids
        if all(comparison -> _favorable(comparison, method), comparisons)]
    length(resolved_candidates) <= 1 || error(
        "internally inconsistent paired intervals produced multiple winners")
    resolved_winner = isempty(resolved_candidates) ? nothing : only(resolved_candidates)

    classification = if !isnothing(resolved_winner)
        :resolved
    elseif length(point_winners) > 1
        :point_tie
    else
        :mc_unresolved
    end

    return ContestSummary(
        contest_id,
        panel,
        ids,
        size(values, 1),
        confidence_level,
        expected,
        point_winners,
        point_winner,
        resolved_winner,
        classification,
        comparisons,
    )
end

function _comparison_dictionary(summary::ContestSummary)
    dictionary = Dict{Tuple{Symbol,Symbol},PairedComparison}()
    for comparison in summary.comparisons
        key = (comparison.method_a, comparison.method_b)
        haskey(dictionary, key) && throw(ArgumentError(
            "contest $(summary.contest_id) contains duplicate paired comparisons"))
        dictionary[key] = comparison
    end
    return dictionary
end

"""
    aggregate_outcomes(summaries)

Aggregate classification counts and preserve the distribution of paired
outcomes for one variance panel. Raw expected-utility levels and raw utility
gaps are deliberately absent from the returned object: CRRA coefficients may
differ across outer contests, so neither levels nor differences are comparable
across contests. Contest-specific paired gaps remain available in each
`ContestSummary.comparisons` record for uncertainty classification and for
conversion to a common welfare scale before any cross-contest aggregation.

Exact sample-mean ties (`point_ties`) and Monte Carlo-unresolved contests
(`mc_unresolved`) are reported separately. `unresolved` includes both and is
exactly the number of contests without a clear paired-interval winner.
"""
function aggregate_outcomes(summaries::AbstractVector{ContestSummary})
    isempty(summaries) && throw(ArgumentError("at least one contest is required"))
    first_summary = summaries[1]
    method_ids = first_summary.method_ids
    panel = first_summary.panel
    n_paths = first_summary.n_paths
    confidence_level = first_summary.confidence_level

    all(summary -> summary.method_ids == method_ids, summaries) ||
        throw(ArgumentError("all summaries must use the same ordered method identifiers"))
    all(summary -> summary.panel == panel, summaries) ||
        throw(ArgumentError("aggregate each risk panel separately"))
    all(summary -> summary.n_paths == n_paths, summaries) ||
        throw(ArgumentError("all summaries must use the same number of hidden paths"))
    all(summary -> summary.confidence_level == confidence_level, summaries) ||
        throw(ArgumentError("all summaries must use the same confidence level"))
    contest_ids = getfield.(summaries, :contest_id)
    length(unique(contest_ids)) == length(contest_ids) ||
        throw(ArgumentError("contest identifiers must be unique within a panel"))

    expected_pairs = [(method_ids[a], method_ids[b])
        for a in 1:length(method_ids)-1 for b in a+1:length(method_ids)]
    point_counts = Dict(method => 0 for method in method_ids)
    resolved_counts = Dict(method => 0 for method in method_ids)
    resolved_loss_counts = Dict(method => 0 for method in method_ids)
    tie_counts = Dict{Tuple{Vararg{Symbol}},Int}()
    point_ties = 0
    mc_unresolved = 0

    for summary in summaries
        comparison_dictionary = _comparison_dictionary(summary)
        Set(keys(comparison_dictionary)) == Set(expected_pairs) ||
            throw(ArgumentError(
                "contest $(summary.contest_id) does not contain every expected method pair"))
        for pair in expected_pairs
            comparison = comparison_dictionary[pair]
            comparison.n_paths == n_paths || throw(ArgumentError(
                "paired comparison path count disagrees with its contest summary"))
            comparison.confidence_level == confidence_level || throw(ArgumentError(
                "paired comparison confidence level disagrees with its contest summary"))
        end

        if isnothing(summary.point_winner)
            point_ties += 1
            tie_key = Tuple(summary.point_winners)
            tie_counts[tie_key] = get(tie_counts, tie_key, 0) + 1
        else
            winner = summary.point_winner::Symbol
            point_counts[winner] += 1
        end

        if isnothing(summary.resolved_winner)
            summary.classification == :mc_unresolved && (mc_unresolved += 1)
        else
            winner = summary.resolved_winner::Symbol
            resolved_counts[winner] += 1
            for method in method_ids
                method == winner || (resolved_loss_counts[method] += 1)
            end
        end
    end

    n_contests = length(summaries)
    unresolved = n_contests - sum(values(resolved_counts))
    point_shares = Dict(method => point_counts[method] / n_contests
                        for method in method_ids)
    resolved_win_shares = Dict(method => resolved_counts[method] / n_contests
                               for method in method_ids)
    resolved_loss_shares = Dict(method => resolved_loss_counts[method] / n_contests
                                for method in method_ids)
    classification_counts = (
        resolved=n_contests - unresolved,
        mc_unresolved=mc_unresolved,
        point_tie=point_ties,
    )

    return (;
        panel,
        n_contests,
        n_paths,
        confidence_level,
        method_ids,
        point_counts,
        point_shares,
        point_ties,
        point_tie_share=point_ties / n_contests,
        tie_counts,
        resolved_counts,
        resolved_win_shares,
        resolved_loss_counts,
        resolved_loss_shares,
        mc_unresolved,
        mc_unresolved_share=mc_unresolved / n_contests,
        unresolved,
        unresolved_share=unresolved / n_contests,
        classification_counts,
    )
end
