include("deterministic_model.jl")
using Roots

const FUSION_XRTOL = 1e-12
const FUSION_POSITIVE_PROBE = sqrt(eps(Float64))
const FUSION_VALUE_RTOL = 32 * eps(Float64)
const DOMAIN_SNAP_ULPS = UInt64(4)

"""An unbounded comparison that could not be certified before the search cap."""
struct FusionTailError <: Exception
    path::Int
    period::Int
    search_bound::Float64
    parameters::NamedTuple
    elder::PolicySegment
    younger::PolicySegment
end

function Base.showerror(io::IO, error::FusionTailError)
    print(io, "PTD fusion reached its finite search bound without certifying the tail",
        " at path ", error.path, ", period ", error.period,
        ", search_bound=", error.search_bound, "; parameters=", error.parameters,
        "; elder=", error.elder, "; younger=", error.younger,
        ". Experiment stopped; an uncertified tail was not accepted.")
end

"""Thread-safe record of unresolved tails for an explicitly shared options object."""
struct FusionTailDiagnostics
    lock::ReentrantLock
    events::Vector{FusionTailError}
end
FusionTailDiagnostics() = FusionTailDiagnostics(ReentrantLock(), FusionTailError[])

"""
    FusionOptions(; search_bound=1e16, max_expansions=64,
        allow_uncertified_tail=false, diagnostics=FusionTailDiagnostics())

Finite safeguards for comparing candidates on an unbounded wealth interval.
Geometric bracketing stops at `search_bound`; the final permitted expansion
jumps to that bound. An unresolved comparison is recorded and raises
`FusionTailError`, so experiment runs cannot silently accept an uncertified tail.
Diagnostic callers may explicitly opt into the old approximate endpoint choice
with `allow_uncertified_tail=true`; this still records and warns about the event.
A finite cap is not an analytical certificate of global optimality.
"""
struct FusionOptions
    search_bound::Float64
    max_expansions::Int
    allow_uncertified_tail::Bool
    diagnostics::FusionTailDiagnostics

    function FusionOptions(; search_bound::Real=1e16,
        max_expansions::Integer=64, allow_uncertified_tail::Bool=false,
        diagnostics::FusionTailDiagnostics=FusionTailDiagnostics())
        bound = Float64(search_bound)
        isfinite(bound) && bound > 0.0 || throw(ArgumentError(
            "The fusion search bound must be finite and positive."))
        max_expansions > 0 || throw(ArgumentError(
            "The fusion expansion limit must be positive."))
        return new(bound, Int(max_expansions), allow_uncertified_tail, diagnostics)
    end
end

"""Snapshot the unresolved-tail count and identifying events for these options."""
function fusion_tail_diagnostics(options::FusionOptions)
    lock(options.diagnostics.lock) do
        events = copy(options.diagnostics.events)
        return (; unresolved_count=length(events), events)
    end
end

function _record_unresolved_tail!(options::FusionOptions,
    problem::DeterministicProblem, elder::PolicySegment, younger::PolicySegment;
    i::Int, t::Int)
    parameters = (; beta=problem.β, gamma=problem.γ, r=problem.r,
        floor=problem.xFloor, incomes=copy(problem.incomes[i, :]))
    event = FusionTailError(i, t, options.search_bound, parameters, elder, younger)
    lock(options.diagnostics.lock) do
        push!(options.diagnostics.events, event)
    end
    options.allow_uncertified_tail || throw(event)
    @warn "PTD fusion reached its finite search bound; the selected tail is an approximation, and earlier decisions visiting it are not certified." path=i period=t search_bound=options.search_bound parameters
    return nothing
end


function _within_domain_ulps(left::Float64, right::Float64;
    max_ulps::UInt64=DOMAIN_SNAP_ULPS)
    all(isfinite, (left, right)) && left >= 0.0 && right >= 0.0 || return false
    left_bits = reinterpret(UInt64, left)
    right_bits = reinterpret(UInt64, right)
    distance = left_bits >= right_bits ? left_bits - right_bits : right_bits - left_bits
    return distance <= max_ulps
end


function _ordered_envelope_error(detail::AbstractString;
    i::Int64, t::Int64, ps1::PolicySegment, ps2::PolicySegment)
    throw(ErrorException("Deterministic PTD ordered-envelope invariant failed " *
                         "at path $i, period $t: $detail. elder=$ps1, younger=$ps2"))
end


"""Assert that the younger candidate weakly saves more on its overlap."""
function _assert_ordered_saving(ps1::PolicySegment, ps2::PolicySegment,
    a::Float64, b::Float64; i::Int64, t::Int64)

    a <= b || _ordered_envelope_error("candidate domains do not intersect";
        i=i, t=t, ps1=ps1, ps2=ps2)
    probes = isfinite(b) ? (a, b) : (a,)
    for x in probes
        elderSaving = _policy_saving(ps1, x)
        youngerSaving = _policy_saving(ps2, x)
        tol = _policy_tolerance(x, elderSaving, youngerSaving)
        youngerSaving >= elderSaving - tol ||
            _ordered_envelope_error(
                "monotone saving is violated at x=$x " *
                "($elderSaving -> $youngerSaving)";
                i=i, t=t, ps1=ps1, ps2=ps2)
    end

    # On an unbounded overlap, checking any finite collection of wealth points
    # is insufficient: the affine saving difference could turn negative farther
    # out.  Its slope is `(1-b1_younger) - (1-b1_elder)`, so nonnegativity at
    # the left endpoint plus a nonnegative slope is necessary and sufficient on
    # the entire ray.
    if isinf(b)
        elderSlope = 1.0 - ps1.b1
        youngerSlope = 1.0 - ps2.b1
        youngerSlope >= elderSlope ||
            _ordered_envelope_error(
                "saving-order slope is negative on the unbounded overlap " *
                "($elderSlope -> $youngerSlope)";
                i=i, t=t, ps1=ps1, ps2=ps2)
    end
    return nothing
end


"""Validate domains and the total saving order inherited from the parent policy."""
function _validate_ordered_candidates(pc::Vector{PolicySegment}; i::Int64, t::Int64)
    isempty(pc) && throw(ErrorException(
        "No deterministic PTD candidates at path $i, period $t."))
    for (j, ps) in pairs(pc)
        validate_policy_collection([ps];
            context="candidate $j at path $i, period $t", require_coverage=false)
        for elderIndex in 1:j-1
            elder = pc[elderIndex]
            # Inverse Euler propagation can make either endpoint decrease: a
            # later, higher-saving candidate may overlap or nest inside any
            # earlier domain after intervening candidates are removed. The fusion operator
            # relies on the inherited total order, so certify every feasible
            # elder--younger overlap before mutating the candidate collection.
            a = max(elder.xStart, ps.xStart)
            b = min(elder.xEnd, ps.xEnd)
            a <= b && _assert_ordered_saving(elder, ps, a, b; i=i, t=t)
        end
    end
    return nothing
end


function _assert_representable_value_order(cmp_a::Int64, cmp_b::Int64;
    i::Int64, t::Int64, ps1::PolicySegment, ps2::PolicySegment)
    if cmp_a < 0 && cmp_b > 0
        _ordered_envelope_error(
            "candidate values cross in the forbidden direction";
            i=i, t=t, ps1=ps1, ps2=ps2)
    elseif cmp_a == 0 && cmp_b > 0
        # A tolerance tie owned by the higher-saving candidate at the left
        # endpoint followed by strict elder dominance cannot be encoded by an
        # ordered half-open segment collection.  Failing loudly is preferable
        # to silently violating either the tie rule or monotone saving.
        _ordered_envelope_error(
            "a save-more tie at the left endpoint is followed by elder dominance";
            i=i, t=t, ps1=ps1, ps2=ps2)
    end
    return nothing
end


"""Merge adjacent copies of exactly the same affine policy."""
function _coalesce_duplicate_segments(policy::Vector{PolicySegment})
    result = PolicySegment[]
    for ps in policy
        if !isempty(result) && result[end].xEnd == ps.xStart &&
           result[end].b0 == ps.b0 && result[end].b1 == ps.b1
            elder = result[end]
            result[end] = PolicySegment(xStart=elder.xStart, xEnd=ps.xEnd,
                b0=elder.b0, b1=elder.b1)
        else
            push!(result, ps)
        end
    end
    return result
end


"""
    descendant_operator(problem::DeterministicProblem, ps::PolicySegment, yNext::Float64)::PolicySegment

Find the policy child of parent policy segment `ps` for `problem` given
the income level at the period of `ps` is `yNext`.
"""
function descendant_operator(problem::DeterministicProblem, ps::PolicySegment; yNext::Float64)::PolicySegment

    # An Euler descendant is valid only on a nondegenerate interval where current
    # assets are positive and the next-period floor is not binding. Its stored
    # closed domain includes the limiting boundary. A zero-width closure is not a
    # distinct branch: at zero assets it duplicates consume-all, while a shared
    # parent endpoint is owned by the adjacent segment on the right.
    zLower = max(ps.xStart, yNext, problem.xFloor)
    zUpper = ps.xEnd

    if zUpper ≤ zLower
        return NaNPolicySegment
    end

    # With CRRA utility, inverse Euler propagation of the affine parent
    # c⁺(z)=b₀+b₁z is affine exactly.  Computing its coefficients directly
    # avoids reconstructing the line from two floating-point evaluations and,
    # in particular, removes the arbitrary finite proxy formerly used when the
    # parent has an unbounded domain.
    R = 1.0 + problem.r
    κ = (problem.β * R)^(-1.0 / problem.γ)
    q = κ * ps.b1 * R
    denom = 1.0 + q
    slope = q / denom
    intercept = κ * (ps.b0 + ps.b1 * yNext) / denom

    # Parametric endpoint map x(z)=(z-y)/R+κc⁺(z) avoids cancellation in
    # division by 1-slope when the child is close to the 45-degree line.
    map_state(z) = (z - yNext) / R + κ * (ps.b0 + ps.b1 * z)
    xStart = map_state(zLower)
    xEnd = isinf(zUpper) ? Inf64 : map_state(zUpper)

    # A representable parent interval can collapse to one floating-point state
    # under the affine map. It then carries no numerically representable domain.
    xEnd ≤ xStart && return NaNPolicySegment

    if all(isfinite, (intercept, slope, xStart)) &&
       (isfinite(xEnd) || xEnd == Inf64) && 0.0 ≤ slope < 1.0
        return PolicySegment(xStart=xStart, xEnd=xEnd, b0=intercept, b1=slope)
    end

    throw(ErrorException("Invalid Euler descendant for parent $ps: " *
                         "domain=[$xStart, $xEnd], b0=$intercept, b1=$slope"))
end


"""
    descendant_operator(problem::DeterministicProblem; i::Int64, t::Int64)::Vector{PolicySegment}

Find all the policy children at **t-1** of policy function
at `t` on `problem`'s income path `i`."""
function descendant_operator(problem::DeterministicProblem; i::Int64, t::Int64)::Vector{PolicySegment}

    y = problem.incomes[i, t]
    policy = problem.policy[i, t]
    validate_policy_collection(policy;
        context="parent policy at path $i, period $t")
    children = PolicySegment[]
    for ps in policy
        child = descendant_operator(problem, ps, yNext=y)
        isnan(child) || push!(children, child)
    end
    isempty(children) && throw(ErrorException(
        "No reachable Euler descendants at path $i, period $(t-1)."))
    _validate_ordered_candidates(children; i=i, t=t-1)
    return children
end


"""Evaluate both candidate actions using the model's actual floor transition."""
function _candidate_values(x::Float64, problem::DeterministicProblem,
    ps1::PolicySegment, ps2::PolicySegment; i::Int64, t::Int64)

    c1, c2 = c_x(ps1, x), c_x(ps2, x)
    a1, a2 = x - c1, x - c2
    xNext_a1 = xNext_a(problem, a1, i=i, t=t)
    v1 = u(problem, c1) + problem.β * v_x(problem, xNext_a1, i=i, t=t + 1)

    xNext_a2 = xNext_a(problem, a2, i=i, t=t)
    v2 = u(problem, c2) + problem.β * v_x(problem, xNext_a2, i=i, t=t + 1)

    return v1, v2
end


"""
    candidate_value_difference(x, problem, elder, younger; i, t)

Bellman objective of the elder action minus that of the younger action, each
followed by the already solved continuation policy. Both candidates must be
feasible at `x`; positive saving never disables the cash-floor transition.
"""
function candidate_value_difference(x::Float64, problem::DeterministicProblem,
    ps1::PolicySegment, ps2::PolicySegment; i::Int64, t::Int64)::Float64
    v1, v2 = _candidate_values(x, problem, ps1, ps2, i=i, t=t)
    return v1 - v2
end


"""
Compare two candidate values with a relative roundoff tolerance.  A zero result
is a numerical value tie, resolved by the ordered envelope in favour of the
weakly higher-saving (younger) candidate.
"""
function _value_comparison(x::Float64, problem::DeterministicProblem,
    ps1::PolicySegment, ps2::PolicySegment; i::Int64, t::Int64)::Int64

    v1, v2 = _candidate_values(x, problem, ps1, ps2, i=i, t=t)
    if isnan(v1) || isnan(v2)
        _ordered_envelope_error(
            "undefined candidate value at x=$x (elder=$v1, younger=$v2)";
            i=i, t=t, ps1=ps1, ps2=ps2)
    end
    v1 == v2 && return 0
    (isinf(v1) || isinf(v2)) && return v1 > v2 ? 1 : -1
    difference = v1 - v2
    tolerance = FUSION_VALUE_RTOL * max(1.0, abs(v1), abs(v2))
    return abs(difference) <= tolerance ? 0 : (difference > 0.0 ? 1 : -1)
end


"""
    fusion_boundary(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment,
    a::Float64, b::Float64; i::Int64, t::Int64)::Float64

Given `problem`'s two policy segments `ps1` and `ps2` on the income path `i`
at time `t`, where `ps1` is the elder brother, find the the intersection
of two value functions derived from `ps1` and `ps2` on `[a, b] ⊂ R`.

# Returns
- `a` if `ps1` is dominated by `ps2` on `[a, b]`.
- `b` if `ps2` is dominated by `ps1` on `[a, b]`.
- `c` ∈ `(a, b)` if `ps1` is better on `[a, c]` and `ps2` is better on `[c, b]`.
"""
function fusion_boundary(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment,
    a::Float64, b::Float64; i::Int64, t::Int64)::Float64

    a < b || _ordered_envelope_error(
        "a nonempty comparison interval is required; received [$a, $b]";
        i=i, t=t, ps1=ps1, ps2=ps2)
    _assert_ordered_saving(ps1, ps2, a, b; i=i, t=t)

    # CRRA utility may be -Inf at zero, making a direct value difference NaN.
    # The policy choice at the zero-measure endpoint is immaterial, so evaluate
    # the right limit at a small positive point inside the overlap instead.
    aProbe = if a == 0.0 && b > 0.0
        min(FUSION_POSITIVE_PROBE, a + 0.01 * (b - a))
    else
        a
    end

    cmp_a = _value_comparison(aProbe, problem, ps1, ps2, i=i, t=t)
    cmp_b = _value_comparison(b, problem, ps1, ps2, i=i, t=t)
    _assert_representable_value_order(cmp_a, cmp_b;
        i=i, t=t, ps1=ps1, ps2=ps2)

    if cmp_a > 0 && cmp_b < 0
        return find_zero(x -> candidate_value_difference(x, problem, ps1, ps2, i=i, t=t),
            (aProbe, b), Roots.Brent(), xrtol=FUSION_XRTOL)
    elseif cmp_a == 0 && cmp_b == 0
        # This includes duplicate candidates and a value tangency.  Inspect an
        # interior point so a tangency does not create a spurious switch.  If
        # the candidates remain tied, the younger candidate owns the overlap:
        # vector policy lookup is half-open and therefore implements the same
        # save-more rule at the boundary.
        midpoint = aProbe + (b - aProbe) / 2.0
        cmp_mid = _value_comparison(midpoint, problem, ps1, ps2, i=i, t=t)
        cmp_mid > 0 && _ordered_envelope_error(
            "a save-more endpoint tie encloses strict elder dominance";
            i=i, t=t, ps1=ps1, ps2=ps2)
        return a
    elseif cmp_a >= 0 && cmp_b >= 0  # ps1 dominates ps2 on [a, b]
        return b
    elseif cmp_a <= 0 && cmp_b <= 0  # ps2 dominates ps1 on [a, b]
        return a
    else
        _ordered_envelope_error(
            "unclassified endpoint comparisons ($cmp_a, $cmp_b) on [$a, $b]";
            i=i, t=t, ps1=ps1, ps2=ps2)
    end
end


"""
    pairwise_fusion_step(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment;
    i::Int64, t::Int64)::Tuple{PolicySegment, PolicySegment}

Apply the paper's pairwise fusion step to ordered candidates `ps1` and `ps2`.
"""
function pairwise_fusion_step(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment;
    i::Int64, t::Int64, options::FusionOptions=FusionOptions())::Tuple{PolicySegment,PolicySegment}

    if isinf(ps1.xEnd) && isinf(ps2.xEnd) # This always happens at T-1.
        return pairwise_fusion_step_unbounded(problem, ps1, ps2; i, t, options)
    else
        return pairwise_fusion_step_bounded(problem, ps1, ps2, i=i, t=t)
    end
end


"""
    pairwise_fusion_step_bounded(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment;
    i::Int64, t::Int64)::Tuple{PolicySegment, PolicySegment}

Fuse one policy segment `ps1` with its younger brother policy segment `ps2`
on the income path `i` at time `t` of `problem`
in the policy children collection by finding the kink,
where **at least one of the domains of ps1, ps2 is bounded**.
"""
function pairwise_fusion_step_bounded(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment;
    i::Int64, t::Int64)::Tuple{PolicySegment,PolicySegment}

    a, b = max(ps1.xStart, ps2.xStart), min(ps1.xEnd, ps2.xEnd)
    if a > b
        forward_gap = ps1.xEnd < ps2.xStart
        if forward_gap && _within_domain_ulps(ps1.xEnd, ps2.xStart)
            # The exact endpoint map can round a mathematically shared boundary
            # to adjacent Float64 values.  Snap the younger segment left by at
            # most DOMAIN_SNAP_ULPS so the solved collection remains exactly
            # covered; larger gaps remain hard errors.
            ps2New = PolicySegment(xStart=ps1.xEnd, xEnd=ps2.xEnd,
                b0=ps2.b0, b1=ps2.b1)
            return ps1, ps2New
        end
        _ordered_envelope_error("candidate domains do not intersect";
            i=i, t=t, ps1=ps1, ps2=ps2)
    end
    _assert_ordered_saving(ps1, ps2, a, b; i=i, t=t)
    if a == b # ps1 and ps2 are exactly adjacent.
        ps1New = ps1
        ps2New = PolicySegment(xStart=ps1.xEnd, xEnd=ps2.xEnd, b0=ps2.b0, b1=ps2.b1)
    elseif a < b # b < Inf since at least one of ps1, ps2 are bounded.
        c = fusion_boundary(problem, ps1, ps2, a, b, i=i, t=t)
        if a < c < b
            ps1New = PolicySegment(xStart=ps1.xStart, xEnd=c, b0=ps1.b0, b1=ps1.b1)
            ps2New = PolicySegment(xStart=c, xEnd=ps2.xEnd, b0=ps2.b0, b1=ps2.b1)
        elseif a == c # ps2 dominates ps1 on the common domain.
            ps1New = (a > ps1.xStart
                      ? PolicySegment(xStart=ps1.xStart, xEnd=a,
                b0=ps1.b0, b1=ps1.b1)
                      : NaNPolicySegment)
            ps2New = ps2
        elseif b == c # ps1 dominates ps2 on the common domain.
            ps1New = ps1
            ps2New = (b < ps2.xEnd
                      ? PolicySegment(xStart=b, xEnd=ps2.xEnd, b0=ps2.b0, b1=ps2.b1)
                      : NaNPolicySegment)
        else
            throw(ErrorException("a=$a < c=$c < b=$b doesn't hold!"))
        end
    end

    return ps1New, ps2New
end


"""
    pairwise_fusion_step_unbounded(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment;
    i::Int64, t::Int64)::Tuple{PolicySegment, PolicySegment}

Fuse the *corner* policy segment `ps1` with its younger brother policy segment `ps2`
on the income path `i` at time `t` of `problem` in the policy children collection
by finding the kink, where **both the domains of ps1, ps2 are unbounded**.
(This always happens at period T-1)
"""
function pairwise_fusion_step_unbounded(problem::DeterministicProblem, ps1::PolicySegment, ps2::PolicySegment;
    i::Int64, t::Int64, options::FusionOptions=FusionOptions())::Tuple{PolicySegment,PolicySegment}

    a = max(ps1.xStart, ps2.xStart)
    a < options.search_bound || throw(ArgumentError(
        "The fusion search bound $(options.search_bound) must exceed the " *
        "candidate overlap start $a; increase FusionOptions.search_bound."))
    _assert_ordered_saving(ps1, ps2, a, Inf64; i=i, t=t)
    probe = ps2.xStart == 0.0 ? FUSION_POSITIVE_PROBE : ps2.xStart
    left_comparison = _value_comparison(
        probe, problem, ps1, ps2, i=i, t=t)
    if left_comparison <= 0
        if left_comparison == 0
            # Do not let the unbounded fast path bypass the representability
            # checks in `fusion_boundary`.  A genuine/tolerance tie must remain owned
            # by the higher-saving candidate on a finite diagnostic interval;
            # forbidden elder dominance raises there.
            diagnostic_upper = min(options.search_bound,
                max(2.0 * max(ps2.xStart, probe), 1.0))
            diagnostic_kink = fusion_boundary(problem, ps1, ps2,
                ps2.xStart, diagnostic_upper, i=i, t=t)
            diagnostic_kink == ps2.xStart || _ordered_envelope_error(
                "unbounded left tie did not resolve to the save-more candidate";
                i=i, t=t, ps1=ps1, ps2=ps2)
        end
        # ps2 is always better at its domain since ps1.xStart ≤ ps2.xStart.
        ps1New = ps1.xStart < ps2.xStart ?
                 PolicySegment(xStart=ps1.xStart, xEnd=ps2.xStart,
            b0=ps1.b0, b1=ps1.b1) : NaNPolicySegment
        ps2New = ps2
    else
        b = min(max(2.0 * ps2.xStart, 1.0), options.search_bound)
        c = b
        for expansion in 1:options.max_expansions
            # Always inspect the declared bound before exhausting the budget.
            expansion == options.max_expansions && (b = options.search_bound)
            c = fusion_boundary(problem, ps1, ps2, ps2.xStart, b, i=i, t=t)
            c == b || break
            if b == options.search_bound
                comparison = _value_comparison(b, problem, ps1, ps2; i, t)
                _record_unresolved_tail!(options, problem, ps1, ps2; i, t)
                # A strict elder win means no switch on the searched overlap
                # under single crossing. A numerical tie is owned by the
                # higher-saving younger segment at the cap.
                comparison > 0 && return ps1, NaNPolicySegment
                break
            end
            b = min(b > options.search_bound / 2.0 ?
                options.search_bound : 2.0 * b, options.search_bound)
        end
        ps1New = PolicySegment(xStart=ps1.xStart, xEnd=c, b0=ps1.b0, b1=ps1.b1)
        ps2New = PolicySegment(xStart=c, xEnd=ps2.xEnd, b0=ps2.b0, b1=ps2.b1)
    end

    return ps1New, ps2New
end


"""
    last_overlapping_segment(pc::Vector{PolicySegment}, ps::PolicySegment)::Int64

Find the index of the last policy segment in `pc` such that
its domain has non-empty intersection with `ps`.
"""
function last_overlapping_segment(pc::Vector{PolicySegment}, ps::PolicySegment)::Int64
    ind = findlast(ps2 -> max(ps2.xStart, ps.xStart) ≤
                          min(ps2.xEnd, ps.xEnd), pc)
    # If ps.xStart > pc[end].xEnd, return length(pc).
    return isnothing(ind) ? length(pc) : ind
end


"""
    fusion_operator(problem::DeterministicProblem, pc::Vector{PolicySegment};
    i::Int64, t::Int64)::Vector{PolicySegment}

Apply the paper's fusion operator to the ordered policy children `pc`, seeded
with consume-all. Partition at every domain endpoint and fuse only candidates
that cover the same cell, preserving both outside-domain pieces. This reference
sweep uses at most quadratically many candidate comparisons, then coalesces
adjacent copies of the same affine policy.
"""
function fusion_operator(problem::DeterministicProblem, pc::Vector{PolicySegment};
    i::Int64, t::Int64, options::FusionOptions=FusionOptions())::Vector{PolicySegment}

    _validate_ordered_candidates(pc; i=i, t=t)
    corner = PolicySegment(xStart=0.0, xEnd=Inf64, b0=0.0, b1=1.0)
    candidates = vcat([corner], pc)

    # A bounded younger candidate can end before an incumbent. Partitioning
    # first preserves the incumbent outside that overlap; simply replacing the
    # last stack entry could incorrectly discard its surviving right suffix.
    endpoints = sort!(unique!(vcat([0.0, Inf64],
        [ps.xStart for ps in pc], [ps.xEnd for ps in pc])))
    boundaries = Float64[]
    for endpoint in endpoints
        # Inverse Euler maps of a shared parent join can differ by a few ULPs.
        # Preserve the earlier endpoint and snap only these roundoff-sized gaps.
        if isempty(boundaries) || !_within_domain_ulps(boundaries[end], endpoint)
            push!(boundaries, endpoint)
        end
    end

    policy = PolicySegment[]
    for cell in 1:length(boundaries)-1
        lower, upper = boundaries[cell], boundaries[cell+1]
        envelope = PolicySegment[]
        for candidate in candidates
            begins_before = candidate.xStart <= lower ||
                _within_domain_ulps(candidate.xStart, lower)
            ends_after = candidate.xEnd >= upper ||
                _within_domain_ulps(candidate.xEnd, upper)
            begins_before && ends_after || continue
            younger = PolicySegment(xStart=lower, xEnd=upper,
                b0=candidate.b0, b1=candidate.b1)

            # Every active candidate initially covers this entire cell. Each
            # accepted switch therefore leaves the younger candidate covering
            # its right endpoint; no outside-domain incumbent is discarded.
            while !isempty(envelope)
                elder = pop!(envelope)
                left, right = pairwise_fusion_step(problem, elder, younger;
                    i, t, options)
                if !isnan(left)
                    push!(envelope, left)
                    younger = right
                    break
                end
                younger = right
            end
            isnan(younger) || push!(envelope, younger)
        end
        append!(policy, envelope)
    end

    policy = _coalesce_duplicate_segments(policy)
    validate_policy_collection(policy;
        context="solved policy at path $i, period $t")
    return policy
end


"""
    deterministic_ptd_path!(problem::DeterministicProblem; i::Int64)

Solve `problem`'s policy functions from period `tStart`
to the final period for the income path `i` using PTD algorithm.
"""
function deterministic_ptd_path!(problem::DeterministicProblem; i::Int64,
    tStart::Int64=1, options::FusionOptions=FusionOptions())
    1 <= i <= size(problem.incomes, 1) ||
        throw(ArgumentError("Income-path index i=$i is outside the model."))
    1 <= tStart <= problem.tMax ||
        throw(ArgumentError("tStart must be in 1:$(problem.tMax); received $tStart."))
    validate_policy_collection(problem.policy[i, problem.tMax];
        context="terminal policy at path $i")
    # Backward induction on one income path.
    for t in problem.tMax-1:-1:tStart
        # Policy children at t.
        pc = descendant_operator(problem, i=i, t=t + 1)
        # Refined policy collection.
        problem.policy[i, t] = fusion_operator(problem, pc; i, t, options)
    end
end


"""
    deterministic_ptd!(problem::DeterministicProblem; tStart::Int64=1, threshold::Int64=10)

Solve `problem`'s policy functions from period `tStart` to the final period
for all possible income paths using PTD algorithm.
If the number of income paths reaches `threshold`, multiple threads are used.
With `tStart > 1`, later dates are rebuilt from the terminal condition while
earlier stored dates are left untouched.
`options` controls finite unbounded-fusion safeguards. By default, an unresolved
tail raises `FusionTailError` with the affected model, path, and period; diagnostic
callers must explicitly opt into an approximate tail.
"""
function deterministic_ptd!(problem::DeterministicProblem; tStart::Int64=1,
    threshold::Int64=10, options::FusionOptions=FusionOptions())

    1 <= tStart <= problem.tMax ||
        throw(ArgumentError("tStart must be in 1:$(problem.tMax); received $tStart."))
    threshold >= 1 || throw(ArgumentError("Thread threshold must be positive."))
    N = size(problem.incomes)[1] # Number of income pathes.
    fill!(problem.vFloor, NaN64)

    if N ≥ threshold # Use multi-threading for large N.
        # Solve policy functions for all income paths.
        Threads.@threads for i in 1:N
            deterministic_ptd_path!(problem; i, tStart, options)
        end
    else # For small N, single thread is faster.
        # Solve policy functions for all income paths.
        for i in 1:N
            deterministic_ptd_path!(problem; i, tStart, options)
        end
    end
end
