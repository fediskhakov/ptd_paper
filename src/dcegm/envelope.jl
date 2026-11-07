using Interpolations

# 1.Grids types.

""" Abstrct type for Grids in all the models"""
abstract type Grids end

"""
1D Grids used in models with deterministic or i.i.d. income processes. 

# Fields
- `x::Vector{Float64}`: State grid which is _not_ necessarily increasing.

- `c::Vector{Float64}`: Consumption policy on `x` grid. 

- `v::Vector{Float64}` Values on `x` grid.
"""
struct Grids1d <: Grids
    x::Vector{Float64}
    c::Vector{Float64}
    v::Vector{Float64}

    function Grids1d(x::Vector{Float64}, c::Vector{Float64}, v::Vector{Float64})
        if length(x) == length(c) == length(v)
            return new(x, c, v)
        else
            throw(ArgumentError("x, c, v must have the same length."))
        end
    end
end


"""
2D Grids used in models with AR(1) income processes.

# Fields 
- `x::Vector{Float64}`: State grid which is _not_ necessarily increasing.

- `y::Vector{Float64}`: Strictly increasing persistent-income grid.

- `C::Matrix{Float64}`: Consumption policy on `x-y` grids, c[i, j] = c(x[i], y[j]).

- `V::Matrix{Float64}`: Values on `x-y` grids, v[i, j] = v(x[i], y[j]).
"""
struct Grids2d <: Grids
    x::Vector{Float64}
    y::Vector{Float64}
    C::Matrix{Float64}
    V::Matrix{Float64}

    function Grids2d(x::Vector{Float64}, y::Vector{Float64}, C::Matrix{Float64},
        V::Matrix{Float64})

        if (length(x) == size(C, 1) == size(V, 1)) &&
           (length(y) == size(C, 2) == size(V, 2))
            return new(x, y, C, V)
        else
            throw(ArgumentError("C and V must have the same shape: " *
                                "length(x) by length(y)."))
        end
    end
end


## 1.1 NaN grids.
const NaNGrids1d = Grids1d([NaN64,], [NaN64,], [NaN64,])
const NaNGrids2d = Grids2d([NaN64,], [NaN64,], [NaN64;;], [NaN64;;])
Base.isnan(g::Grids) = isnan(g.x[1])

## 1.2 Indexing and length for Grids1d.
### 1.2.1 Common methods.
Base.length(gr::Grids) = length(gr.x)

Base.lastindex(gr::Grids) = Base.lastindex(gr.x)

### 1.2.2 Grids1d methods.
"""Slicing `gr1d` by x index range `ur_x`."""
Base.getindex(gr1d::Grids1d, ur_x::UnitRange{Int64}) = Grids1d(
    gr1d.x[ur_x], gr1d.c[ur_x], gr1d.v[ur_x])

"""Zero 1D-Grids with `n` x-grids."""
Base.zeros(::Type{Grids1d}, n::Int) = Grids1d(zeros(n), zeros(n), zeros(n))

"""Update `gr1d[i]` to `xcv`, which is a vector of length 3."""
function Base.setindex!(gr1d::Grids1d, xcv::Vector{Float64}, i::Int64)
    gr1d.x[i], gr1d.c[i], gr1d.v[i] = xcv
end

### 1.2.3 Grids2d methods.
"""Slicing `gr2d` by x index range `ur_x`."""
Base.getindex(gr2d::Grids2d, ur_x::UnitRange{Int64}) = Grids2d(gr2d.x[ur_x],
    gr2d.y, gr2d.C[ur_x, :], gr2d.V[ur_x, :])

"""Zero 2D-Grids with `n_x` x-grids and `n_y` y-grids."""
function Base.zeros(::Type{Grids2d}, n_x::Int, n_y::Int)
    return Grids2d(zeros(n_x), zeros(n_y), zeros(n_x, n_y), zeros(n_x, n_y))
end

"""Get 1D-Grids from 2D-Grids by fixing y index to `i_y`."""
function Base.getindex(gr2d::Grids2d, i_y::Int64)
    return Grids1d(gr2d.x, gr2d.C[:, i_y], gr2d.V[:, i_y])
end

"""Update `gr2d.C[:, i_y]`, `gr2d.V[:, i_y]` to `gr1d.c`, `gr1d.v`."""
function Base.setindex!(gr2d::Grids2d, i_y::Int64, gr1d::Grids1d)
    @assert all(gr1d.x .== gr2d.x) "x-coordinates must be the same."
    gr2d.C[:, i_y] = gr1d.c
    gr2d.V[:, i_y] = gr1d.v
end

# 2. Upper-envelope calculation for DCEGM.

# This is a pure-Julia implementation of the secondary-envelope construction in
# Fedor Iskhakov's `polyline.m` (https://github.com/fediskhakov/dcegm): split
# loopbacks into monotone branches while repeating each turning point, then take
# the upper envelope of those branches. Unlike the old implementation, unrelated
# economic candidates remain separate and every pairwise value crossing inside a
# knot interval is considered. The latter is necessary when a third branch lies
# above the crossing of the winners at the interval endpoints.

"""Return `(branch, direction)` pairs for all maximal monotone sections."""
function _directional_branches(grids::Grids1d)
    length(grids) >= 2 || throw(ArgumentError("An envelope candidate needs at least two points."))
    all(isfinite, grids.x) || throw(ArgumentError("Envelope state grids must be finite."))
    all(isfinite, grids.c) || throw(ArgumentError("Envelope policies must be finite."))
    all(isfinite, grids.v) || throw(ArgumentError("Envelope values must be finite."))

    branches = Tuple{Grids1d,Int}[]
    first_ind = 1
    direction = 0
    n = length(grids)

    function add_branch!(lo::Int, hi::Int, dir::Int)
        hi > lo || return
        branch = grids[lo:hi]
        if dir < 0
            branch = Grids1d(reverse(branch.x), reverse(branch.c), reverse(branch.v))
        end
        all(diff(branch.x) .> 0.0) ||
            throw(ArgumentError("Each envelope branch must have a strictly increasing state grid."))
        push!(branches, (branch, dir))
    end

    for i in 1:n-1
        dx = grids.x[i+1] - grids.x[i]
        if dx == 0.0
            add_branch!(first_ind, i, direction)
            first_ind = i + 1
            direction = 0
            continue
        end

        next_direction = dx > 0.0 ? 1 : -1
        if direction == 0
            direction = next_direction
        elseif next_direction != direction
            # Repeat the turning point in the two adjacent sections, as required
            # by the secondary-envelope construction.
            add_branch!(first_ind, i, direction)
            first_ind = i
            direction = next_direction
        end
    end
    add_branch!(first_ind, n, direction)

    isempty(branches) && throw(ArgumentError("Envelope candidate has no nondegenerate branch."))
    return branches
end

"""Return all maximal monotone branches, with decreasing branches reversed."""
function _monotone_branches(grids::Grids1d)::Vector{Grids1d}
    return first.(_directional_branches(grids))
end

partition(grids::Grids1d)::Vector{Grids1d} = _monotone_branches(grids)

struct _EnvelopePiece
    branch::Int
    x0::Float64
    x1::Float64
    c0::Float64
    c1::Float64
    dc::Float64
    v0::Float64
    v1::Float64
    dv::Float64
end

function _policy(piece::_EnvelopePiece, x::Float64)
    x == piece.x0 && return piece.c0
    x == piece.x1 && return piece.c1
    return piece.c0 + piece.dc * (x - piece.x0)
end

function _value(piece::_EnvelopePiece, x::Float64)
    x == piece.x0 && return piece.v0
    x == piece.x1 && return piece.v1
    return piece.v0 + piece.dv * (x - piece.x0)
end

struct _EnvelopeInterval
    x0::Float64
    x1::Float64
    piece::Int
end

const ENVELOPE_VALUE_RTOL = 1e-12
const ENVELOPE_VALUE_ATOL = 1e-12

"""Choose the highest-value piece, saving more when values are tied."""
function _envelope_winner(active::Vector{Int}, pieces::Vector{_EnvelopePiece},
    state::Float64)::Int

    values = [_value(pieces[index], state) for index in active]
    all(value -> !isnan(value), values) || throw(ErrorException(
        "An envelope candidate evaluated to NaN at state $state."))
    best_value = maximum(values)
    tied = [local_index for local_index in eachindex(active)
            if values[local_index] == best_value ||
               isapprox(values[local_index], best_value;
                   rtol=ENVELOPE_VALUE_RTOL, atol=ENVELOPE_VALUE_ATOL)]
    saving_winner = argmin(_policy(pieces[active[local_index]], state)
                           for local_index in tied)
    return active[tied[saving_winner]]
end

function _pieces(branches::Vector{Grids1d})
    pieces = _EnvelopePiece[]
    for (branch_ind, branch) in enumerate(branches)
        for i in 1:length(branch)-1
            dx = branch.x[i+1] - branch.x[i]
            dx > 0.0 || throw(ArgumentError("Envelope branch is not strictly increasing."))
            push!(pieces, _EnvelopePiece(branch_ind, branch.x[i], branch.x[i+1],
                branch.c[i], branch.c[i+1],
                (branch.c[i+1] - branch.c[i]) / dx,
                branch.v[i], branch.v[i+1],
                (branch.v[i+1] - branch.v[i]) / dx))
        end
    end
    return pieces
end

function _crossings!(cuts::Vector{Float64}, active::Vector{Int},
    pieces::Vector{_EnvelopePiece}, lo::Float64, hi::Float64)

    for i in 1:length(active)-1, j in i+1:length(active)
        p, q = pieces[active[i]], pieces[active[j]]
        slope_diff = p.dv - q.dv
        slope_diff == 0.0 && continue
        root = lo - (_value(p, lo) - _value(q, lo)) / slope_diff
        lo < root < hi && isfinite(root) && push!(cuts, root)
    end
    sort!(unique!(cuts))
    return cuts
end

function _envelope_intervals(pieces::Vector{_EnvelopePiece})
    knots = sort!(unique!(vcat([Float64[p.x0, p.x1] for p in pieces]...)))
    intervals = _EnvelopeInterval[]

    for knot_ind in 1:length(knots)-1
        lo, hi = knots[knot_ind], knots[knot_ind+1]
        lo < hi || continue
        active = findall(p -> p.x0 <= lo && hi <= p.x1, pieces)
        isempty(active) && throw(ArgumentError(
            "Envelope candidates leave an uncovered interval [$lo, $hi]."))

        cuts = _crossings!(Float64[lo, hi], active, pieces, lo, hi)
        for cut_ind in 1:length(cuts)-1
            left, right = cuts[cut_ind], cuts[cut_ind+1]
            left < right || continue
            midpoint = left + (right - left) / 2
            winner = _envelope_winner(active, pieces, midpoint)

            # Do not merge across structural knots. A candidate can touch the
            # upper envelope at exactly such a knot while losing on both open
            # neighboring intervals; that isolated welfare tie can carry a
            # different, saving-intensive policy.
            push!(intervals, _EnvelopeInterval(left, right, winner))
        end
    end
    return intervals
end

function _push_values!(x::Vector{Float64}, c::Vector{Float64},
    v::Vector{Float64}, state::Float64, policy::Float64, value::Float64)

    if !isempty(x) && state == x[end]
        if value > v[end] && !isapprox(value, v[end];
            rtol=ENVELOPE_VALUE_RTOL, atol=ENVELOPE_VALUE_ATOL)
            c[end], v[end] = policy, value
        elseif isapprox(value, v[end]; rtol=ENVELOPE_VALUE_RTOL,
            atol=ENVELOPE_VALUE_ATOL) && policy < c[end]
            # Equal welfare selects lower consumption, i.e. more saving.
            c[end], v[end] = policy, max(value, v[end])
        end
    elseif isempty(x) || state > x[end]
        push!(x, state)
        push!(c, policy)
        push!(v, value)
    end
    return nothing
end

function _push_point!(x::Vector{Float64}, c::Vector{Float64}, v::Vector{Float64},
    state::Float64, piece::_EnvelopePiece)

    return _push_values!(x, c, v, state, _policy(piece, state),
        _value(piece, state))
end

function _exact_envelope_choice(pieces::Vector{_EnvelopePiece}, state::Float64)
    active = findall(piece -> piece.x0 <= state <= piece.x1, pieces)
    isempty(active) && throw(ErrorException(
        "No envelope candidate is defined at structural state $state."))
    winner = _envelope_winner(active, pieces, state)
    best_value = maximum(_value(pieces[index], state) for index in active)
    return _policy(pieces[winner], state), best_value
end

function _same_envelope_choice(policy_a::Float64, value_a::Float64,
    policy_b::Float64, value_b::Float64)

    return isapprox(policy_a, policy_b; rtol=1e-10, atol=1e-12) &&
           isapprox(value_a, value_b; rtol=ENVELOPE_VALUE_RTOL,
               atol=ENVELOPE_VALUE_ATOL)
end

"""
    upper_envelope(candidates::AbstractVector{<:Grids1d}) -> Grids1d

Return the piecewise-linear upper value envelope of separate DCEGM candidate
curves. Loopbacks within a candidate are split into monotone branches. Policy
discontinuities at continuous value crossings are represented by the old branch
at `prevfloat(x*)`. At the crossing itself, equal values select lower
consumption (more saving).
"""
function upper_envelope(candidates::AbstractVector{<:Grids1d})::Grids1d
    isempty(candidates) && throw(ArgumentError("At least one envelope candidate is required."))
    branches = reduce(vcat, (_monotone_branches(candidate) for candidate in candidates))
    pieces = _pieces(branches)
    intervals = _envelope_intervals(pieces)
    isempty(intervals) && throw(ArgumentError("The envelope has no nondegenerate interval."))

    x, c, v = Float64[], Float64[], Float64[]
    for (i, interval) in enumerate(intervals)
        piece = pieces[interval.piece]
        left, right = interval.x0, interval.x1

        if i == 1
            exact_policy, exact_value = _exact_envelope_choice(pieces, left)
            next_policy, next_value = _policy(piece, left), _value(piece, left)
            if !_same_envelope_choice(exact_policy, exact_value,
                next_policy, next_value)
                _push_values!(x, c, v, left, exact_policy, exact_value)
                right_limit = nextfloat(left)
                right_limit < right &&
                    _push_point!(x, c, v, right_limit, piece)
            end
        else
            previous = pieces[intervals[i-1].piece]
            exact_policy, exact_value = _exact_envelope_choice(pieces, left)
            previous_policy, previous_value =
                _policy(previous, left), _value(previous, left)
            next_policy, next_value = _policy(piece, left), _value(piece, left)
            differs_left = !_same_envelope_choice(exact_policy, exact_value,
                previous_policy, previous_value)
            differs_right = !_same_envelope_choice(exact_policy, exact_value,
                next_policy, next_value)

            if differs_left || differs_right
                old_right = prevfloat(left)
                if old_right > intervals[i-1].x0
                    # The previous endpoint was appended on its earlier pass;
                    # replace it with the left-limit point.
                    if !isempty(x) && x[end] == left
                        pop!(x); pop!(c); pop!(v)
                    end
                    _push_point!(x, c, v, old_right, previous)
                end
                # Evaluate every candidate defined at the structural knot, not
                # only the open-interval winners. This retains an isolated
                # save-more tie (or isolated value spike) exactly at `left`.
                _push_values!(x, c, v, left, exact_policy, exact_value)
                right_limit = nextfloat(left)
                if differs_right && right_limit < right
                    # Return immediately to the open-interval winner so the
                    # exact knot choice is not linearly smeared to the right.
                    _push_point!(x, c, v, right_limit, piece)
                end
            end
        end

        _push_point!(x, c, v, left, piece)
        if i == length(intervals)
            exact_policy, exact_value = _exact_envelope_choice(pieces, right)
            previous_policy, previous_value =
                _policy(piece, right), _value(piece, right)
            if !_same_envelope_choice(exact_policy, exact_value,
                previous_policy, previous_value)
                left_limit = prevfloat(right)
                left_limit > left &&
                    _push_point!(x, c, v, left_limit, piece)
                _push_values!(x, c, v, right, exact_policy, exact_value)
            else
                _push_point!(x, c, v, right, piece)
            end
        else
            _push_point!(x, c, v, right, piece)
        end
    end

    length(x) >= 2 && all(diff(x) .> 0.0) ||
        throw(ErrorException("Upper-envelope output is not strictly ordered."))
    return Grids1d(x, c, v)
end

upper_envelope(grids::Grids1d)::Grids1d = upper_envelope(Grids1d[grids])
