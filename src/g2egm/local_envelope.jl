export EnvelopeOptions,
       EnvelopeAudit,
       EnvelopeResult,
       HoleFillRecord,
       barycentric_weights,
       padded_common_bbox,
       project_local_envelope,
       tensor_cell_triangles,
       fill_local_holes!

"""
Numerical controls for the local-simplex/common-grid envelope.

`index_padding` expands a mapped triangle's common-grid bounding box by the
declared number of indices in each direction. `minimum_weight` bounds limited
barycentric extrapolation. A triangle is rejected when the absolute value of
its determinant is no larger than
`degenerate_atol + degenerate_rtol * ||e1|| * ||e2||`.
"""
Base.@kwdef struct EnvelopeOptions
    index_padding::Int = 2
    minimum_weight::Float64 = -0.25
    degenerate_atol::Float64 = 0.0
    degenerate_rtol::Float64 = 64eps(Float64)
    value_atol::Float64 = 1.0e-12
    value_rtol::Float64 = 1.0e-12
    hole_radius::Int = 2
    fill_holes::Bool = true
end

"""Counters making every envelope and hole-repair decision auditable."""
mutable struct EnvelopeAudit
    tensor_cells::Int
    triangles_total::Int
    triangles_invalid::Int
    triangles_regime_rejected::Int
    triangles_degenerate::Int
    triangles_processed::Int
    common_nodes_examined::Int
    barycentric_rejected::Int
    negative_barycentric_candidates::Int
    infeasible_candidates::Int
    nonfinite_value_candidates::Int
    finite_candidates::Int
    envelope_updates::Int
    negative_barycentric_winner_updates::Int
    save_more_ties::Int
    direct_holes_before_fill::Int
    hole_neighbor_candidates_checked::Int
    holes_filled::Int
    holes_remaining::Int
    winners_with_negative_barycentric_weight::Int
    minimum_barycentric_weight_among_winners::Float64
end

EnvelopeAudit() = EnvelopeAudit(ntuple(_ -> 0, 20)..., Inf)

"""One deterministic, value-recomputed repair of a common-grid hole."""
struct HoleFillRecord
    target::CartesianIndex{2}
    source::CartesianIndex{2}
    squared_index_distance::Int
    neighbor_candidates_checked::Int
    consumption::Float64
    recomputed_value::Float64
    source_minimum_barycentric_weight::Float64
    source_triangle::Int
end

"""Result of projecting one segment's endogenous tensor grid."""
struct EnvelopeResult
    consumption::Matrix{Float64}
    value::Matrix{Float64}
    direct_coverage::BitMatrix
    coverage::BitMatrix
    candidate_count::Matrix{Int}
    winner_minimum_barycentric_weight::Matrix{Float64}
    winner_triangle::Matrix{Int}
    fill_target::BitMatrix
    hole_fills::Vector{HoleFillRecord}
    audit::EnvelopeAudit
end

function _validate_options(options::EnvelopeOptions)
    options.index_padding >= 0 || throw(ArgumentError(
        "Envelope index padding must be nonnegative."))
    isfinite(options.minimum_weight) && options.minimum_weight <= 1 / 3 ||
        throw(ArgumentError(
            "The minimum barycentric weight must be finite and no larger than one third."))
    all(isfinite, (options.degenerate_atol, options.degenerate_rtol,
                   options.value_atol, options.value_rtol)) ||
        throw(ArgumentError("Envelope tolerances must be finite."))
    all(>=(0.0), (options.degenerate_atol, options.degenerate_rtol,
                  options.value_atol, options.value_rtol)) ||
        throw(ArgumentError("Envelope tolerances must be nonnegative."))
    options.hole_radius >= 0 || throw(ArgumentError(
        "The local hole-fill radius must be nonnegative."))
    return options
end

function _validate_common_grid(grid::AbstractVector, name::String)
    !isempty(grid) || throw(ArgumentError("The common $name grid cannot be empty."))
    all(isfinite, grid) || throw(ArgumentError(
        "The common $name grid must be finite."))
    previous = first(grid)
    for current in Iterators.drop(grid, 1)
        current - previous > 0 || throw(ArgumentError(
            "The common $name grid must be strictly increasing."))
        previous = current
    end
    return nothing
end

"""
Return the lower-left and upper-right triangles of tensor cell `(i, j)`.
"""
function tensor_cell_triangles(i::Integer, j::Integer)
    i >= 1 && j >= 1 || throw(ArgumentError(
        "Tensor-cell indices must be positive."))
    lower_left = (CartesianIndex(i, j), CartesianIndex(i, j + 1),
                  CartesianIndex(i + 1, j))
    upper_right = (CartesianIndex(i + 1, j + 1),
                   CartesianIndex(i + 1, j), CartesianIndex(i, j + 1))
    return (lower_left, upper_right)
end

"""
Compute barycentric weights of `(x, y)` in a mapped triangle.

Returns `nothing` for a numerically degenerate triangle.
"""
function barycentric_weights(vertices::NTuple{3,NTuple{2,<:Real}},
    x::Real, y::Real; degenerate_atol::Real=0.0,
    degenerate_rtol::Real=64eps(Float64))

    atol, rtol = Float64(degenerate_atol), Float64(degenerate_rtol)
    all(isfinite, (atol, rtol)) && atol >= 0.0 && rtol >= 0.0 ||
        throw(ArgumentError("Degeneracy tolerances must be finite and nonnegative."))
    points = map(vertices) do vertex
        (Float64(vertex[1]), Float64(vertex[2]))
    end
    all(point -> all(isfinite, point), points) || throw(ArgumentError(
        "Triangle vertices must be finite."))
    target_x, target_y = Float64(x), Float64(y)
    all(isfinite, (target_x, target_y)) || throw(ArgumentError(
        "The barycentric target must be finite."))

    (x1, y1), (x2, y2), (x3, y3) = points
    edge1 = (x1 - x3, y1 - y3)
    edge2 = (x2 - x3, y2 - y3)
    determinant = edge1[1] * edge2[2] - edge2[1] * edge1[2]
    scale = hypot(edge1...) * hypot(edge2...)
    abs(determinant) <= atol + rtol * scale && return nothing

    weight1 = ((y2 - y3) * (target_x - x3) +
               (x3 - x2) * (target_y - y3)) / determinant
    weight2 = ((y3 - y1) * (target_x - x3) +
               (x1 - x3) * (target_y - y3)) / determinant
    weight3 = 1.0 - weight1 - weight2
    return (weight1, weight2, weight3)
end

function _padded_span(grid::AbstractVector{<:Real}, lower::Float64,
    upper::Float64, padding::Int)

    lower <= upper || throw(ArgumentError("A bounding interval must be ordered."))
    padding >= 0 || throw(ArgumentError("Bounding-box padding must be nonnegative."))
    first_index = searchsortedfirst(grid, lower) - padding
    last_index = searchsortedlast(grid, upper) + padding
    first_index = max(first_index, 1)
    last_index = min(last_index, length(grid))
    return first_index:last_index
end

"""Return padded common-grid index ranges covering `vertices`."""
function padded_common_bbox(common_x::AbstractVector{<:Real},
    common_y::AbstractVector{<:Real},
    vertices::NTuple{3,NTuple{2,<:Real}}; index_padding::Integer=2)

    _validate_common_grid(common_x, "first-state")
    _validate_common_grid(common_y, "second-state")
    padding = Int(index_padding)
    padding >= 0 || throw(ArgumentError(
        "Envelope index padding must be nonnegative."))
    all(vertex -> all(isfinite, vertex), vertices) || throw(ArgumentError(
        "Triangle vertices must be finite."))
    x_lower, x_upper = extrema(vertex[1] for vertex in vertices)
    y_lower, y_upper = extrema(vertex[2] for vertex in vertices)
    return (_padded_span(common_x, Float64(x_lower), Float64(x_upper), padding),
            _padded_span(common_y, Float64(y_lower), Float64(y_upper), padding))
end

function _same_regime(regimes, triangle)
    isnothing(regimes) && return true
    first_signature = regimes[triangle[1]]
    return isequal(regimes[triangle[2]], first_signature) &&
           isequal(regimes[triangle[3]], first_signature)
end

function _valid_triangle(mapped_x, mapped_y, consumption, valid_vertices,
    triangle)

    return all(triangle) do index
        (isnothing(valid_vertices) || valid_vertices[index]) &&
            isfinite(mapped_x[index]) && isfinite(mapped_y[index]) &&
            isfinite(consumption[index])
    end
end

function _value_tolerance(options::EnvelopeOptions,
    candidate::Float64, incumbent::Float64)

    return options.value_atol + options.value_rtol *
        max(abs(candidate), abs(incumbent), 1.0)
end

"""
Return `(replace, save_more_tie)` under the deterministic envelope rule.

Within the declared numerical value tolerance, lower consumption (more saving
at a common cash state) wins.
"""
function _prefer_candidate(candidate_value::Float64,
    candidate_consumption::Float64, incumbent_value::Float64,
    incumbent_consumption::Float64, options::EnvelopeOptions)

    tolerance = _value_tolerance(options, candidate_value, incumbent_value)
    if candidate_value > incumbent_value + tolerance
        return true, false
    elseif abs(candidate_value - incumbent_value) <= tolerance
        if candidate_consumption < incumbent_consumption
            return true, true
        elseif candidate_consumption > incumbent_consumption
            return false, true
        end
    end
    return false, false
end

function _callback_value(value_at, x::Float64, y::Float64,
    consumption::Float64)

    value = value_at(x, y, consumption)
    value isa Real || throw(ArgumentError(
        "The value callback must return a real scalar."))
    return Float64(value)
end

function _callback_feasible(feasible, x::Float64, y::Float64,
    consumption::Float64)

    result = feasible(x, y, consumption)
    result isa Bool || throw(ArgumentError(
        "The feasibility callback must return Bool."))
    return result
end

function _bounding_fill_target(direct_coverage::BitMatrix)
    target = falses(size(direct_coverage))
    covered_indices = findall(direct_coverage)
    isempty(covered_indices) && return target
    coordinates = Tuple.(covered_indices)
    first_indices = first.(coordinates)
    second_indices = last.(coordinates)
    target[minimum(first_indices):maximum(first_indices),
           minimum(second_indices):maximum(second_indices)] .= true
    return target
end

"""
Fill common-grid holes from directly covered nodes within a fixed index radius,
always recomputing feasibility and value at the target state.

Newly filled nodes are never used as sources, so the result is independent of
scan order. Every feasible nearby choice is evaluated at the hole and
upper-enveloped; the same save-more tie
rule makes exact and numerical welfare ties deterministic. The function
mutates `consumption` and `value` and returns coverage, target mask, and an
explicit record for each fill.
"""
function fill_local_holes!(consumption::Matrix{Float64},
    value::Matrix{Float64}, direct_coverage::BitMatrix,
    common_x::AbstractVector{<:Real}, common_y::AbstractVector{<:Real};
    feasible, value_at, options::EnvelopeOptions=EnvelopeOptions(),
    fill_target::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    winner_minimum_barycentric_weight::Union{Nothing,Matrix{Float64}}=nothing,
    winner_triangle::Union{Nothing,Matrix{Int}}=nothing,
    audit::EnvelopeAudit=EnvelopeAudit())

    _validate_options(options)
    size(consumption) == size(value) == size(direct_coverage) ||
        throw(DimensionMismatch(
            "Consumption, value, and direct coverage must have the same shape."))
    size(consumption) == (length(common_x), length(common_y)) ||
        throw(DimensionMismatch(
            "Common grids do not match the envelope matrix dimensions."))
    isnothing(winner_minimum_barycentric_weight) ||
        size(winner_minimum_barycentric_weight) == size(consumption) ||
        throw(DimensionMismatch(
            "The winner-weight provenance matrix has the wrong shape."))
    isnothing(winner_triangle) || size(winner_triangle) == size(consumption) ||
        throw(DimensionMismatch(
            "The winner-triangle provenance matrix has the wrong shape."))
    isnothing(winner_minimum_barycentric_weight) == isnothing(winner_triangle) ||
        throw(ArgumentError(
            "Winner-weight and winner-triangle provenance must be supplied together."))
    _validate_common_grid(common_x, "first-state")
    _validate_common_grid(common_y, "second-state")

    target = isnothing(fill_target) ? _bounding_fill_target(direct_coverage) :
        BitMatrix(fill_target)
    size(target) == size(direct_coverage) || throw(DimensionMismatch(
        "The hole-fill target mask has the wrong shape."))
    coverage = copy(direct_coverage)
    records = HoleFillRecord[]
    holes = findall(target .& .!direct_coverage)
    audit.direct_holes_before_fill += length(holes)
    radius = options.hole_radius

    for target_index in holes
        target_i, target_j = Tuple(target_index)
        sources = Tuple{Int,CartesianIndex{2}}[]
        for source_j in max(1, target_j - radius):min(size(coverage, 2), target_j + radius)
            for source_i in max(1, target_i - radius):min(size(coverage, 1), target_i + radius)
                source = CartesianIndex(source_i, source_j)
                direct_coverage[source] || continue
                distance = (source_i - target_i)^2 + (source_j - target_j)^2
                push!(sources, (distance, source))
            end
        end
        sort!(sources; by=source -> (source[1], Tuple(source[2])...))

        x, y = Float64(common_x[target_i]), Float64(common_y[target_j])
        checked = 0
        selected_source = nothing
        selected_distance = 0
        selected_consumption = NaN
        selected_value = -Inf
        for (distance, source) in sources
            candidate_consumption = consumption[source]
            checked += 1
            audit.hole_neighbor_candidates_checked += 1
            _callback_feasible(feasible, x, y, candidate_consumption) || continue
            candidate_value = _callback_value(
                value_at, x, y, candidate_consumption)
            isfinite(candidate_value) || continue
            if isnothing(selected_source)
                selected_source = source
                selected_distance = distance
                selected_consumption = candidate_consumption
                selected_value = candidate_value
            else
                replace, save_more_tie = _prefer_candidate(
                    candidate_value, candidate_consumption,
                    selected_value, selected_consumption, options)
                save_more_tie && (audit.save_more_ties += 1)
                if replace
                    selected_source = source
                    selected_distance = distance
                    selected_consumption = candidate_consumption
                    selected_value = candidate_value
                end
            end
        end

        if !isnothing(selected_source)
            consumption[target_index] = selected_consumption
            value[target_index] = selected_value
            coverage[target_index] = true
            source_minimum_weight =
                isnothing(winner_minimum_barycentric_weight) ? Inf :
                winner_minimum_barycentric_weight[selected_source]
            source_triangle = isnothing(winner_triangle) ? 0 :
                winner_triangle[selected_source]
            if !isnothing(winner_minimum_barycentric_weight)
                winner_minimum_barycentric_weight[target_index] =
                    source_minimum_weight
                winner_triangle[target_index] = source_triangle
            end
            audit.holes_filled += 1
            push!(records, HoleFillRecord(target_index, selected_source,
                selected_distance, checked, selected_consumption,
                selected_value, source_minimum_weight, source_triangle))
        end
    end
    audit.holes_remaining += count(target .& .!coverage)
    return (coverage=coverage, fill_target=target, records=records)
end

"""
Project an endogenous tensor-node surface to a regular common grid.

Each tensor cell contributes its two local triangles. Candidate consumption is
barycentrically interpolated, but candidate value is *always* obtained from
`value_at(common_x, common_y, candidate_consumption)`. `feasible` is called
before `value_at`. Optional `regimes` must match the node-array shape; a
triangle whose three vertex signatures are not identical is rejected rather
than interpolated across a structural kink.
"""
function project_local_envelope(mapped_x::AbstractMatrix{<:Real},
    mapped_y::AbstractMatrix{<:Real},
    node_consumption::AbstractMatrix{<:Real},
    common_x::AbstractVector{<:Real}, common_y::AbstractVector{<:Real};
    feasible, value_at,
    valid_vertices::Union{Nothing,AbstractMatrix{Bool}}=nothing,
    regimes::Union{Nothing,AbstractMatrix}=nothing,
    options::EnvelopeOptions=EnvelopeOptions(),
    fill_target::Union{Nothing,AbstractMatrix{Bool}}=nothing)

    _validate_options(options)
    size(mapped_x) == size(mapped_y) == size(node_consumption) ||
        throw(DimensionMismatch(
            "Mapped states and node consumption must have the same shape."))
    size(mapped_x, 1) >= 2 && size(mapped_x, 2) >= 2 ||
        throw(ArgumentError(
            "The endogenous tensor surface needs at least two nodes per dimension."))
    isnothing(valid_vertices) || size(valid_vertices) == size(mapped_x) ||
        throw(DimensionMismatch("The vertex-validity mask has the wrong shape."))
    isnothing(regimes) || size(regimes) == size(mapped_x) ||
        throw(DimensionMismatch("The regime-signature array has the wrong shape."))
    _validate_common_grid(common_x, "first-state")
    _validate_common_grid(common_y, "second-state")

    common_shape = (length(common_x), length(common_y))
    consumption = fill(NaN, common_shape)
    value = fill(-Inf, common_shape)
    direct_coverage = falses(common_shape)
    candidate_count = zeros(Int, common_shape)
    winner_minimum_weight = fill(Inf, common_shape)
    winner_triangle = zeros(Int, common_shape)
    audit = EnvelopeAudit()

    for j in 1:size(mapped_x, 2)-1, i in 1:size(mapped_x, 1)-1
        audit.tensor_cells += 1
        for triangle in tensor_cell_triangles(i, j)
            audit.triangles_total += 1
            if !_valid_triangle(mapped_x, mapped_y, node_consumption,
                                valid_vertices, triangle)
                audit.triangles_invalid += 1
                continue
            end
            if !_same_regime(regimes, triangle)
                audit.triangles_regime_rejected += 1
                continue
            end
            vertices = ntuple(3) do corner
                index = triangle[corner]
                (Float64(mapped_x[index]), Float64(mapped_y[index]))
            end
            probe = barycentric_weights(vertices, vertices[1]...;
                degenerate_atol=options.degenerate_atol,
                degenerate_rtol=options.degenerate_rtol)
            if isnothing(probe)
                audit.triangles_degenerate += 1
                continue
            end
            audit.triangles_processed += 1
            x_indices, y_indices = padded_common_bbox(
                common_x, common_y, vertices;
                index_padding=options.index_padding)
            for common_j in y_indices, common_i in x_indices
                audit.common_nodes_examined += 1
                x = Float64(common_x[common_i])
                y = Float64(common_y[common_j])
                weights = barycentric_weights(vertices, x, y;
                    degenerate_atol=options.degenerate_atol,
                    degenerate_rtol=options.degenerate_rtol)::NTuple{3,Float64}
                if any(weight -> weight < options.minimum_weight, weights)
                    audit.barycentric_rejected += 1
                    continue
                end
                minimum_weight = minimum(weights)
                has_negative_weight = minimum_weight < 0.0
                has_negative_weight &&
                    (audit.negative_barycentric_candidates += 1)
                candidate_consumption = sum(weights[corner] *
                    Float64(node_consumption[triangle[corner]]) for corner in 1:3)
                if !_callback_feasible(
                    feasible, x, y, candidate_consumption)
                    audit.infeasible_candidates += 1
                    continue
                end
                candidate_value = _callback_value(
                    value_at, x, y, candidate_consumption)
                if !isfinite(candidate_value)
                    audit.nonfinite_value_candidates += 1
                    continue
                end
                audit.finite_candidates += 1
                candidate_count[common_i, common_j] += 1
                index = CartesianIndex(common_i, common_j)
                if !direct_coverage[index]
                    consumption[index] = candidate_consumption
                    value[index] = candidate_value
                    direct_coverage[index] = true
                    winner_minimum_weight[index] = minimum_weight
                    winner_triangle[index] = audit.triangles_total
                    audit.envelope_updates += 1
                    has_negative_weight &&
                        (audit.negative_barycentric_winner_updates += 1)
                else
                    replace, save_more_tie = _prefer_candidate(
                        candidate_value, candidate_consumption,
                        value[index], consumption[index], options)
                    save_more_tie && (audit.save_more_ties += 1)
                    if replace
                        consumption[index] = candidate_consumption
                        value[index] = candidate_value
                        winner_minimum_weight[index] = minimum_weight
                        winner_triangle[index] = audit.triangles_total
                        audit.envelope_updates += 1
                        has_negative_weight &&
                            (audit.negative_barycentric_winner_updates += 1)
                    end
                end
            end
        end
    end

    if options.fill_holes
        filled = fill_local_holes!(consumption, value, direct_coverage,
            common_x, common_y; feasible, value_at, options, fill_target,
            winner_minimum_barycentric_weight=winner_minimum_weight,
            winner_triangle, audit)
        coverage = filled.coverage
        target = filled.fill_target
        records = filled.records
    else
        coverage = copy(direct_coverage)
        target = isnothing(fill_target) ?
            _bounding_fill_target(direct_coverage) : BitMatrix(fill_target)
        size(target) == size(direct_coverage) || throw(DimensionMismatch(
            "The hole-fill target mask has the wrong shape."))
        records = HoleFillRecord[]
        audit.direct_holes_before_fill += count(target .& .!direct_coverage)
        audit.holes_remaining += audit.direct_holes_before_fill
    end

    winner_weights = winner_minimum_weight[coverage]
    audit.winners_with_negative_barycentric_weight =
        count(<(0.0), winner_weights)
    audit.minimum_barycentric_weight_among_winners =
        isempty(winner_weights) ? Inf : minimum(winner_weights)

    return EnvelopeResult(consumption, value, direct_coverage, coverage,
        candidate_count, winner_minimum_weight, winner_triangle,
        target, records, audit)
end
