export TetraEnvelopeOptions, TetraEnvelopeAudit, TetraEnvelopeResult,
       barycentric_weights_3d, project_tetra_envelope, tensor_cell_tetrahedra

"""Numerical tolerances for three-dimensional tetrahedral projection."""
Base.@kwdef struct TetraEnvelopeOptions
    index_padding::Int = 1
    minimum_weight::Float64 = -0.20
    determinant_tolerance::Float64 = 1.0e-13
    value_atol::Float64 = 1.0e-11
    value_rtol::Float64 = 1.0e-10
end

"""Mutable audit counters for tetrahedral envelope construction."""
Base.@kwdef mutable struct TetraEnvelopeAudit
    cells::Int = 0
    tetrahedra_considered::Int = 0
    tetrahedra_degenerate::Int = 0
    tetrahedra_processed::Int = 0
    common_nodes_examined::Int = 0
    barycentric_rejections::Int = 0
    infeasible_candidates::Int = 0
    finite_candidates::Int = 0
    envelope_updates::Int = 0
    extrapolated_candidates::Int = 0
    extrapolated_winners::Int = 0
    minimum_winning_weight::Float64 = Inf
end

"""Consumption, value, coverage, and audit output of tetrahedral projection."""
struct TetraEnvelopeResult
    consumption::Array{Float64,3}
    value::Array{Float64,3}
    coverage::BitArray{3}
    winner_minimum_weight::Array{Float64,3}
    audit::TetraEnvelopeAudit
end

"""Six tetrahedra covering tensor cell `(i,j,k)` around diagonal 000--111."""
function tensor_cell_tetrahedra(i::Integer, j::Integer, k::Integer)
    i >= 1 && j >= 1 && k >= 1 || throw(ArgumentError(
        "tensor-cell indices must be positive"))
    p000 = CartesianIndex(i, j, k)
    p100 = CartesianIndex(i + 1, j, k)
    p010 = CartesianIndex(i, j + 1, k)
    p001 = CartesianIndex(i, j, k + 1)
    p110 = CartesianIndex(i + 1, j + 1, k)
    p101 = CartesianIndex(i + 1, j, k + 1)
    p011 = CartesianIndex(i, j + 1, k + 1)
    p111 = CartesianIndex(i + 1, j + 1, k + 1)
    return ((p000, p100, p110, p111),
            (p000, p100, p101, p111),
            (p000, p010, p110, p111),
            (p000, p010, p011, p111),
            (p000, p001, p101, p111),
            (p000, p001, p011, p111))
end

_det3(a, b, c) =
    a[1] * (b[2] * c[3] - b[3] * c[2]) -
    b[1] * (a[2] * c[3] - a[3] * c[2]) +
    c[1] * (a[2] * b[3] - a[3] * b[2])

_sub3(a, b) = (a[1] - b[1], a[2] - b[2], a[3] - b[3])
_norm3(a) = sqrt(a[1]^2 + a[2]^2 + a[3]^2)

"""Barycentric weights of a point in a mapped tetrahedron, or `nothing`."""
function barycentric_weights_3d(vertices::NTuple{4,NTuple{3,<:Real}},
    target::NTuple{3,<:Real}; determinant_tolerance::Real=1.0e-13)
    points = map(v -> Float64.(v), vertices)
    p = Float64.(target)
    all(v -> all(isfinite, v), points) && all(isfinite, p) ||
        throw(ArgumentError("tetrahedron coordinates must be finite"))
    a, b, c = _sub3(points[1], points[4]),
              _sub3(points[2], points[4]),
              _sub3(points[3], points[4])
    rhs = _sub3(p, points[4])
    determinant = _det3(a, b, c)
    tolerance = Float64(determinant_tolerance) *
        max(_norm3(a) * _norm3(b) * _norm3(c), 1.0)
    abs(determinant) <= tolerance && return nothing
    w1 = _det3(rhs, b, c) / determinant
    w2 = _det3(a, rhs, c) / determinant
    w3 = _det3(a, b, rhs) / determinant
    return (w1, w2, w3, 1.0 - w1 - w2 - w3)
end

function _span(grid, lower, upper, padding)
    first_index = max(1, searchsortedfirst(grid, lower) - padding)
    last_index = min(length(grid), searchsortedlast(grid, upper) + padding)
    return first_index:last_index
end

function _prefer(candidate_v, candidate_c, incumbent_v, incumbent_c,
    options::TetraEnvelopeOptions)
    isfinite(candidate_v) || return false
    isfinite(incumbent_v) || return true
    tolerance = options.value_atol + options.value_rtol *
        max(abs(candidate_v), abs(incumbent_v), 1.0)
    return candidate_v > incumbent_v + tolerance ||
        (abs(candidate_v - incumbent_v) <= tolerance &&
         candidate_c < incumbent_c)
end

"""
Project a mapped three-dimensional endogenous tensor grid to a declared common
grid. Candidate values are always recomputed by `value_at`; interpolation of
vertex values is never used for the upper-envelope comparison.
"""
function project_tetra_envelope(mapped_x::Array{Float64,3},
    mapped_e::Array{Float64,3}, mapped_m::Array{Float64,3},
    node_consumption::Array{Float64,3}, common_x::Vector{Float64},
    common_e::Vector{Float64}, common_m::Vector{Float64}; feasible, value_at,
    valid_vertices::BitArray{3}=trues(size(mapped_x)),
    options::TetraEnvelopeOptions=TetraEnvelopeOptions())

    size(mapped_x) == size(mapped_e) == size(mapped_m) ==
        size(node_consumption) == size(valid_vertices) ||
        throw(DimensionMismatch("mapped tetrahedron arrays must have one shape"))
    options.index_padding >= 0 || throw(ArgumentError(
        "tetrahedron index padding must be nonnegative"))
    -1.0 <= options.minimum_weight <= 0.0 || throw(ArgumentError(
        "minimum tetrahedron weight must lie in [-1,0]"))
    shape = (length(common_x), length(common_e), length(common_m))
    consumption = fill(NaN, shape)
    value = fill(-Inf, shape)
    coverage = falses(shape)
    winner_weight = fill(Inf, shape)
    audit = TetraEnvelopeAudit()
    tetra_id = 0

    @inbounds for k in 1:size(mapped_x, 3)-1,
                  j in 1:size(mapped_x, 2)-1,
                  i in 1:size(mapped_x, 1)-1
        audit.cells += 1
        for tetrahedron in tensor_cell_tetrahedra(i, j, k)
            tetra_id += 1
            audit.tetrahedra_considered += 1
            all(valid_vertices[index] for index in tetrahedron) || continue
            vertices = ntuple(q -> begin
                index = tetrahedron[q]
                (mapped_x[index], mapped_e[index], mapped_m[index])
            end, 4)
            weights_at_centroid = barycentric_weights_3d(vertices,
                ntuple(d -> sum(vertices[q][d] for q in 1:4) / 4.0, 3);
                determinant_tolerance=options.determinant_tolerance)
            if isnothing(weights_at_centroid)
                audit.tetrahedra_degenerate += 1
                continue
            end
            audit.tetrahedra_processed += 1
            xs = _span(common_x, minimum(v[1] for v in vertices),
                maximum(v[1] for v in vertices), options.index_padding)
            es = _span(common_e, minimum(v[2] for v in vertices),
                maximum(v[2] for v in vertices), options.index_padding)
            ms = _span(common_m, minimum(v[3] for v in vertices),
                maximum(v[3] for v in vertices), options.index_padding)
            (isempty(xs) || isempty(es) || isempty(ms)) && continue
            for ck in ms, cj in es, ci in xs
                audit.common_nodes_examined += 1
                target = (common_x[ci], common_e[cj], common_m[ck])
                weights = barycentric_weights_3d(vertices, target;
                    determinant_tolerance=options.determinant_tolerance)
                isnothing(weights) && continue
                minimum_weight = minimum(weights)
                if minimum_weight < options.minimum_weight ||
                   maximum(weights) > 1.0 - 3.0 * options.minimum_weight
                    audit.barycentric_rejections += 1
                    continue
                end
                minimum_weight < 0.0 && (audit.extrapolated_candidates += 1)
                candidate_c = sum(weights[q] *
                    node_consumption[tetrahedron[q]] for q in 1:4)
                if !isfinite(candidate_c) ||
                   !feasible(target[1], target[2], target[3], candidate_c)
                    audit.infeasible_candidates += 1
                    continue
                end
                candidate_v = Float64(value_at(
                    target[1], target[2], target[3], candidate_c))
                isfinite(candidate_v) || continue
                audit.finite_candidates += 1
                index = CartesianIndex(ci, cj, ck)
                if _prefer(candidate_v, candidate_c,
                    value[index], consumption[index], options)
                    consumption[index] = candidate_c
                    value[index] = candidate_v
                    coverage[index] = true
                    winner_weight[index] = minimum_weight
                    audit.envelope_updates += 1
                end
            end
        end
    end
    winners = winner_weight[coverage]
    if !isempty(winners)
        audit.minimum_winning_weight = minimum(winners)
        audit.extrapolated_winners = count(weight -> weight < 0.0, winners)
    end
    return TetraEnvelopeResult(
        consumption, value, coverage, winner_weight, audit)
end
