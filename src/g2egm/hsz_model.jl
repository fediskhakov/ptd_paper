"""Declared post-decision and common grids for the three-state HSZ solve."""
struct HSZG2EGMGrid
    asset::Vector{Float64}
    post_log_earnings::Vector{Float64}
    post_log_medical::Vector{Float64}
    cash::Vector{Float64}
    log_earnings::Vector{Float64}
    log_medical::Vector{Float64}

    function HSZG2EGMGrid(asset, post_log_earnings, post_log_medical,
        cash, log_earnings, log_medical)
        grids = map(x -> collect(Float64, x),
            (asset, post_log_earnings, post_log_medical,
             cash, log_earnings, log_medical))
        names = ("asset", "post log earnings", "post log medical",
                 "cash", "common log earnings", "common log medical")
        for (grid, name) in zip(grids, names)
            length(grid) >= 2 && all(isfinite, grid) &&
                all(diff(grid) .> 0.0) || throw(ArgumentError(
                    "$name grid must be finite and strictly increasing"))
        end
        first(grids[1]) == 0.0 || throw(ArgumentError(
            "the HSZ asset grid must begin at zero"))
        extrema(grids[2]) == extrema(grids[5]) || throw(ArgumentError(
            "post and common log-earnings grids must share endpoints"))
        extrema(grids[3]) == extrema(grids[6]) || throw(ArgumentError(
            "post and common log-medical grids must share endpoints"))
        return new(grids...)
    end
end

"""Numerical options for the paper's HSZ G2EGM solve."""
Base.@kwdef struct HSZG2EGMOptions
    bounding_box_padding::Int = 1
    minimum_barycentric_weight::Float64 = -0.20
    determinant_tolerance::Float64 = 1.0e-13
    value_atol::Float64 = 1.0e-11
    value_rtol::Float64 = 1.0e-10
    enforce_monotone_saving::Bool = true
    monotonicity_atol::Float64 = 1.0e-9
    monotonicity_rtol::Float64 = 1.0e-9
    # Retained for artifact/options compatibility; exact-cell repair is analytic.
    direct_iterations::Int = 64
end

"""Mutable construction diagnostics for an HSZ G2EGM solution."""
Base.@kwdef mutable struct HSZG2EGMSolveDiagnostics
    quadrature_expectations::Int = 0
    continuation_queries::Int = 0
    tetrahedra_considered::Int = 0
    tetrahedra_processed::Int = 0
    tetrahedra_degenerate::Int = 0
    tetrahedron_candidates::Int = 0
    tetrahedron_winners::Int = 0
    extrapolated_winners::Int = 0
    constraint_winners::Int = 0
    floor_kink_candidates::Int = 0
    floor_kink_winners::Int = 0
    q25_candidate_revaluations::Int = 0
    monotonicity_violations::Int = 0
    monotonicity_repairs::Int = 0
    monotonicity_violations_remaining::Int = 0
    direct_repair_winners::Int = 0
    asset_ceiling_winners::Int = 0
    minimum_cash_margin::Float64 = Inf
    minimum_earnings_margin::Float64 = Inf
    minimum_medical_margin::Float64 = Inf
end

"""Thread-safe query diagnostics for an HSZ G2EGM solution."""
mutable struct HSZG2EGMQueryDiagnostics
    consumption_queries::Int
    value_queries::Int
    domain_errors::Int
    asset_ceiling_choices::Int
    minimum_cash_margin::Float64
    minimum_earnings_margin::Float64
    minimum_medical_margin::Float64
    lock::ReentrantLock
end

HSZG2EGMQueryDiagnostics() = HSZG2EGMQueryDiagnostics(
    0, 0, 0, 0, Inf, Inf, Inf, ReentrantLock())

"""Solved age-indexed G2EGM policy and value arrays for the HSZ model."""
struct HSZG2EGMSolution{R<:GaussHermiteRule{2,25}}
    config::StochasticPTDConfig
    process::HSZProcess
    base_rule::R
    grid::HSZG2EGMGrid
    options::HSZG2EGMOptions
    policy::Vector{Array{Float64,3}}
    value::Vector{Array{Float64,3}}
    solve_diagnostics::HSZG2EGMSolveDiagnostics
    query_diagnostics::HSZG2EGMQueryDiagnostics
    fingerprint::String
end

function _centered_sinh_grid(mean::Float64, radius::Float64,
    points::Int, curvature::Float64)
    coordinates = range(-1.0, 1.0; length=points)
    transformed = curvature == 0.0 ? collect(coordinates) :
        sinh.(curvature .* coordinates) ./ sinh(curvature)
    grid = mean .+ radius .* transformed
    grid[1], grid[end] = mean - radius, mean + radius
    isodd(points) && (grid[(points + 1) ÷ 2] = mean)
    return grid
end

"""Publication diagnostic grid, declared independently of simulated paths."""
function hsz_g2egm_grid(process::HSZProcess;
    asset_maximum::Real=300.0, asset_points::Integer=141,
    common_cash_points::Integer=141,
    post_earnings_points::Integer=9, post_medical_points::Integer=9,
    common_earnings_points::Integer=9, common_medical_points::Integer=9,
    earnings_radius::Real=3.2, medical_radius::Real=3.2,
    cash_floor::Real=3.0, gross_return::Real=1.10)

    amax = Float64(asset_maximum)
    eradius, mradius = Float64(earnings_radius), Float64(medical_radius)
    amax > 0.0 && eradius > 0.0 && mradius > 0.0 || throw(ArgumentError(
        "HSZ grid maxima and radii must be positive"))
    asset = _single_log_grid(0.0, amax, Int(asset_points))
    post_e = _centered_sinh_grid(process.unconditional_log_earnings,
        eradius, Int(post_earnings_points), 2.0)
    post_m = _centered_sinh_grid(process.unconditional_log_medical,
        mradius, Int(post_medical_points), 2.0)
    common_e = _centered_sinh_grid(process.unconditional_log_earnings,
        eradius, Int(common_earnings_points), 1.7)
    common_m = _centered_sinh_grid(process.unconditional_log_medical,
        mradius, Int(common_medical_points), 1.7)
    maximum_net_income = exp(last(post_e)) - exp(first(post_m))
    cash_maximum = Float64(gross_return) * amax + maximum_net_income
    cash = _single_log_grid(Float64(cash_floor), cash_maximum,
        Int(common_cash_points))
    return HSZG2EGMGrid(asset, post_e, post_m, cash, common_e, common_m)
end

function _trilinear(xgrid, ygrid, zgrid, values::Array{Float64,3},
    x::Float64, y::Float64, z::Float64)
    ix, iy, iz = _search_interval(xgrid, x),
                 _search_interval(ygrid, y), _search_interval(zgrid, z)
    1 <= ix < length(xgrid) || throw(DomainError(x,
        "first trilinear coordinate is outside the grid"))
    1 <= iy < length(ygrid) || throw(DomainError(y,
        "second trilinear coordinate is outside the grid"))
    1 <= iz < length(zgrid) || throw(DomainError(z,
        "third trilinear coordinate is outside the grid"))
    wx = (x - xgrid[ix]) / (xgrid[ix + 1] - xgrid[ix])
    wy = (y - ygrid[iy]) / (ygrid[iy + 1] - ygrid[iy])
    wz = (z - zgrid[iz]) / (zgrid[iz + 1] - zgrid[iz])
    result = 0.0
    @inbounds for dz in 0:1, dy in 0:1, dx in 0:1
        weight = (dx == 0 ? 1.0 - wx : wx) *
                 (dy == 0 ? 1.0 - wy : wy) *
                 (dz == 0 ? 1.0 - wz : wz)
        result += weight * values[ix + dx, iy + dy, iz + dz]
    end
    return result
end

function _hsz_array_fingerprint(policy, value)
    context = SHA.SHA2_256_CTX()
    SHA.update!(context, collect(codeunits("ptd-hsz-g2egm-arrays-v1\n")))
    for (label, arrays) in (("policy", policy), ("value", value))
        SHA.update!(context, collect(codeunits("$label|$(length(arrays))\n")))
        for array in arrays
            SHA.update!(context, collect(codeunits("$(size(array))\n")))
            words = hton.(reinterpret(UInt64, vec(array)))
            SHA.update!(context, reinterpret(UInt8, words))
        end
    end
    return bytes2hex(SHA.digest!(context))
end

solution_fingerprint(solution::HSZG2EGMSolution) = solution.fingerprint
