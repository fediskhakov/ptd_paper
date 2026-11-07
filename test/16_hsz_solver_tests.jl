# Tests for src/g2egm/hsz_model.jl and src/g2egm/hsz_solver.jl (file 16).
module HSZG2EGMSolverTests
using Test
import Ptd
const G2 = Ptd.G2EGM

# Independent scalar search used only on intervals where the reference
# continuation is affine and the CRRA objective is concave.
function golden_reference(objective, lower, upper)
    endpoints = (lower, upper)
    ratio = (sqrt(5.0) - 1.0) / 2.0
    left, right = upper - ratio * (upper - lower), lower + ratio * (upper - lower)
    vl, vr = objective(left), objective(right)
    for _ in 1:100
        if vl < vr
            lower, left, vl = left, right, vr
            right = lower + ratio * (upper - lower)
            vr = objective(right)
        else
            upper, right, vr = right, left, vl
            left = upper - ratio * (upper - lower)
            vl = objective(left)
        end
    end
    actions = [endpoints...; lower; left; right; upper]
    values = objective.(actions)
    best = argmax(values)
    return (saving=actions[best], value=values[best])
end

function constant_income_fixture()
    process = Ptd.HSZProcess(rho_earnings=0.0, sigma_earnings=0.0,
        rho_medical=0.0, sigma_medical=0.0,
        unconditional_log_earnings=log(2.0), unconditional_log_medical=0.0)
    config = Ptd.StochasticPTDConfig(beta=0.96, gamma=1.0,
        return_rate=0.0, cash_floor=1.0, terminal_date=2)
    rule = Ptd.gauss_hermite_rule(process, 5)
    grid = G2.HSZG2EGMGrid(collect(0.0:1.0:5.0), [-1.0, 1.0], [-1.0, 1.0],
        [1.0, 2.0, 2.39, 2.4, 2.41, 3.0, 4.0, 5.0, 8.0],
        [-1.0, 1.0], [-1.0, 1.0])
    bump = cash -> 4.0 * max(1.0 - abs(cash - 2.4) / 0.01, 0.0)
    child_value = [bump(cash) for cash in grid.cash,
        _ in grid.log_earnings, _ in grid.log_medical]
    child_policy = [cash for cash in grid.cash,
        _ in grid.log_earnings, _ in grid.log_medical]
    return (; config, process, rule, grid, bump, child_value, child_policy)
end

@testset "HSZ Q25 tetrahedral G2EGM" begin
    process = Ptd.HSZProcess(rho_earnings=0.9, sigma_earnings=0.1,
        rho_medical=0.9, sigma_medical=0.1,
        unconditional_log_earnings=log(2.0),
        unconditional_log_medical=log(0.5))
    config = Ptd.StochasticPTDConfig(beta=0.98, gamma=1.0,
        return_rate=0.1, cash_floor=3.0, terminal_date=3)
    rule = Ptd.gauss_hermite_rule(process, 5)
    @test rule isa Ptd.GaussHermiteRule{2,25}
    @test sum(rule.weights) ≈ 1.0
    grid = Ptd.G2EGM.hsz_g2egm_grid(process; asset_maximum=30.0,
        asset_points=11, common_cash_points=13,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5)
    solution = Ptd.G2EGM.build_hsz_solution(config, process;
        base_rule=rule, grid=grid)
    observed = Ptd.ObservedState(1, 10.0,
        Ptd.HSZState(log(2.0), log(0.5)))
    consumption = Ptd.G2EGM.action(solution, observed)
    @test 0.0 < consumption <= observed.cash
    @test Ptd.G2EGM.action(solution, observed) == consumption
    @test Ptd.G2EGM.action(solution, Ptd.ObservedState(3, 10.0,
        observed.exogenous)) == 10.0
    @test Ptd.G2EGM.quadrature_audit(solution).total_nodes == 25
    @test Ptd.G2EGM.diagnostics(solution).solve.tetrahedra_processed > 0
    audit = Ptd.G2EGM.diagnostics(solution).solve
    @test audit.monotonicity_violations_remaining == 0
    @test audit.constraint_winners + audit.tetrahedron_winners +
          audit.floor_kink_winners + audit.direct_repair_winners ==
          length(grid.cash) * length(grid.log_earnings) *
          length(grid.log_medical) * (config.terminal.final_date - 1)
    @test Ptd.G2EGM.HSZ_G2EGM_SOLVER_SCHEMA ==
          "hsz-tetrahedral-g2egm-exact-cell-repair-v2"
    @test occursin(r"^[0-9a-f]{64}$",
        Ptd.G2EGM.solution_fingerprint(solution))
    @test isnothing(Ptd.G2EGM.validate_alignment(
        solution, config, process, rule, grid))
end

@testset "HSZ direct repair includes hidden interpolation peaks" begin
    (; config, process, rule, grid, bump, child_value, child_policy) = constant_income_fixture()
    cash = 6.0
    # Every coarse asset interval is searched, as in the former HSZ repair.
    # Golden search on [1,2] discards the narrow peak at a=1.4 before sampling it.
    objective = saving -> log(cash - saving) + config.beta * sum(rule.weights) * bump(saving + 1.0)
    old_best = maximum(golden_reference(objective, grid.asset[i], grid.asset[i + 1]).value
        for i in 1:length(grid.asset)-1)
    result = G2._hsz_direct_action(config, process, rule, grid, G2.HSZG2EGMOptions(),
        child_value, child_policy, cash, 0.0, 0.0, G2.HSZG2EGMSolveDiagnostics())
    @test result.saving ≈ 1.4 atol=3e-15 rtol=0.0
    @test result.value ≈ objective(1.4) atol=5e-13 rtol=0.0
    @test result.value > old_best + 3.0
    @test 0.0 < result.consumption <= cash
    @test 0.0 <= result.saving <= last(grid.asset)
end

@testset "HSZ Q25 repair agrees with independent off-grid objective" begin
    process = Ptd.HSZProcess(rho_earnings=0.35, sigma_earnings=0.05,
        rho_medical=0.4, sigma_medical=0.07,
        unconditional_log_earnings=log(1.5), unconditional_log_medical=log(0.7))
    config = Ptd.StochasticPTDConfig(beta=0.96, gamma=1.0,
        return_rate=0.1, cash_floor=1.0, terminal_date=2)
    rule = Ptd.gauss_hermite_rule(process, 5)
    grid = G2.HSZG2EGMGrid([0.0, 2.0, 5.0], [-1.5, 0.0, 1.5], [-1.5, 0.0, 1.5],
        [1.0, 2.85, 3.0, 3.15, 5.0, 12.0], [-1.5, 0.0, 1.5], [-1.5, 0.0, 1.5])
    # This formula is piecewise affine in cash and bilinear in e,m, so its
    # samples reproduce it exactly under trilinear interpolation. The oracle
    # evaluates the formula directly, without any solver interpolation helper.
    h = x -> 0.2x + 3max(1.0 - abs(x - 3.0) / 0.15, 0.0) + 0.7max(x - 5.0, 0.0)
    child = (x, e, m) -> (1.0 + 0.12e - 0.08m + 0.07e*m) * h(x) + 0.1e - 0.2m
    child_value = [child(x, e, m) for x in grid.cash, e in grid.log_earnings, m in grid.log_medical]
    child_policy = [x for x in grid.cash, _ in grid.log_earnings, _ in grid.log_medical]
    earnings, medical = 0.17, -0.26
    branches = [begin
        e = process.unconditional_log_earnings + process.rho_earnings *
            (earnings - process.unconditional_log_earnings) + process.std_earnings * node[1]
        m = process.unconditional_log_medical + process.rho_medical *
            (medical - process.unconditional_log_medical) + process.std_medical * node[2]
        (; e, m, income=exp(e) - exp(m), weight)
    end for (node, weight) in zip(rule.nodes, rule.weights)]
    @test all(branch.e ∉ grid.log_earnings && branch.m ∉ grid.log_medical for branch in branches)
    @test length(unique(branch.income for branch in branches)) == 25
    gross, cash = 1.0 + config.return_rate, 6.0
    objective = saving -> log(cash - saving) + config.beta * sum(
        branch.weight * child(max(gross * saving + branch.income, config.cash_floor),
            branch.e, branch.m) for branch in branches)
    # The independent formula changes only at its known kinks and the floor.
    boundaries = [0.0, last(grid.asset)]
    for branch in branches, knot in (1.0, 2.85, 3.0, 3.15, 5.0)
        saving = (knot - branch.income) / gross
        0.0 < saving < last(grid.asset) && push!(boundaries, saving)
    end
    sort!(unique!(boundaries))
    references = [golden_reference(objective, boundaries[i], boundaries[i + 1])
        for i in 1:length(boundaries)-1]
    reference = references[argmax(getfield.(references, :value))]
    result = G2._hsz_direct_action(config, process, rule, grid, G2.HSZG2EGMOptions(),
        child_value, child_policy, cash, earnings, medical, G2.HSZG2EGMSolveDiagnostics())
    @test result.value ≈ reference.value atol=2e-12 rtol=0.0
    @test result.value ≈ objective(result.saving) atol=2e-12 rtol=0.0
    @test result.saving ≈ reference.saving atol=2e-6 rtol=0.0
end

@testset "HSZ flat utility repair preserves analytic saving" begin
    (; process, rule) = constant_income_fixture()
    config = Ptd.StochasticPTDConfig(beta=0.96, gamma=3.5,
        return_rate=0.0, cash_floor=1.0, terminal_date=2)
    grid = G2.HSZG2EGMGrid([0.0, 5_000.0, 50_000.0], [-1.0, 1.0], [-1.0, 1.0],
        [1.0, 8193.0, 16385.0, 32769.0, 65537.0, 131073.0],
        [-1.0, 1.0], [-1.0, 1.0])
    # Powers of two keep the same tiny slope exactly representable in every
    # stored cash cell. Several candidates have rounding-ambiguous utility.
    slope = 2.0^-46
    child_value = [1000.0 + (x - 1.0) * slope for x in grid.cash,
        _ in grid.log_earnings, _ in grid.log_medical]
    child_policy = [x for x in grid.cash, _ in grid.log_earnings, _ in grid.log_medical]
    expected_consumption = setprecision(BigFloat, 256) do
        Float64((BigFloat(config.beta) * BigFloat(slope) * sum(BigFloat.(rule.weights)))^
            (-1 / BigFloat(config.gamma)))
    end
    options = G2.HSZG2EGMOptions(value_atol=1.0, value_rtol=1.0)
    cash_states = [20_000.0, 20_000.001, 20_000.002]
    results = [G2._hsz_direct_action(config, process, rule, grid, options,
        child_value, child_policy, cash, 0.0, 0.0, G2.HSZG2EGMSolveDiagnostics()) for cash in cash_states]
    for (cash, result) in zip(cash_states, results)
        @test result.consumption ≈ expected_consumption atol=1e-8 rtol=0.0
        @test result.saving ≈ cash - expected_consumption atol=1e-8 rtol=0.0
    end
    @test all(diff(getfield.(results, :saving)) .> 0.0)
    @test_throws ArgumentError G2._hsz_direct_action(config, process, rule, grid, options,
        child_value, child_policy, 0.0, 0.0, 0.0, G2.HSZG2EGMSolveDiagnostics())
end

@testset "HSZ monotonicity repair propagates and audits residual violations" begin
    (; config, process, rule, grid, child_value, child_policy) = constant_income_fixture()
    options = G2.HSZG2EGMOptions()
    policy = [G2._hsz_direct_action(config, process, rule, grid, options,
        child_value, child_policy, x, e, m, G2.HSZG2EGMSolveDiagnostics()).consumption
        for x in grid.cash, e in grid.log_earnings, m in grid.log_medical]
    # Only nodes 2->3 initially violate monotonicity. Repairing node 3 then
    # reveals a new violation at 3->4, which a single repair pass would miss.
    policy[2, :, :] .= grid.cash[2] - 1.5
    policy[3, :, :] .= grid.cash[3] - 0.3
    policy[4, :, :] .= grid.cash[4] - 0.9
    original = copy(policy)
    values = zeros(size(policy))
    sources = zeros(UInt8, size(policy))
    audit_only = G2.HSZG2EGMSolveDiagnostics()
    G2._repair_hsz_monotonicity!(policy, values, config, process, rule, grid,
        G2.HSZG2EGMOptions(enforce_monotone_saving=false), child_value, child_policy,
        audit_only; winner_source=sources)
    slices = length(grid.log_earnings) * length(grid.log_medical)
    @test policy == original
    @test all(iszero, values) && all(iszero, sources)
    @test audit_only.monotonicity_repairs == 0
    @test audit_only.monotonicity_violations_remaining == slices
    audit = G2.HSZG2EGMSolveDiagnostics()
    G2._repair_hsz_monotonicity!(policy, values, config, process, rule, grid,
        options, child_value, child_policy, audit; winner_source=sources)
    last_audited = searchsortedlast(grid.cash, last(grid.asset))
    for k in eachindex(grid.log_medical), j in eachindex(grid.log_earnings)
        savings = grid.cash[1:last_audited] .- policy[1:last_audited, j, k]
        @test all(diff(savings) .>= -1e-12)
        @test savings[2:4] ≈ fill(1.4, 3) atol=3e-15 rtol=0.0
        @test all(sources[2:4, j, k] .== UInt8(3))
    end
    @test audit.monotonicity_repairs == 3slices
    @test audit.monotonicity_violations_remaining == 0
end

end # module
