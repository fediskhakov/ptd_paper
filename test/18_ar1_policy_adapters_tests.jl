# Tests for src/simulation/policies/ar1.jl and source-order file 18.
using Test
import Ptd

const ARPA = Ptd

@testset verbose=true "AR(1) contest policy adapters" begin
    @testset "declared G2EGM grids are deterministic and evaluator-blind" begin
        pilot = ARPA.pilot_ar1_g2_grid_configuration()
        reported = ARPA.candidate_ar1_g2_grid_configuration()

        @test pilot.schema == :ar1_g2egm_pilot_v1
        @test reported.schema == :ar1_g2egm_candidate_reported_v1
        @test length(pilot.asset_grid) == 401
        @test length(pilot.post_log_income_grid) == 31
        @test length(pilot.common_cash_grid) == 301
        @test length(pilot.common_log_income_grid) == 21
        @test length(reported.asset_grid) == 1_601
        @test length(reported.post_log_income_grid) == 61
        @test length(reported.common_cash_grid) == 1_201
        @test length(reported.common_log_income_grid) == 41
        for grids in (pilot, reported)
            @test extrema(grids.asset_grid) == (0.0, 30_000.0)
            @test extrema(grids.post_log_income_grid) == (-3.0, 3.0)
            @test extrema(grids.common_log_income_grid) == (-3.0, 3.0)
            @test first(grids.common_cash_grid) == 3.0
            @test last(grids.common_cash_grid) ==
                1.10 * 30_000.0 + exp(3.0)
            @test all(diff(grids.asset_grid) .> 0.0)
            @test all(diff(grids.post_log_income_grid) .> 0.0)
            @test all(diff(grids.common_cash_grid) .> 0.0)
            @test all(diff(grids.common_log_income_grid) .> 0.0)
            @test grids.post_log_income_grid[
                (length(grids.post_log_income_grid) + 1) ÷ 2] == 0.0
            @test grids.common_log_income_grid[
                (length(grids.common_log_income_grid) + 1) ÷ 2] == 0.0
        end

        declared = ARPA._plain_ar1_g2_grid_configuration(pilot)
        @test declared.asset.points == 401
        @test declared.post_log_income.points == 31
        @test declared.common_cash.points == 301
        @test declared.common_log_income.points == 21
        @test all(value -> occursin(r"^[0-9a-f]{64}$", value),
            (declared.asset.fingerprint,
             declared.post_log_income.fingerprint,
             declared.common_cash.fingerprint,
             declared.common_log_income.fingerprint))
        @test declared ==
            ARPA._plain_ar1_g2_grid_configuration(
                ARPA.pilot_ar1_g2_grid_configuration())

        @test_throws ArgumentError ARPA.AR1G2ContestGridConfiguration(
            :bad, [0.0, 1.0], [-1.0, 0.0, 1.0], [3.0, 4.0],
            [-2.0, 0.0, 2.0])
        @test_throws ArgumentError ARPA.AR1G2ContestGridConfiguration(
            :bad, [0.0, 1.0], [-1.0, 0.0, 1.0], [3.0, 4.0],
            [-1.0, 0.0, 1.0]; barycentric_weight_floor=0.1)
    end

    @testset "PTD-SCEQ and PTD-OSA are pure state-only AR(1) actions" begin
        config = ARPA.StochasticPTDConfig(beta=0.96, gamma=1.5,
            return_rate=0.02, cash_floor=1.0, terminal_date=2)
        process = ARPA.LogAR1Process(rho=0.9, sigma=0.1,
            unconditional_log_mean=0.0)
        rule = ARPA.gauss_hermite_rule(process, 5)
        ptd_osa_options = ARPA.PTDOSAOptions(direct_iterations=8)
        sceq = ARPA.build_ar1_contest_policy(
            :ptd_sceq, config, process; base_rule=rule)
        oa = ARPA.build_ar1_contest_policy(
            :ptd_osa, config, process; base_rule=rule, ptd_osa_options)

        @test sceq isa ARPA.PTDSCEQAR1ContestPolicy
        @test oa isa ARPA.PTDOSAAR1ContestPolicy
        @test isnothing(ARPA.validate_ar1_contest_policy_alignment(
            sceq, :ptd_sceq, config, process))
        @test isnothing(ARPA.validate_ar1_contest_policy_alignment(
            oa, :ptd_osa, config, process))
        @test ARPA.ar1_contest_quadrature_audit(sceq) ==
            (method=:ptd_sceq, uses_quadrature=false)
        audit = ARPA.ar1_contest_quadrature_audit(oa)
        @test audit.nodes_per_dimension == 5
        @test audit.base_nodes == rule.nodes
        @test audit.base_weights == rule.weights

        initial = ARPA.ObservedState(1, 3.0, ARPA.LogAR1State(0.0))
        terminal = ARPA.ObservedState(2, 2.25, ARPA.LogAR1State(0.05))
        for (policy, method) in ((sceq, :ptd_sceq), (oa, :ptd_osa))
            action = policy(initial)
            @test action.method == method
            @test 0.0 <= action.consumption <= initial.cash
            @test action.saving ≈ initial.cash - action.consumption
            terminal_action = policy(terminal)
            @test terminal_action.consumption == terminal.cash
            @test terminal_action.saving == 0.0
            @test !applicable(policy, terminal, zeros(1, 1))
        end
        @test_throws ArgumentError sceq(ARPA.ObservedState(
            1, 3.0, ARPA.IIDState()))
        @test_throws ArgumentError ARPA.build_ar1_contest_policy(
            :ptd_osa, config, process;
            base_rule=ARPA.gauss_hermite_rule(process, 3))
        @test_throws ArgumentError ARPA.build_ar1_contest_policy(
            :unknown, config, process)
    end

    @testset "adapter binds only to pure-Julia G2EGM" begin
        @test ARPA.G2EGM isa Module
        @test all(name -> isdefined(ARPA.G2EGM, name),
            ARPA.AR1_G2EGM_REQUIRED_API)
        pilot = ARPA.pilot_ar1_g2_grid_configuration()
        solver_grid = ARPA._materialize_g2egm_grids(pilot)
        solver_options = ARPA._materialize_g2egm_options(
            (;), pilot.barycentric_weight_floor)
        @test solver_grid isa ARPA.G2EGM.AR1G2EGMGrid
        @test solver_options isa ARPA.G2EGM.AR1G2EGMOptions
        @test solver_grid.asset == pilot.asset_grid
        @test solver_grid.post_log_income == pilot.post_log_income_grid
        @test solver_grid.cash == pilot.common_cash_grid
        @test solver_grid.log_income == pilot.common_log_income_grid
        @test solver_options.minimum_barycentric_weight ==
            pilot.barycentric_weight_floor
        @test_throws ArgumentError ARPA._materialize_g2egm_options(
            (minimum_barycentric_weight=-0.1,),
            pilot.barycentric_weight_floor)
        checkpointed = ARPA._checkpoint_g2egm_diagnostic((
            solve=(minimum_solve_cash_margin=Inf, calls=1),
            query=(minimum_log_income_margin=0.25,)))
        @test checkpointed.solve.minimum_solve_cash_margin === nothing
        @test checkpointed.query.minimum_log_income_margin == 0.25
        @test_throws ErrorException ARPA._checkpoint_g2egm_diagnostic(
            (maximum_error=Inf,))
    end

    @testset "real G2EGM adapter survives an authenticated checkpoint" begin
        config = ARPA.StochasticPTDConfig(beta=0.96, gamma=1.0,
            return_rate=0.10, cash_floor=3.0, terminal_date=2)
        process = ARPA.LogAR1Process(rho=0.9, sigma=0.1,
            unconditional_log_mean=0.0)
        rule = ARPA.gauss_hermite_rule(process, 5)
        solver_grid = ARPA.G2EGM.ar1_g2egm_grid(
            asset_maximum=20.0,
            asset_points=201,
            common_cash_points=241,
            post_log_income_points=31,
            common_log_income_points=25,
            cash_floor=3.0,
            gross_return=1.10,
            log_income_bounds=(-3.0, 3.0),
        )
        contest_grid = ARPA.AR1G2ContestGridConfiguration(
            :ar1_g2egm_checkpoint_smoke_v1,
            solver_grid.asset,
            solver_grid.post_log_income,
            solver_grid.cash,
            solver_grid.log_income)
        policy = ARPA.build_ar1_contest_policy(
            :g2egm, config, process; base_rule=rule,
            g2egm_grids=contest_grid, g2egm_options=(;))
        @test policy isa ARPA.G2EGMAR1ContestPolicy
        initial = ARPA.ObservedState(1, 5.0, ARPA.LogAR1State(0.0))
        @test_throws DomainError policy(ARPA.ObservedState(
            1, last(solver_grid.asset) + 1.0e-6,
            ARPA.LogAR1State(0.0)))
        padded_terminal = ARPA.ObservedState(
            2, last(solver_grid.asset) + 1.0,
            ARPA.LogAR1State(0.0))
        @test policy(padded_terminal).consumption == padded_terminal.cash
        @test_throws DomainError policy(ARPA.ObservedState(
            2, last(solver_grid.cash) + 1.0e-6,
            ARPA.LogAR1State(0.0)))
        ARPA.reset_ar1_contest_diagnostics!(policy)
        evaluation = ARPA.evaluate_policy_paths(
            policy, :g2egm, config, process, initial, zeros(1, 1))
        diagnostics = (
            quadrature=ARPA.ar1_contest_quadrature_audit(policy),
            solver=ARPA.ar1_contest_diagnostics(policy),
        )
        @test all(isfinite, evaluation.utility)
        @test diagnostics.quadrature.nodes_per_dimension == 5
        @test diagnostics.solver.solver.query.domain_errors == 0

    end
end
