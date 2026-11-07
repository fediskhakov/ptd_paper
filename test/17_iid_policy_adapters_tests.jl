# Tests for src/simulation/policies/iid.jl and source-order file 17.
using Random
using JLD2

const PA = Ptd

_same_action(left, right) = all(name -> isequal(getfield(left, name),
    getfield(right, name)), fieldnames(PA.ActionResult))

function _finite_or_nothing_diagnostic(value)
    if value === nothing || value isa Bool || value isa Symbol ||
       value isa AbstractString || value isa Integer
        return true
    elseif value isa AbstractFloat
        return isfinite(value)
    elseif value isa NamedTuple || value isa Tuple || value isa AbstractVector
        return all(_finite_or_nothing_diagnostic, values(value))
    elseif value isa AbstractDict
        return all(_finite_or_nothing_diagnostic, keys(value)) &&
            all(_finite_or_nothing_diagnostic, values(value))
    end
    return false
end

@testset verbose=true "IID contest policy adapters" begin
    config = PA.StochasticPTDConfig(beta=0.96, gamma=1.5,
        return_rate=0.02, cash_floor=1.0, terminal_date=3)
    process = PA.IIDLognormalProcess(sigma=0.1)
    base_rule = PA.gauss_hermite_rule(process, 5)
    test_grid = collect(range(0.0, 12.0; length=33))
    quick_ptd_osa = PA.PTDOSAOptions(direct_iterations=8)

    @testset "separate construction, alignment, and shared rule" begin
        @test :validate_iid_contest_policy_alignment in names(PA)
        reported_grid = PA.iid_contest_asset_grid()
        @test length(reported_grid) == PA.IID_DCEGM_REPORTED_POINTS == 6_401
        @test extrema(reported_grid) ==
            (0.0, PA.IID_DCEGM_ASSET_MAXIMUM) == (0.0, 60_000.0)
        @test all(diff(reported_grid) .> 0.0)
        @test reported_grid[2] ≈ exp(log1p(60_000.0) / 6_400) - 1.0
        @test_throws ArgumentError PA.iid_dcegm_asset_grid(1)
        @test_throws ArgumentError PA.iid_dcegm_asset_grid(4;
            asset_maximum=Inf)
        terminal_config = PA.StochasticPTDConfig(beta=0.96, gamma=1.5,
            return_rate=0.02, cash_floor=1.0, terminal_date=1)
        default_dcegm = PA.build_iid_contest_policy(
            :dcegm, terminal_config, process; base_rule=base_rule)
        @test length(default_dcegm.asset_grid) == 6_401
        @test extrema(default_dcegm.asset_grid) == (0.0, 60_000.0)

        sceq = PA.build_iid_contest_policy(:ptd_sceq, config, process;
            asset_grid=test_grid, base_rule=base_rule)
        oa = PA.build_iid_contest_policy(:ptd_osa, config, process;
            asset_grid=test_grid, base_rule=base_rule, ptd_osa_options=quick_ptd_osa)
        dcegm = PA.build_iid_contest_policy(:dcegm, config, process;
            asset_grid=test_grid, base_rule=base_rule)

        @test sceq isa PA.PTDSCEQIIDContestPolicy
        @test oa isa PA.PTDOSAIIDContestPolicy
        @test dcegm isa PA.DCEGMIIDContestPolicy
        @test sceq.cache !== oa.cache
        @test !sceq.cache.verify_integrity_on_hit
        @test !oa.cache.verify_integrity_on_hit
        @test isnothing(PA.validate_iid_contest_policy_alignment(
            sceq, :ptd_sceq, config, process))
        @test isnothing(PA.validate_iid_contest_policy_alignment(
            oa, :ptd_osa, config, process))
        @test isnothing(PA.validate_iid_contest_policy_alignment(
            dcegm, :dcegm, config, process))
        @test dcegm.solution.solve_diagnostics.quadrature_expectations > 0
        @test dcegm.solution.solve_diagnostics.quadgk_fallbacks == 0
        @test dcegm.asset_grid == test_grid
        @test dcegm.solution.agent.bf.β == config.beta
        @test dcegm.solution.agent.bf.γ == config.gamma
        @test dcegm.solution.agent.bf.r == config.return_rate
        @test dcegm.solution.agent.bf.xFloor == config.cash_floor
        @test dcegm.solution.agent.bf.tMax == config.terminal.final_date

        audit = PA.iid_contest_quadrature_audit(oa, dcegm)
        @test audit.nodes_bit_identical
        @test audit.weights_bit_identical
        @test audit.income_nodes_bit_identical
        @test audit.ptd_osa.nodes_per_dimension == 5
        @test audit.ptd_osa.base_node_bits == audit.dcegm.base_node_bits
        @test audit.ptd_osa.base_weight_bits ==
            audit.dcegm.distribution_weight_bits
        @test sum(audit.dcegm.distribution_weights .*
            audit.dcegm.distribution_nodes) ≈ 1.0 rtol=2e-13

        wrong_beta = PA.StochasticPTDConfig(beta=0.95, gamma=config.gamma,
            return_rate=config.return_rate, cash_floor=config.cash_floor,
            terminal_date=config.terminal.final_date)
        wrong_horizon = PA.StochasticPTDConfig(beta=config.beta,
            gamma=config.gamma, return_rate=config.return_rate,
            cash_floor=config.cash_floor,
            terminal_date=config.terminal.final_date + 1)
        wrong_sigma = PA.IIDLognormalProcess(sigma=0.5)
        for (policy, method) in
            ((sceq, :ptd_sceq), (oa, :ptd_osa), (dcegm, :dcegm))
            @test_throws ArgumentError PA.validate_iid_contest_policy_alignment(
                policy, method === :ptd_sceq ? :ptd_osa : :ptd_sceq,
                config, process)
            @test_throws ArgumentError PA.validate_iid_contest_policy_alignment(
                policy, method, wrong_beta, process)
            @test_throws ArgumentError PA.validate_iid_contest_policy_alignment(
                policy, method, config, wrong_sigma)
            @test_throws ArgumentError PA.validate_iid_contest_policy_alignment(
                policy, method, wrong_horizon, process)
        end

        cross_config_ptd_osa = PA.build_iid_contest_policy(
            :ptd_osa, wrong_beta, process;
            base_rule=base_rule, ptd_osa_options=quick_ptd_osa)
        @test_throws ArgumentError PA.iid_contest_quadrature_audit(
            cross_config_ptd_osa, dcegm)
        high_variance_rule = PA.gauss_hermite_rule(wrong_sigma, 5)
        cross_panel_ptd_osa = PA.build_iid_contest_policy(
            :ptd_osa, config, wrong_sigma;
            base_rule=high_variance_rule, ptd_osa_options=quick_ptd_osa)
        @test_throws ArgumentError PA.iid_contest_quadrature_audit(
            cross_panel_ptd_osa, dcegm)

        mismatched_grid = copy(dcegm.asset_grid)
        mismatched_grid[2] = nextfloat(mismatched_grid[2])
        grid_misaligned = PA.DCEGMIIDContestPolicy(dcegm.config,
            dcegm.process, dcegm.base_rule, dcegm.egm_rule,
            dcegm.solution, mismatched_grid, dcegm.policy_fingerprint)
        @test_throws ErrorException PA.validate_iid_contest_policy_alignment(
            grid_misaligned, :dcegm, config, process)

        short_config = PA.StochasticPTDConfig(beta=config.beta,
            gamma=config.gamma, return_rate=config.return_rate,
            cash_floor=config.cash_floor, terminal_date=2)
        short_dcegm = PA.build_iid_contest_policy(:dcegm, short_config,
            process; asset_grid=test_grid, base_rule=base_rule)
        horizon_misaligned = PA.DCEGMIIDContestPolicy(config, process,
            short_dcegm.base_rule, short_dcegm.egm_rule,
            short_dcegm.solution, short_dcegm.asset_grid,
            short_dcegm.policy_fingerprint)
        @test_throws ErrorException PA.validate_iid_contest_policy_alignment(
            horizon_misaligned, :dcegm, config, process)

        @test_throws ArgumentError PA.build_iid_contest_policy(
            :dcegm, config, PA.IIDLognormalProcess(sigma=0.1,
                mean_income=2.0); asset_grid=test_grid)
        @test_throws ArgumentError PA.build_iid_contest_policy(
            :dcegm, config, PA.LogAR1Process(rho=0.8, sigma=0.1);
            asset_grid=test_grid)
        @test_throws ArgumentError PA.build_iid_contest_policy(
            :unknown, config, process)
        @test_throws ArgumentError PA.build_iid_contest_policy(
            :ptd_osa, config, process;
            base_rule=PA.gauss_hermite_rule(process, 3))
        @test_throws ArgumentError PA.build_iid_contest_policy(
            :dcegm, config, process; asset_grid=[0.0, 2.0, 1.0])
    end

    @testset "pure state-only actions and actual hidden-path evaluator" begin
        sceq = PA.build_iid_contest_policy(:ptd_sceq, config, process;
            base_rule=base_rule)
        oa = PA.build_iid_contest_policy(:ptd_osa, config, process;
            base_rule=base_rule, ptd_osa_options=quick_ptd_osa)
        dcegm = PA.build_iid_contest_policy(:dcegm, config, process;
            asset_grid=test_grid, base_rule=base_rule)
        actions = (ptd_sceq=sceq, ptd_osa=oa, dcegm=dcegm)
        initial = PA.ObservedState(1, 3.0, PA.IIDState())
        at_floor = PA.ObservedState(1, config.cash_floor, PA.IIDState())
        terminal = PA.ObservedState(3, 2.25, PA.IIDState())
        solve_epoch = dcegm.solution.solve_epoch

        initial_actions = map(policy -> policy(initial), actions)
        uncached_sceq = PA.ptd_sceq(
            config, process, initial; evaluate_objective=false)
        uncached_ptd_osa = PA.ptd_osa(config, process, initial, base_rule;
            options=quick_ptd_osa)
        @test _same_action(initial_actions.ptd_sceq, uncached_sceq)
        @test _same_action(initial_actions.ptd_osa, uncached_ptd_osa)
        for method in PA.IID_CONTEST_METHODS
            action = getproperty(initial_actions, method)
            @test action.method == method
            @test 0.0 <= action.consumption <= initial.cash
            @test action.saving ≈ initial.cash - action.consumption atol=1e-14
            @test !isempty(action.plan_fingerprint)
            if method === :ptd_osa
                @test !isnothing(action.objective)
                @test !isnan(action.objective)
            else
                @test isnothing(action.objective)
            end
        end
        @test initial_actions.dcegm.plan_fingerprint ==
            dcegm.policy_fingerprint
        @test length(dcegm.policy_fingerprint) == 64
        for policy in actions
            action = policy(at_floor)
            @test 0.0 <= action.consumption <= at_floor.cash
            @test action.saving ≈ at_floor.cash - action.consumption atol=1e-14
        end
        for policy in actions
            action = policy(terminal)
            @test action.consumption == terminal.cash
            @test action.saving == 0.0
            @test applicable(policy, terminal)
            @test !applicable(policy, terminal, zeros(2, 2))
            @test_throws MethodError policy(terminal; seed=123)
            @test_throws MethodError policy(terminal; rng=Xoshiro(123))
        end
        @test_throws ArgumentError sceq(PA.ObservedState(
            1, 3.0, PA.LogAR1State(0.0)))
        @test_throws ArgumentError oa(PA.ObservedState(1, 0.5, PA.IIDState()))

        hidden_a = randn(Xoshiro(41), 3, 2)
        hidden_b = randn(Xoshiro(91), 3, 2)
        copy_a, copy_b = copy(hidden_a), copy(hidden_b)
        evaluation_a = PA.evaluate_contest_paths(
            actions, config, process, initial, hidden_a)
        evaluation_b = PA.evaluate_contest_paths(
            actions, config, process, initial, hidden_b)
        @test evaluation_a.method_ids == collect(PA.IID_CONTEST_METHODS)
        @test evaluation_b.method_ids == collect(PA.IID_CONTEST_METHODS)
        @test hidden_a == copy_a
        @test hidden_b == copy_b
        @test hidden_a != hidden_b
        for method in PA.IID_CONTEST_METHODS
            @test _same_action(getproperty(initial_actions, method),
                getproperty(actions, method)(initial))
        end
        @test dcegm.solution.solve_epoch == solve_epoch

        expected_rng = copy(Random.default_rng())
        map(policy -> policy(initial), actions)
        @test rand() == rand(expected_rng)
    end

    @testset "evaluation diagnostics are serializable and reset without setup loss" begin
        policies = PA.build_iid_contest_policies(config, process;
            asset_grid=test_grid, base_rule=base_rule, ptd_osa_options=quick_ptd_osa)
        state = PA.ObservedState(1, 3.0, PA.IIDState())
        policies.ptd_sceq(state)
        policies.ptd_osa(state)
        policies.dcegm(state)
        snapshot = PA.iid_contest_diagnostics(policies)
        @test snapshot.ptd_sceq.cache_builds == 1
        @test snapshot.ptd_sceq.verified_cache_entries ==
            snapshot.ptd_sceq.cache_entries
        @test snapshot.ptd_osa.calls == 1
        @test snapshot.ptd_osa.verified_cache_entries ==
            snapshot.ptd_osa.cache_entries
        @test snapshot.ptd_osa.certified_calls +
            snapshot.ptd_osa.certification_failures == 1
        @test snapshot.dcegm.solve.quadrature_expectations > 0
        @test snapshot.dcegm.query.consumption_queries == 1
        @test snapshot.dcegm.query.value_queries == 0
        @test snapshot.dcegm.solve.min_continuation_lower_log_income_margin ===
            nothing
        @test snapshot.dcegm.solve.min_continuation_upper_log_income_margin ===
            nothing
        @test snapshot.dcegm.query.min_lower_income_margin === nothing
        @test snapshot.dcegm.query.min_upper_income_margin === nothing
        @test _finite_or_nothing_diagnostic(snapshot.dcegm)

        finite_margin = policies.dcegm.solution.solve_diagnostics.
            min_continuation_lower_wealth_margin
        @test isfinite(finite_margin)
        @test reinterpret(UInt64,
            snapshot.dcegm.solve.min_continuation_lower_wealth_margin) ==
            reinterpret(UInt64, finite_margin)
        @test PA._checkpoint_diagnostic(Inf, :test_unavailable_margin;
            positive_infinity_is_unavailable=true) === nothing
        @test_throws ErrorException PA._checkpoint_diagnostic(
            Inf, :test_positive_infinity)
        @test_throws ErrorException PA._checkpoint_diagnostic(
            -Inf, :test_negative_infinity)
        @test_throws ErrorException PA._checkpoint_diagnostic(
            -Inf, :test_negative_infinity;
            positive_infinity_is_unavailable=true)
        @test_throws ErrorException PA._checkpoint_diagnostic(NaN, :test_nan)

        solve_diagnostics = policies.dcegm.solution.solve_diagnostics
        original_maximum = solve_diagnostics.max_continuation_wealth_extrapolation
        try
            solve_diagnostics.max_continuation_wealth_extrapolation = NaN
            @test_throws ErrorException PA.iid_contest_diagnostics(
                policies.dcegm)
        finally
            solve_diagnostics.max_continuation_wealth_extrapolation =
                original_maximum
        end
        try
            solve_diagnostics.max_continuation_wealth_extrapolation = Inf
            @test_throws ErrorException PA.iid_contest_diagnostics(
                policies.dcegm)
        finally
            solve_diagnostics.max_continuation_wealth_extrapolation =
                original_maximum
        end
        original_minimum = solve_diagnostics.min_continuation_lower_wealth_margin
        try
            solve_diagnostics.min_continuation_lower_wealth_margin = -Inf
            @test_throws ErrorException PA.iid_contest_diagnostics(
                policies.dcegm)
        finally
            solve_diagnostics.min_continuation_lower_wealth_margin =
                original_minimum
        end

        mktemp() do path, io
            close(io)
            JLD2.jldsave(path; diagnostics=snapshot)
            loaded = JLD2.load(path, "diagnostics")
            @test loaded == snapshot
        end

        reset = PA.reset_iid_contest_diagnostics!(policies)
        @test reset.ptd_sceq.cache_entries == 1
        @test reset.ptd_sceq.verified_cache_entries == 1
        @test reset.ptd_sceq.cache_hits == 0
        @test reset.ptd_sceq.cache_misses == 0
        @test reset.ptd_osa.cache_entries == 1
        @test reset.ptd_osa.verified_cache_entries == 1
        @test reset.ptd_osa.calls == 0
        @test reset.dcegm.solve.quadrature_expectations ==
            snapshot.dcegm.solve.quadrature_expectations
        @test reset.dcegm.query.consumption_queries == 0
        @test reset.dcegm.query.value_queries == 0

        stale = PA.build_iid_contest_policy(:dcegm, config, process;
            asset_grid=test_grid, base_rule=base_rule)
        PA.DcEgm.solve_egm!(stale.solution.agent; quadrature=stale.egm_rule)
        @test_throws ErrorException stale(state)
    end
end
