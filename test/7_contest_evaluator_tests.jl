# Tests for src/simulation/common/contest_evaluator.jl and source-order file 7.
using Test
import Ptd

const CE = Ptd

consume_all(method) = observed -> CE.ActionResult(
    observed.cash, 0.0, 0.0, method, "test")

mutable struct ScopedConsumeAll
    method::Symbol
end

(policy::ScopedConsumeAll)(observed) = CE.ActionResult(
    observed.cash, 0.0, 0.0, policy.method, "scoped")

@testset "Hidden-path expected-utility evaluator" begin
    config = CE.StochasticPTDConfig(beta=0.9, gamma=1.0,
        return_rate=0.0, cash_floor=1.0, terminal_date=3)
    process = CE.IIDLognormalProcess(sigma=0.0)
    initial = CE.ObservedState(1, 3.0, CE.IIDState())
    innovations = [-9.0 -8.0; 7.0 6.0]
    observed_arguments = CE.ObservedState[]

    sceq_action = observed -> begin
        push!(observed_arguments, observed)
        CE.ActionResult(observed.cash, 0.0, 0.0, :ptd_sceq, "test")
    end
    evaluation = CE.evaluate_policy_paths(
        sceq_action, :ptd_sceq, config, process, initial, innovations)

    # With sigma zero, both paths give x=(3,1,1) under consume-all.
    expected = log(3.0) + 0.9 * log(1.0) + 0.9^2 * log(1.0)
    @test evaluation.utility == fill(expected, 2)
    @test evaluation.floor_hits == [0, 0]
    @test length(observed_arguments) == 6
    @test all(argument isa CE.ObservedState for argument in observed_arguments)

    trajectory_rows = CE.TrajectoryRow[]
    traced = CE.evaluate_policy_paths(
        consume_all(:ptd_sceq), :ptd_sceq, config, process, initial,
        innovations; trajectory_rows, trajectory_path_offset=10)
    @test length(trajectory_rows) == 6
    @test getfield.(trajectory_rows, :path) == [11, 11, 11, 12, 12, 12]
    @test getfield.(getfield.(trajectory_rows, :observed), :date) ==
        [1, 2, 3, 1, 2, 3]
    @test all(isimmutable, trajectory_rows)
    @test all(!isnothing(row.transition) for row in trajectory_rows[[1, 2, 4, 5]])
    @test all(isnothing(row.transition) for row in trajectory_rows[[3, 6]])
    for path in 11:12
        rows = filter(row -> row.path == path, trajectory_rows)
        @test sum(getfield.(rows, :discounted_utility)) ==
            traced.utility[path - 10]
    end
    @test_throws ArgumentError CE.evaluate_policy_paths(
        consume_all(:ptd_sceq), :ptd_sceq, config, process, initial,
        innovations; trajectory_rows=CE.TrajectoryRow[],
        trajectory_path_offset=-1)

    common = CE.evaluate_contest_paths(
        (dcegm=consume_all(:dcegm),
         ptd_osa=consume_all(:ptd_osa),
         ptd_sceq=sceq_action),
        config, process, initial, innovations)
    @test common.method_ids == [:ptd_sceq, :ptd_osa, :dcegm]
    @test common.utility[:, 1] == common.utility[:, 2] == common.utility[:, 3]
    @test size(common.utility) == (2, 3)

    bad_method = consume_all(:dcegm)
    @test_throws ArgumentError CE.evaluate_policy_paths(
        bad_method, :ptd_sceq, config, process, initial, innovations)
    @test_throws DimensionMismatch CE.evaluate_policy_paths(
        sceq_action, :ptd_sceq, config, process, initial, zeros(2, 1))
    @test_throws ArgumentError CE.evaluate_contest_paths(
        (ptd_sceq=sceq_action, ptd_osa=consume_all(:ptd_osa)),
        config, process, initial, innovations)

    # Innovations in column q are revealed only after action date q. With
    # consume-all, next cash is exactly max(next income, floor), making timing
    # and the floor transfer independently transparent.
    timing_config = CE.StochasticPTDConfig(beta=0.8, gamma=2.0,
        return_rate=0.25, cash_floor=0.5, terminal_date=3)
    timing_process = CE.IIDLognormalProcess(sigma=1.0)
    timing_initial = CE.ObservedState(1, 2.0, CE.IIDState())
    timing_shocks = reshape([-2.0, 1.0], 1, 2)
    timing_states = CE.ObservedState[]
    timing_action = observed -> begin
        push!(timing_states, observed)
        CE.ActionResult(observed.cash, 0.0, 0.0, :ptd_sceq, "timing")
    end
    timing = CE.evaluate_policy_paths(timing_action, :ptd_sceq,
        timing_config, timing_process, timing_initial, timing_shocks)
    income2 = exp(-0.5 - 2.0)
    income3 = exp(-0.5 + 1.0)
    cash2 = max(income2, 0.5)
    cash3 = max(income3, 0.5)
    u_gamma_two(c) = 1.0 - inv(c)
    manual = u_gamma_two(2.0) +
        0.8 * u_gamma_two(cash2) +
        0.8^2 * u_gamma_two(cash3)
    @test getfield.(timing_states, :cash) ≈ [2.0, cash2, cash3]
    @test timing.utility[1] ≈ manual
    @test timing.floor_hits == [1]
    @test timing.total_transfers[1] ≈ 0.5 - income2

    # Zero consumption is admissible. It has finite CRRA utility for gamma<1
    # and the correct extended-real value for gamma>=1.
    zero_then_terminal(method) = observed -> begin
        terminal = observed.date == 2
        consumption = terminal ? observed.cash : 0.0
        CE.ActionResult(consumption, observed.cash - consumption,
            0.0, method, "zero")
    end
    low_gamma = CE.StochasticPTDConfig(beta=0.9, gamma=0.5,
        return_rate=0.0, cash_floor=1.0, terminal_date=2)
    zero_finite = CE.evaluate_policy_paths(zero_then_terminal(:ptd_sceq),
        :ptd_sceq, low_gamma, process, initial, zeros(1, 1))
    @test only(zero_finite.utility) ≈ -0.2
    high_gamma = CE.StochasticPTDConfig(beta=0.9, gamma=1.0,
        return_rate=0.0, cash_floor=1.0, terminal_date=2)
    zero_infinite = CE.evaluate_policy_paths(zero_then_terminal(:ptd_sceq),
        :ptd_sceq, high_gamma, process, initial, zeros(1, 1))
    @test only(zero_infinite.utility) == -Inf

    # Starting at T requires a paths-by-zero-shocks matrix and evaluates only
    # terminal utility, with no hidden draw or extra discount factor.
    terminal_initial = CE.ObservedState(3, 2.5, CE.IIDState())
    terminal_only = CE.evaluate_policy_paths(consume_all(:ptd_sceq),
        :ptd_sceq, config, process, terminal_initial, zeros(2, 0))
    @test terminal_only.utility == fill(log(2.5), 2)
    @test terminal_only.floor_hits == [0, 0]
    @test_throws DimensionMismatch CE.evaluate_policy_paths(
        consume_all(:ptd_sceq), :ptd_sceq, config, process,
        terminal_initial, zeros(2, 1))

    inconsistent = observed -> CE.ActionResult(
        observed.cash / 2, 0.0, 0.0, :ptd_sceq, "bad-saving")
    @test_throws ArgumentError CE.evaluate_policy_paths(inconsistent,
        :ptd_sceq, config, process, initial, innovations)
    nonterminal_at_terminal = observed -> CE.ActionResult(
        observed.cash / 2, observed.cash / 2, 0.0, :ptd_sceq, "bad-terminal")
    @test_throws ArgumentError CE.evaluate_policy_paths(nonterminal_at_terminal,
        :ptd_sceq, config, process, initial, innovations)

    # A stateful callback that changes its initial action by hidden path is
    # rejected at the repeated initial state (same state must mean same action).
    calls = Ref(0)
    anticipative_by_order = observed -> begin
        calls[] += 1
        consumption = calls[] == 1 ? observed.cash : observed.cash / 2
        CE.ActionResult(consumption, observed.cash - consumption,
            0.0, :ptd_sceq, "leaky")
    end
    @test_throws ArgumentError CE.evaluate_policy_paths(anticipative_by_order,
        :ptd_sceq, config, process, initial, innovations)

    ar1 = CE.LogAR1Process(rho=0.8, sigma=0.1)
    ar1_initial = CE.ObservedState(2, 2.0, CE.LogAR1State(0.0))
    one_step_config = CE.StochasticPTDConfig(beta=0.9, gamma=2.0,
        return_rate=0.0, cash_floor=1.0, terminal_date=3)
    @test CE.allowed_stochastic_methods(ar1) ==
        (:ptd_sceq, :ptd_osa, :g2egm)
    @test length(CE.evaluate_policy_paths(consume_all(:g2egm), :g2egm,
        one_step_config, ar1, ar1_initial, zeros(2, 1)).utility) == 2
    @test_throws ArgumentError CE.evaluate_contest_paths(
        (ptd_sceq=consume_all(:ptd_sceq),
         ptd_osa=consume_all(:ptd_osa), g2egm=consume_all(:g2egm)),
        one_step_config, ar1, ar1_initial, zeros(2, 1))

    hsz = CE.HSZProcess(rho_earnings=0.9, sigma_earnings=0.1,
        rho_medical=0.8, sigma_medical=0.1,
        unconditional_log_earnings=log(2.0),
        unconditional_log_medical=log(0.5))
    hsz_initial = CE.ObservedState(
        2, 2.0, CE.HSZState(log(2.0), log(0.5)))
    hsz_action = consume_all(:ptd_osa)
    hsz_result = CE.evaluate_policy_paths(hsz_action, :ptd_osa,
        one_step_config, hsz, hsz_initial, zeros(2, 1, 2))
    @test length(hsz_result.utility) == 2
    @test CE.allowed_stochastic_methods(process) ==
        (:ptd_sceq, :ptd_osa, :dcegm)
    @test CE.allowed_stochastic_methods(hsz) ==
        (:ptd_sceq, :ptd_osa, :g2egm)
    hsz_g2_result = CE.evaluate_policy_paths(consume_all(:g2egm),
        :g2egm, one_step_config, hsz, hsz_initial, zeros(2, 1, 2))
    @test length(hsz_g2_result.utility) == 2
    @test_throws ArgumentError CE.evaluate_contest_paths(
        (ptd_sceq=consume_all(:ptd_sceq),
         ptd_osa=hsz_action, g2egm=consume_all(:g2egm)),
        one_step_config, hsz, hsz_initial, zeros(2, 1, 2))
    @test_throws DimensionMismatch CE.evaluate_policy_paths(hsz_action, :ptd_osa,
        one_step_config, hsz, hsz_initial, zeros(2, 1))
    @test_throws ArgumentError CE.evaluate_policy_paths(consume_all(:ptd_sceq),
        :ptd_sceq, config, process,
        CE.ObservedState(1, 3.0, CE.LogAR1State(0.0)), innovations)

    @testset "persistent nonanticipativity audit across ten batches" begin
        audit_config = CE.StochasticPTDConfig(beta=0.93, gamma=1.7,
            return_rate=0.02, cash_floor=1.0, terminal_date=3)
        audit_process = CE.IIDLognormalProcess(sigma=0.25)
        audit_initial = CE.ObservedState(1, 4.0, CE.IIDState())
        hidden = reshape([sin(Float64(index)) for index in 1:40], 20, 2)
        stable_policy = observed -> begin
            terminal = observed.date == audit_config.terminal.final_date
            consumption = terminal ? observed.cash : 0.6 * observed.cash
            CE.ActionResult(consumption, observed.cash - consumption,
                0.0, :ptd_sceq, "stable-batch-policy")
        end

        monolithic = CE.evaluate_policy_paths(stable_policy, :ptd_sceq,
            audit_config, audit_process, audit_initial, hidden)
        persistent = CE.PolicyActionAudit()
        batches = [CE.evaluate_policy_paths(stable_policy, :ptd_sceq,
            audit_config, audit_process, audit_initial,
            @view(hidden[2index-1:2index, :]); action_audit=persistent)
            for index in 1:10]
        batched_utility = reduce(vcat, getfield.(batches, :utility))
        batched_floor_hits = reduce(vcat, getfield.(batches, :floor_hits))
        batched_transfers = reduce(vcat, getfield.(batches, :total_transfers))

        @test reinterpret(UInt64, batched_utility) ==
            reinterpret(UInt64, monolithic.utility)
        @test batched_floor_hits == monolithic.floor_hits
        @test reinterpret(UInt64, batched_transfers) ==
            reinterpret(UInt64, monolithic.total_transfers)
        @test persistent.bound
        @test persistent.method == :ptd_sceq
        @test persistent.initial_state_type === typeof(audit_initial)
        @test !isempty(persistent.initial_state_token)
        @test length(persistent.state_actions) >= 1

        # Each one-path call sees the initial state only once. The default
        # call-local map cannot compare the two calls, while one persistent
        # audit rejects the changed initial action in the second batch.
        initial_calls = Ref(0)
        changing_policy = observed -> begin
            if observed.date == audit_initial.date
                initial_calls[] += 1
                consumption = initial_calls[] == 1 ?
                    0.6 * observed.cash : 0.5 * observed.cash
            else
                consumption = observed.date ==
                    audit_config.terminal.final_date ? observed.cash :
                    0.6 * observed.cash
            end
            CE.ActionResult(consumption, observed.cash - consumption,
                0.0, :ptd_sceq, "changing-batch-policy")
        end
        changing_audit = CE.PolicyActionAudit()
        CE.evaluate_policy_paths(changing_policy, :ptd_sceq,
            audit_config, audit_process, audit_initial, hidden[1:1, :];
            action_audit=changing_audit)
        @test_throws ArgumentError CE.evaluate_policy_paths(
            changing_policy, :ptd_sceq, audit_config, audit_process,
            audit_initial, hidden[2:2, :]; action_audit=changing_audit)

        scoped = ScopedConsumeAll(:ptd_sceq)
        scope_audit = CE.PolicyActionAudit()
        CE.evaluate_policy_paths(scoped, :ptd_sceq, audit_config,
            audit_process, audit_initial, hidden[1:1, :];
            action_audit=scope_audit)
        @test_throws ArgumentError CE.evaluate_policy_paths(
            ScopedConsumeAll(:ptd_sceq), :ptd_sceq, audit_config,
            audit_process, audit_initial, hidden[2:2, :];
            action_audit=scope_audit)
        @test_throws ArgumentError CE.evaluate_policy_paths(scoped,
            :ptd_sceq, audit_config, audit_process,
            CE.ObservedState(1, 4.5, CE.IIDState()), hidden[2:2, :];
            action_audit=scope_audit)
        scoped.method = :ptd_osa
        @test_throws ArgumentError CE.evaluate_policy_paths(scoped, :ptd_osa,
            audit_config, audit_process, audit_initial, hidden[2:2, :];
            action_audit=scope_audit)

        state_scope = ScopedConsumeAll(:ptd_osa)
        state_audit = CE.PolicyActionAudit()
        CE.evaluate_policy_paths(state_scope, :ptd_osa, audit_config,
            audit_process, audit_initial, hidden[1:1, :];
            action_audit=state_audit)
        ar1_scope_process = CE.LogAR1Process(rho=0.8, sigma=0.25)
        ar1_scope_initial = CE.ObservedState(
            1, 4.0, CE.LogAR1State(0.0))
        @test_throws ArgumentError CE.evaluate_policy_paths(state_scope,
            :ptd_osa, audit_config, ar1_scope_process, ar1_scope_initial,
            hidden[2:2, :]; action_audit=state_audit)
    end
end
