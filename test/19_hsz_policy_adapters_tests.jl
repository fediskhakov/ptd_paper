# Tests for src/simulation/policies/hsz.jl and source-order file 19.
@testset "HSZ example policy adapters" begin
    process = Ptd.HSZProcess(rho_earnings=0.9, sigma_earnings=0.1,
        rho_medical=0.9, sigma_medical=0.1,
        unconditional_log_earnings=log(2.0),
        unconditional_log_medical=log(0.5))
    config = Ptd.StochasticPTDConfig(beta=0.98, gamma=1.0,
        return_rate=0.1, cash_floor=3.0, terminal_date=2)
    rule = Ptd.gauss_hermite_rule(process, 5)
    observed = Ptd.ObservedState(1, 10.0,
        Ptd.HSZState(log(2.0), log(0.5)))
    grid = Ptd.G2EGM.hsz_g2egm_grid(process; asset_maximum=30.0,
        asset_points=9, common_cash_points=11,
        post_earnings_points=5, post_medical_points=5,
        common_earnings_points=5, common_medical_points=5)
    policies = Dict(
        :ptd_sceq => Ptd.build_hsz_contest_policy(
            :ptd_sceq, config, process; base_rule=rule),
        :ptd_osa => Ptd.build_hsz_contest_policy(
            :ptd_osa, config, process; base_rule=rule),
        :g2egm => Ptd.build_hsz_contest_policy(
            :g2egm, config, process; base_rule=rule, g2egm_grid=grid))
    for method in (:ptd_sceq, :ptd_osa, :g2egm)
        action = policies[method](observed)
        @test action.method == method
        @test 0.0 <= action.consumption <= observed.cash
        @test isnothing(Ptd.validate_hsz_contest_policy_alignment(
            policies[method], method, config, process))
    end
    @test Ptd.hsz_contest_quadrature_audit(
        policies[:ptd_sceq]).uses_quadrature == false
    @test Ptd.hsz_contest_quadrature_audit(
        policies[:ptd_osa]).total_nodes == 25
end
