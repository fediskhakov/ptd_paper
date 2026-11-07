# Tests for the plan cache in src/ptd/ptd_sceq.jl.
@testset verbose = true "External immutable PTD plan cache" begin
    config = Ptd.StochasticPTDConfig(beta=0.96, gamma=2.0,
        return_rate=0.02, cash_floor=1.0, terminal_date=5)
    iid = Ptd.IIDLognormalProcess(sigma=0.1)
    observed = Ptd.ObservedState(1, 3.0, Ptd.IIDState())

    @testset "PTD-SCEQ cache hits preserve the action exactly" begin
        cache = Ptd.PTDPlanCache()
        @test Ptd.plan_cache_statistics(cache) ==
              (entries=0, hits=0, misses=0, builds=0)

        uncached = Ptd.ptd_sceq(config, iid, observed)
        cached_first = Ptd.ptd_sceq(config, iid, observed; cache=cache)
        first_stats = Ptd.plan_cache_statistics(cache)
        cached_second = Ptd.ptd_sceq(config, iid, observed; cache=cache)
        second_stats = Ptd.plan_cache_statistics(cache)

        for field in fieldnames(Ptd.ActionResult)
            @test getfield(cached_first, field) == getfield(uncached, field)
            @test getfield(cached_second, field) == getfield(uncached, field)
        end
        @test first_stats == (entries=1, hits=0, misses=1, builds=1)
        @test second_stats == (entries=1, hits=1, misses=1, builds=1)

        path = Ptd.nominal_path(config, iid, observed)
        plan_first = Ptd.cached_ptd_plan!(cache, config, path)
        plan_second = Ptd.cached_ptd_plan!(cache, config, path)
        @test plan_first === plan_second
        @test Ptd.plan_request_fingerprint(config, path) ==
              Ptd.plan_request_fingerprint(config, path)

        trusted_cache = Ptd.PTDPlanCache(verify_integrity_on_hit=false)
        trusted_first = Ptd.ptd_sceq(
            config, iid, observed; cache=trusted_cache)
        trusted_second = Ptd.ptd_sceq(
            config, iid, observed; cache=trusted_cache)
        for field in fieldnames(Ptd.ActionResult)
            @test isequal(getfield(trusted_first, field),
                getfield(uncached, field))
            @test isequal(getfield(trusted_second, field),
                getfield(uncached, field))
        end
        @test Ptd.verify_plan_cache_integrity!(trusted_cache) == 1
    end

    @testset "persistent conditioning states have distinct requests" begin
        cache = Ptd.PTDPlanCache()
        ar1 = Ptd.LogAR1Process(rho=0.8, sigma=0.1)
        low = Ptd.ObservedState(2, 2.0, Ptd.LogAR1State(log(0.8)))
        high = Ptd.ObservedState(2, 2.0, Ptd.LogAR1State(log(1.2)))
        low_path = Ptd.nominal_path(config, ar1, low)
        high_path = Ptd.nominal_path(config, ar1, high)
        @test low_path.fingerprint != high_path.fingerprint
        @test Ptd.plan_request_fingerprint(config, low_path) !=
              Ptd.plan_request_fingerprint(config, high_path)

        Ptd.cached_ptd_plan!(cache, config, low_path)
        Ptd.cached_ptd_plan!(cache, config, high_path)
        @test Ptd.plan_cache_statistics(cache) ==
              (entries=2, hits=0, misses=2, builds=2)
    end

    @testset "PTD-OSA node tails reuse one complete IID nominal plan" begin
        rule = Ptd.gauss_hermite_rule(5)
        uncached = Ptd.ptd_osa_with_diagnostics(
            config, iid, observed, rule)
        cache = Ptd.PTDPlanCache()
        cached_first = Ptd.ptd_osa_with_diagnostics(
            config, iid, observed, rule; cache=cache)
        first_stats = Ptd.plan_cache_statistics(cache)
        cached_second = Ptd.ptd_osa_with_diagnostics(
            config, iid, observed, rule; cache=cache)
        second_stats = Ptd.plan_cache_statistics(cache)

        for field in fieldnames(Ptd.ActionResult)
            @test getfield(cached_first.action, field) ==
                  getfield(uncached.action, field)
            @test getfield(cached_second.action, field) ==
                  getfield(uncached.action, field)
        end
        for field in fieldnames(Ptd.PTDOSADiagnostics)
            @test getfield(cached_first.diagnostics, field) ==
                  getfield(uncached.diagnostics, field)
            @test getfield(cached_second.diagnostics, field) ==
                  getfield(uncached.diagnostics, field)
        end
        @test first_stats == (entries=1, hits=4, misses=1, builds=1)
        @test second_stats == (entries=1, hits=9, misses=1, builds=1)

        reset = Ptd.reset_plan_cache_statistics!(cache)
        @test reset == (entries=1, hits=0, misses=0, builds=0)
        Ptd.cached_ptd_plan!(cache, config,
            Ptd.nominal_path(config, iid,
                Ptd.ObservedState(2, 1.0, Ptd.IIDState())))
        @test Ptd.plan_cache_statistics(cache).hits == 1
        @test Ptd.clear_plan_cache!(cache) ==
              (entries=0, hits=0, misses=0, builds=0)
    end
end
