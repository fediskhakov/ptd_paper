"""Pure, nonanticipative adapters for the two-state HSZ diagnostic."""
abstract type AbstractHSZContestPolicy end

"""PTD-SCEQ policy adapter for the HSZ exercise."""
struct PTDSCEQHSZContestPolicy <: AbstractHSZContestPolicy
    config::StochasticPTDConfig
    process::HSZProcess
    cache::PTDPlanCache
end

"""PTD-OSA policy adapter for the HSZ exercise."""
struct PTDOSAHSZContestPolicy <: AbstractHSZContestPolicy
    config::StochasticPTDConfig
    process::HSZProcess
    base_rule::GaussHermiteRule{2,25}
    options::PTDOSAOptions
    cache::PTDPlanCache
end

"""Solved G2EGM policy adapter for the HSZ exercise."""
struct G2EGMHSZContestPolicy{S} <: AbstractHSZContestPolicy
    config::StochasticPTDConfig
    process::HSZProcess
    base_rule::GaussHermiteRule{2,25}
    solution::S
    policy_fingerprint::String
end

function _validate_hsz_rule(process::HSZProcess, rule)
    rule isa GaussHermiteRule{2,25} || throw(ArgumentError(
        "the HSZ diagnostic requires exactly 5x5=25 quadrature nodes"))
    expected = gauss_hermite_rule(process, 5)
    rule.nodes == expected.nodes && rule.weights == expected.weights ||
        throw(ArgumentError("the HSZ rule is not canonical Q5xQ5"))
    return rule::GaussHermiteRule{2,25}
end

"""Construct the requested policy adapter for the HSZ exercise."""
function build_hsz_contest_policy(method::Symbol,
    config::StochasticPTDConfig, process::HSZProcess;
    base_rule=gauss_hermite_rule(process, 5), g2egm_grid=nothing,
    g2egm_options=G2EGM.HSZG2EGMOptions(), ptd_osa_options=PTDOSAOptions())
    rule = _validate_hsz_rule(process, base_rule)
    if method === :ptd_sceq
        return PTDSCEQHSZContestPolicy(config, process,
            PTDPlanCache(verify_integrity_on_hit=false))
    elseif method === :ptd_osa
        return PTDOSAHSZContestPolicy(config, process, rule, ptd_osa_options,
            PTDPlanCache(verify_integrity_on_hit=false))
    elseif method === :g2egm
        solution = G2EGM.build_hsz_solution(config, process;
            base_rule=rule, grid=g2egm_grid, options=g2egm_options)
        return G2EGMHSZContestPolicy(config, process, rule, solution,
            G2EGM.solution_fingerprint(solution))
    end
    throw(ArgumentError("unknown HSZ diagnostic method $method"))
end

function (policy::PTDSCEQHSZContestPolicy)(observed::ObservedState{HSZState})
    return ptd_sceq(policy.config, policy.process, observed;
        cache=policy.cache, evaluate_objective=false)
end

function (policy::PTDOSAHSZContestPolicy)(observed::ObservedState{HSZState})
    return ptd_osa(policy.config, policy.process, observed, policy.base_rule;
        options=policy.options, cache=policy.cache)
end

function (policy::G2EGMHSZContestPolicy)(observed::ObservedState{HSZState})
    c = G2EGM.action(policy.solution, observed)
    return ActionResult(c, observed.cash - c, nothing, :g2egm,
        policy.policy_fingerprint)
end

"""Validate that an HSZ policy matches the declared model and process."""
function validate_hsz_contest_policy_alignment(policy::AbstractHSZContestPolicy,
    method::Symbol, config::StochasticPTDConfig, process::HSZProcess)
    policy.config == config && policy.process == process || throw(ArgumentError(
        "HSZ policy does not match the requested model"))
    expected = policy isa PTDSCEQHSZContestPolicy ? :ptd_sceq :
        policy isa PTDOSAHSZContestPolicy ? :ptd_osa : :g2egm
    method === expected || throw(ArgumentError(
        "HSZ policy must be registered as $expected"))
    if policy isa PTDOSAHSZContestPolicy
        _validate_hsz_rule(process, policy.base_rule)
    elseif policy isa G2EGMHSZContestPolicy
        _validate_hsz_rule(process, policy.base_rule)
        G2EGM.validate_alignment(policy.solution, config, process,
            policy.base_rule)
    end
    return nothing
end

"""Describe the quadrature rule used by an HSZ contest policy."""
function hsz_contest_quadrature_audit(policy::AbstractHSZContestPolicy)
    if policy isa PTDSCEQHSZContestPolicy
        return (; uses_quadrature=false, dimensions=2,
            nodes_per_dimension=0, total_nodes=0)
    end
    rule = policy.base_rule
    return (; uses_quadrature=true, family=:positive_gauss_hermite,
        dimensions=2, nodes_per_dimension=rule.nodes_per_dimension,
        total_nodes=length(rule.weights), nodes=rule.nodes,
        weights=rule.weights, weight_sum=sum(rule.weights))
end
