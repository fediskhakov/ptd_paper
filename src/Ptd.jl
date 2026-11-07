"""
    Ptd

Implement the policy-tree descendant algorithms and paper simulations.
"""
module Ptd

"""Absolute root of the installed `Ptd` source tree."""
const PROJECT_ROOT = pkgdir(@__MODULE__)

include("ptd/deterministic_solver.jl")
include("ptd/stochastic_model.jl")
include("ptd/ptd_sceq.jl")
include("ptd/ptd_osa.jl")
include("simulation/common/contest_design.jl")
include("simulation/common/contest_evaluator.jl")
include("simulation/common/contest_statistics.jl")
include("simulation/common/storage.jl")
include("simulation/common/reproducibility.jl")
include("simulation/common/contest_artifact.jl")
include("simulation/common/single_path_timing.jl")

export PolicySegment, DeterministicProblem, FusionOptions, u, mu, inv_mu, zNext_a, xNext_a,
    c_x, v_x, descendant_operator, pairwise_fusion_step, fusion_operator,
    deterministic_ptd!, validate_policy_collection,
    AbstractIncomeProcess, IIDState, LogAR1State, HSZState,
    IIDLognormalProcess, LogAR1Process, HSZProcess,
    ObservedState, TransitionResult, TerminalSpec, StochasticPTDConfig,
    income, innovation_dimension, true_transition, next_observed,
    GaussHermiteRule, gauss_hermite_rule, quadrature_expectation,
    NominalPath, nominal_path, PTDPlan, build_ptd_plan, plan_value,
    plan_request_fingerprint, PTDPlanCache, cached_ptd_plan!,
    plan_cache_statistics, reset_plan_cache_statistics!, clear_plan_cache!,
    verify_plan_cache_integrity!,
    marginal_value, segment_boundaries, ActionResult, TrajectoryRow,
    first_action, ptd_sceq,
    PTDOSAOptions, PTDOSADiagnostics, PTDOSAResult, ptd_osa_with_diagnostics, ptd_osa,
    ParameterBounds, ContestDesignConfig, ContestSpec, build_contest_design,
    complete_contest_design, ContestFailure, ContestDesignResult,
    ContestCompletionError, predraw_innovations, panel_seed,
    PairedComparison, ContestSummary, paired_comparison, summarize_contest,
    aggregate_outcomes,
    FINGERPRINT_SCHEMA_VERSION, canonical_token, stable_fingerprint,
    configuration_metadata, git_metadata, package_version_metadata,
    runtime_metadata, simulation_metadata,
    build_contest_artifact, publish_contest_artifact,
    load_contest_artifact, load_single_path_timing_summary,
    PolicyPathEvaluation, ContestPathEvaluation, PolicyActionAudit,
    IID_CONTEST_METHODS,
    allowed_stochastic_methods, evaluate_policy_paths, evaluate_contest_paths

# DCEGM comparator
"""
    DcEgm

Implement the DCEGM comparator used by the paper.
"""
module DcEgm
include("dcegm/simulator.jl")
export build_agent, egm!, solve_egm!, AgentD, AgentIID, AgentAR1,
    EGMSolution, EGMSolveDiagnostics, EGMQueryDiagnostics,
    EGMQuadratureRule, distribution_quadrature,
    u, c_x, xNext_a, action, value_at, reset_query_diagnostics!,
    Grids1d, Grids2d, upper_envelope,
    IIDModel, AR1Model, evaluate_paths, simulation_iid, simulation_ar1
end

"""
    G2EGM

Implement the G2EGM comparators used by the AR(1) and HSZ exercises.
"""
module G2EGM
include("g2egm/G2EGM.jl")
end
include("simulation/policies/iid.jl")
include("simulation/policies/ar1.jl")
include("simulation/policies/hsz.jl")
"""
    IID

Run, authenticate, and render the paper's IID simulation exercise.
"""
module IID
include("simulation/iid/runner.jl")
include("simulation/iid/artifact.jl")
include("simulation/iid/random_timing.jl")
include("simulation/iid/rendering.jl")
end

"""
    AR1

Run, authenticate, and render the paper's AR(1) simulation exercise.
"""
module AR1
include("simulation/ar1/runner.jl")
include("simulation/ar1/exhibits.jl")
include("simulation/ar1/exhibit_rendering.jl")
include("simulation/ar1/artifact.jl")
include("simulation/ar1/contest_path_replay.jl")
include("simulation/ar1/updated_path_exhibit.jl")
include("simulation/ar1/preference_timing.jl")
include("simulation/ar1/rendering.jl")
end

"""
    HSZ

Run and render the paper's HSZ simulation exercise.
"""
module HSZ
include("simulation/hsz/example.jl")
include("simulation/hsz/rendering.jl")
include("simulation/hsz/utility_comparison.jl")
include("simulation/hsz/grid_refinement.jl")
include("simulation/hsz/path_timing.jl")
end

export AbstractIIDContestPolicy, PTDSCEQIIDContestPolicy,
    PTDOSAIIDContestPolicy, DCEGMIIDContestPolicy, IIDContestPolicies,
    IID_DCEGM_ASSET_MAXIMUM, IID_DCEGM_DIAGNOSTIC_POINTS,
    IID_DCEGM_REPORTED_POINTS, IID_DCEGM_VALIDATION_POINTS,
    IID_DCEGM_GRID_SCHEMA, iid_dcegm_asset_grid,
    iid_contest_asset_grid, build_iid_contest_policy,
    build_iid_contest_policies, iid_contest_actions,
    iid_contest_diagnostics, reset_iid_contest_diagnostics!,
    iid_contest_quadrature_audit,
    validate_iid_contest_policy_alignment

export AbstractAR1ContestPolicy, AR1G2ContestGridConfiguration,
    PTDSCEQAR1ContestPolicy, PTDOSAAR1ContestPolicy, G2EGMAR1ContestPolicy,
    AR1ContestPolicies, pilot_ar1_g2_grid_configuration,
    candidate_ar1_g2_grid_configuration, build_ar1_contest_policy,
    build_ar1_contest_policies, ar1_contest_actions,
    ar1_contest_diagnostics, reset_ar1_contest_diagnostics!,
    ar1_contest_quadrature_audit,
    validate_ar1_contest_policy_alignment

export AbstractHSZContestPolicy, PTDSCEQHSZContestPolicy,
    PTDOSAHSZContestPolicy, G2EGMHSZContestPolicy,
    build_hsz_contest_policy, validate_hsz_contest_policy_alignment,
    hsz_contest_quadrature_audit

export DcEgm, G2EGM, IID, AR1, HSZ

"""Measure and render fresh-policy timings for fixed experiment calibrations."""
module FixedCalibrationTiming
include("simulation/common/fixed_calibration_timing.jl")
end

end
