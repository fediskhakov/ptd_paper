"""
Pure-Julia, problem-specific adaptations of the common-grid reconstruction in
Druedahl and Jørgensen's G2EGM algorithm: triangles for scalar AR(1), and
tetrahedra for the two-state HSZ process.

Saving is the sole continuous choice in both models. Local simplexes and the
postdecision interpolant screen interior candidates; final winners and every
constraint, floor-kink, and repair
candidate are revalued with the model's common fresh quadrature operator on the
child common grid before the upper envelope is stored.
"""

using SHA

import ..Ptd: GaussHermiteRule,
              HSZProcess,
              HSZState,
              LogAR1Process,
              LogAR1State,
              ObservedState,
              StochasticPTDConfig,
              gauss_hermite_rule

export AR1G2EGMGrid,
       AR1G2EGMOptions,
       AR1G2EGMSolution,
       G2EGMSolveDiagnostics,
       G2EGMQueryDiagnostics,
       G2EGM_REFERENCE_COMMIT,
       G2EGM_SOLVER_SCHEMA,
       ar1_g2egm_grid,
       build_ar1_solution,
       action,
       consumption,
       value_at,
       validate_alignment,
       diagnostics,
       reset_query_diagnostics!,
       quadrature_audit,
       solution_fingerprint

export HSZG2EGMGrid,
       HSZG2EGMOptions,
       HSZG2EGMSolution,
       HSZG2EGMSolveDiagnostics,
       HSZG2EGMQueryDiagnostics,
       HSZ_G2EGM_SOLVER_SCHEMA,
       hsz_g2egm_grid,
       build_hsz_solution

const G2EGM_REFERENCE_COMMIT =
    "30aa550d8319582f9a17e08192dc8f1361726c9b"
const G2EGM_SOLVER_SCHEMA = "ar1-g2egm-exact-cell-repair-v4"
const HSZ_G2EGM_SOLVER_SCHEMA = "hsz-tetrahedral-g2egm-exact-cell-repair-v2"

include("local_envelope.jl")
include("tetra_envelope.jl")
include("ar1_model.jl")
include("ar1_solver.jl")
include("hsz_model.jl")
include("hsz_solver.jl")
