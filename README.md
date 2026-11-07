# Policy-Tree Descendant Algorithm

Julia code and saved results for:

> **Policy-Tree Descendant Algorithm for Solving Consumption–Saving Problems
> with a Safety Net**
> Wending Liu and Fedor Iskhakov

The paper develops the policy-tree descendant (PTD) algorithm for life-cycle
consumption-saving problems with a consumption floor. The code implements the
deterministic algorithm and the paper's two stochastic methods:

- **PTD-SCEQ**: PTD with certainty-equivalent path
  simulation;
- **PTD-OSA**: PTD with one-step-ahead adaptation.

The paper compares them with DC-EGM in the i.i.d. exercise and with G2EGM in
the AR(1) and HSZ exercises.

## How to replicate

Run the commands from the repository root. The code was tested on Linux
(WSL2); the experiments need up to about 31 GB of memory.

1. **Install Julia 1.12.7.** Install and
   select it with [Juliaup](https://github.com/JuliaLang/juliaup):

   ```sh
   juliaup add 1.12.7
   julia +1.12.7 --version  # confirm it reports 1.12.7
   ```

   You can skip this step if `julia --version` already reports 1.12.7. In that
   case, replace `julia +1.12.7` below with `julia`.

2. **Install dependencies and run the tests.**

   ```sh
   julia +1.12.7 --project=. -e 'using Pkg; Pkg.instantiate()'
   julia +1.12.7 --project=. -e 'using Pkg; Pkg.test(; julia_args=["--threads=1"])'
   ```

   The pinned `Manifest.toml` fixes the package versions.
   [DrWatson.jl](https://juliadynamics.github.io/DrWatson.jl/stable/) is used
   in the paper scripts for project activation and project-relative paths.

3. **To generate the tables and plots without rerunning the experiments,
   use the saved results.**

   ```sh
   julia +1.12.7 --threads=1 --project=. scripts/1_ptd_visualization.jl
   julia +1.12.7 --project=. scripts/2_deterministic.jl --render-only
   julia +1.12.7 --project=. scripts/3_iid.jl --render-only data/iid-contest.jld2 --timing-artifact data/iid-random-timings.jld2
   julia +1.12.7 --project=. scripts/4_ar1.jl --render-only data/ar1-contest-welfare.jld2 data/ar1-contest.jld2 --timing-artifact data/ar1-preference-timings.jld2
   julia +1.12.7 --project=. scripts/5_hsz.jl --render-only --timing-artifact data/hsz-path-timings.jld2
   julia +1.12.7 --project=. scripts/5_hsz.jl --refinement --render-only
   julia +1.12.7 --threads=1 --project=. scripts/6_multibranch.jl
   ```

   The tables are written to `Tables/` and the plots to `Graphs/`, which
   hold the versions used in the paper; on the test machine the commands
   reproduce them byte for byte, so `git status` stays clean. Scripts 2–5
   render the saved experiments; scripts 1 and 6 recompute their
   illustrations. For new AR(1) path trajectories, replace `--render-only`
   with `--update-paths` in the script 4 command.

4. **For fresh replication, rerun the accuracy experiments.**

   ```sh
   julia +1.12.7 --threads=24 --project=. scripts/2_deterministic.jl
   julia +1.12.7 --threads=24 --project=. scripts/3_iid.jl
   julia +1.12.7 --threads=24 --project=. scripts/4_ar1.jl
   julia +1.12.7 --threads=24 --project=. scripts/5_hsz.jl --run
   ```

   The paper's accuracy experiments used 24 Julia threads. Replace
   `--threads=24` with a thread count suited to your machine. Scripts 3 and
   4 refuse to start while
   `git status` shows uncommitted changes, so commit the results of the
   previous commands first.

5. **For fresh replication, measure single-thread speed.**

   ```sh
   julia +1.12.7 --threads=1 --project=. scripts/2_deterministic.jl --timing-only --timing-output data/deterministic-timings-new.jld2
   julia +1.12.7 --threads=1 --project=. scripts/3_iid.jl --timing-only data/iid-contest.jld2 --timing-output data/iid-timings-new.jld2
   julia +1.12.7 --threads=1 --project=. scripts/4_ar1.jl --timing-only data/ar1-contest-welfare.jld2 --timing-output data/ar1-timings-new.jld2
   julia +1.12.7 --threads=1 --project=. scripts/5_hsz.jl --timing-only --output data/hsz-timings-new.jld2
   ```

   Each command uses one Julia thread and one BLAS thread, excluding warmup.
   Timing uses 1,000 deterministic draws, 100 IID agents per variance panel,
   five AR(1) preference settings, and 100 HSZ paths with a shared policy.
   HSZ mean/median times include the same construction cost for each path.

   Timing outputs must be new files. To update the paper tables, repeat the
   rendering commands in step 3 with `--timing-artifact` pointing to the new
   files (add this option for script 2). Reruns produce new timings.

The scripts generate the following paper materials:

| Paper material | Script | Generated files |
|---|---|---|
| Section 3.1 illustrations | `1_ptd_visualization.jl` | `Graphs/policy_value_T.png`, `policy_value_T-1.png`, `policy_value_T-2.png` |
| Section 3.3 policy comparison | `1_ptd_visualization.jl` | `Graphs/d_policies_ptd_egm.png` |
| Section 3.3 deterministic comparison | `2_deterministic.jl` | `Tables/deterministic_performance.tex` |
| Section 4.4.1 IID comparison | `3_iid.jl` | `Tables/iid_contest_summary.tex`, `iid_contest_ce_gaps.tex` |
| Section 4.4.2 AR(1) comparison | `4_ar1.jl` | `Tables/ar1_contest_summary.tex`, `ar1_contest_ce_gaps.tex`; `Graphs/ar1_policy_surface.png`, `ar1_policy_paths.png` |
| Section 4.4.3 HSZ comparison | `5_hsz.jl` | `Tables/hsz_utility_comparison.tex`, `hsz_example_timings_total.tex`; `Graphs/hsz_policy_paths.png` |
| Section 4.4.3 grid refinement (footnote 15) | `5_hsz.jl --refinement` | `Tables/hsz_grid_refinement.tex` |
| Appendix B verification | `6_multibranch.jl` | `Tables/multibranch_verification.tex`; `Graphs/multibranch_model_b.png` |

All scripts are in `scripts/`. Scripts 2–5 combine accuracy, timing, and rendering.
Script 5's `--refinement --render-only` combines the saved 18- and 19-node checks
in `data/hsz-grid-refinement.jld2` and `data/hsz-grid-refinement-19.jld2` with
the 17-node baseline; `--refinement --run` recomputes a check and needs a fresh
`--run` first. Appendix B's verification results are in
`data/multibranch-verification.jld2`.
All numerical results are saved directly in `data/` as JLD2 files. They were
computed with pre-release versions of this code; before release, local paths
and development commit hashes were removed from their metadata, without
changing any number.

## Repository layout

| Path | Contents |
|---|---|
| `src/Ptd.jl` | Package entry point |
| `src/ptd/` | Deterministic and stochastic PTD models and solvers |
| `src/dcegm/` | DC-EGM comparator, exposed as `Ptd.DcEgm` |
| `src/g2egm/` | G2EGM comparators, exposed as `Ptd.G2EGM` |
| `src/simulation/` | Code for the i.i.d., AR(1), and HSZ paper exercises |
| `scripts/` | Scripts that generate the paper's tables and plots |
| `test/` | Tests for the codebase |
| `data/` | JLD2 results for the paper's tables, plots, and footnotes; no subfolders |
| `Tables/` | LaTeX tables generated by the scripts |
| `Graphs/` | Plots generated by the scripts |

## EGM code references

- [Fedor Iskhakov's DC-EGM code](https://github.com/fediskhakov/dcegm) (MATLAB)
- [Florian Oswald's DC-EGM code](https://github.com/floswald/DCEGM.jl) (Julia)
- [Jeppe Druedahl's G2EGM code](https://github.com/JeppeDruedahl/G2EGM) (C++ and MATLAB)

## Citation

If you use this code, please cite:

> Wending Liu and Fedor Iskhakov (2026). “Policy-Tree Descendant Algorithm for
> Solving Consumption–Saving Problems with a Safety Net.” Working paper.

Copyright © 2026 Wending Liu and Fedor Iskhakov. Released under the MIT
License; see `LICENSE`.
