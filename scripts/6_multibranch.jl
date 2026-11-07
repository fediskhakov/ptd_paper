using DrWatson
@quickactivate "Ptd"

"""Standalone bounded verification of the paper's multi-branch appendix example.

Run `julia --project=. scripts/6_multibranch.jl` to regenerate the appendix exhibit.
"""
module MultiBranchExample

using Roots
using Printf
using JLD2
import Optim
import Ptd

const BETA = 0.95
const ROOT_RTOL = 1e-13
const PROJECT = normpath(joinpath(@__DIR__, ".."))

struct Branch
    lower::Float64
    upper::Float64
    intercept::Float64
    slope::Float64
end

const MODEL_B_TRANSITIONS = (
    [Branch(0, 2.5, 1, 0.5), Branch(2.5, 4, -2.75, 2), Branch(4, Inf, -0.75, 1.5)],
    [Branch(0, 1, 2, 0.5), Branch(1, Inf, 0.5, 2)],
)

const TRANSITIONS = (
    [Branch(0, 1.5, 1, 0.5), Branch(1.5, 4, -1.25, 2), Branch(4, Inf, 0.75, 1.5)],
    copy(MODEL_B_TRANSITIONS[2]),
)

utility(c) = c > 0 ? 1 - inv(c) : -Inf
function transition(t, saving; transitions=TRANSITIONS)
    branches = transitions[t]
    i = findfirst(b -> b.lower <= saving < b.upper, branches)
    isnothing(i) && throw(DomainError(saving, "saving outside the transition domain"))
    b = branches[i]
    return b.intercept + b.slope * saving
end

struct Piece
    lower::Float64
    upper::Float64
    intercept::Float64
    slope::Float64
    saving_lower::Float64
    saving_upper::Float64
    source::String
end

consumption(p::Piece, x) = p.intercept + p.slope * x
restrict(p::Piece, l, r) = Piece(l, r, p.intercept, p.slope,
    p.saving_lower, p.saving_upper, p.source)
coordinate_tolerance(x) = 16eps(Float64) * max(abs(x), 1.0)

struct Solution
    lower::Float64
    upper::Vector{Float64}
    policy::Vector{Vector{Piece}}
    candidates::Vector{Vector{Piece}}
    transitions::Tuple{Vector{Branch},Vector{Branch}}
end

function policy_piece(solution::Solution, t, x)
    solution.lower <= x <= solution.upper[t] + coordinate_tolerance(x) ||
        throw(DomainError(x, "outside the declared bounded domain"))
    pieces = solution.policy[t]
    i = findfirst(p -> p.lower <= x < p.upper, pieces)
    isnothing(i) && x >= last(pieces).upper && return last(pieces)
    isnothing(i) && error("uncovered policy state $x at date $t")
    return pieces[i]
end

function value(solution::Solution, t, x)
    t == 3 && return utility(x)
    c = consumption(policy_piece(solution, t, x), x)
    return utility(c) + BETA * value(solution, t + 1,
        transition(t, max(x - c, 0.0); transitions=solution.transitions))
end

function candidate_value(solution::Solution, t, p::Piece, x)
    c = consumption(p, x)
    -coordinate_tolerance(x) <= x - c || return -Inf
    return utility(c) + BETA * value(solution, t + 1,
        transition(t, max(x - c, 0.0); transitions=solution.transitions))
end

"""CRRA inverse Euler descendants on each reachable branch/parent intersection."""
function descendants(solution::Solution, t)
    xmin, xmax = solution.lower, solution.upper[t]
    candidates = Piece[Piece(xmin, xmax, 0, 1, 0, 0, "consume-all")]
    for (j, branch) in enumerate(solution.transitions[t]), (k, parent) in enumerate(solution.policy[t + 1])
        lo = max(branch.lower, (parent.lower - branch.intercept) / branch.slope, 0.0)
        hi = min(branch.upper, (parent.upper - branch.intercept) / branch.slope)
        hi > lo || continue
        factor = inv(sqrt(BETA * branch.slope))
        # x(a) = a + (beta R)^(-1/2) c_next(y + R a).
        map_cash(a) = a + factor * consumption(parent, branch.intercept + branch.slope * a)
        lower, upper = max(xmin, map_cash(lo)), min(xmax, map_cash(hi))
        upper > lower || continue
        q = factor * parent.slope * branch.slope
        slope = q / (1 + q)
        intercept = factor * (parent.intercept + parent.slope * branch.intercept) / (1 + q)
        push!(candidates, Piece(lower, upper, intercept, slope, lo, hi, "child-$j-$k"))
    end
    # Keeping all transition corners is valid; convex corners will be eliminated
    # by value comparison, including the first transition's convex corner.
    for branch in solution.transitions[t][2:end]
        a = branch.lower
        a < xmax || continue
        push!(candidates, Piece(max(xmin, a), xmax, -a, 1, a, a, "corner-$a"))
    end
    sort!(candidates; by=p -> (p.saving_lower, p.saving_upper))
    return candidates
end

function coalesce(pieces)
    result = Piece[]
    for p in pieces
        if !isempty(result) && last(result).intercept == p.intercept && last(result).slope == p.slope
            previous = pop!(result)
            push!(result, restrict(previous, previous.lower, p.upper))
        else
            push!(result, p)
        end
    end
    return result
end

"""Transcription of alg_F.tex's endpoint cells and per-cell stack.

The final endpoint is the declared finite bound.
"""
function fuse(solution::Solution, t, candidates)
    raw = sort!(unique!([solution.lower; solution.upper[t];
        getfield.(candidates, :lower); getfield.(candidates, :upper)]))
    endpoints = Float64[]
    for x in raw
        if isempty(endpoints) || x - last(endpoints) > coordinate_tolerance(x)
            push!(endpoints, x)
        elseif x == solution.upper[t]
            endpoints[end] = x
        end
    end
    pieces = Piece[]
    for cell in 1:length(endpoints)-1
        left, right = endpoints[cell], endpoints[cell + 1]
        envelope = Piece[]
        for q in candidates
            q.lower <= left + coordinate_tolerance(left) &&
                q.upper >= right - coordinate_tolerance(right) || continue
            while !isempty(envelope)
                p = last(envelope)
                L, R = p.lower, p.upper
                gap(x) = candidate_value(solution, t, p, x) - candidate_value(solution, t, q, x)
                if gap(R) >= 0
                    break
                elseif gap(L) <= 0
                    pop!(envelope)
                else
                    switch = find_zero(gap, (L, R), Roots.Brent(); xrtol=ROOT_RTOL)
                    envelope[end] = restrict(p, L, switch)
                    break
                end
            end
            start = isempty(envelope) ? left : last(envelope).upper
            start < right && push!(envelope, restrict(q, start, right))
        end
        isempty(envelope) && error("empty envelope at date $t on [$left,$right]")
        append!(pieces, envelope)
    end
    return coalesce(pieces)
end

function solve(; xmin=0.01, xmax=40.0, transitions=TRANSITIONS)
    0 < xmin < xmax || throw(ArgumentError("require 0 < xmin < xmax"))
    # Nonnegative consumption bounds next resources by f_t(current resources).
    # Propagating the upper bound therefore contains every feasible continuation.
    upper = [Float64(xmax), transition(1, Float64(xmax); transitions)]
    push!(upper, transition(2, upper[2]; transitions))
    policy = [Piece[] for _ in 1:3]
    policy[3] = [Piece(xmin, upper[3], 0, 1, 0, 0, "terminal")]
    solution = Solution(xmin, upper, policy, [Piece[] for _ in 1:3], transitions)
    for t in 2:-1:1
        solution.candidates[t] = descendants(solution, t)
        solution.policy[t] = fuse(solution, t, solution.candidates[t])
    end
    return solution
end

"""Independent bounded maximization using Optim's derivative-free Brent method."""
function maximize_concave(objective, lo::BigFloat, hi::BigFloat)
    result = Optim.optimize(a -> -objective(a), lo, hi, Optim.Brent();
        rel_tol=sqrt(eps(BigFloat)), abs_tol=eps(BigFloat), iterations=1000)
    Optim.converged(result) || error("reference Brent optimization did not converge")
    # Optim's bounded search does not test the exact endpoints.
    candidates = (lo, Optim.minimizer(result), hi)
    values = objective.(candidates)
    return candidates[argmax(values)]
end

# These reference routines operate only on saving choices and transition regimes.
# They never consume PTD policy pieces, descendants, or fusion roots. Extended
# precision resolves a flat objective near its maximum without derivatives.
const REFERENCE_PRECISION = 128

function _reference_second(x::BigFloat, branch::Branch)
    lo, hi = BigFloat(branch.lower), min(BigFloat(branch.upper), x)
    hi > lo || return (saving=lo, consumption=zero(x), value=BigFloat(-Inf))
    objective(a) = utility(x - a) + BigFloat(BETA) *
        utility(BigFloat(branch.intercept) + BigFloat(branch.slope) * a)
    a = maximize_concave(objective, lo, hi)
    return (saving=a, consumption=x - a, value=objective(a))
end

function choose_reference(results)
    best = maximum(r.value for r in results)
    tolerance = 16eps(Float64) * max(abs(best), 1.0)
    eligible = filter(r -> r.value >= best - tolerance, results)
    return eligible[argmax(getfield.(eligible, :saving))]
end

_float_reference(result) = (saving=Float64(result.saving),
    consumption=Float64(result.consumption), value=Float64(result.value))

function reference_second(x, branch::Branch)
    return setprecision(BigFloat, REFERENCE_PRECISION) do
        _float_reference(_reference_second(BigFloat(x), branch))
    end
end

function reference_second(x; transitions=TRANSITIONS)
    return setprecision(BigFloat, REFERENCE_PRECISION) do
        _float_reference(choose_reference([
            _reference_second(BigFloat(x), b) for b in transitions[2]]))
    end
end

function reference_first(x; transitions=TRANSITIONS)
    return setprecision(BigFloat, REFERENCE_PRECISION) do
        cash = BigFloat(x)
        results = NamedTuple{(:saving, :consumption, :value),Tuple{BigFloat,BigFloat,BigFloat}}[]
        for first_branch in transitions[1], second_branch in transitions[2]
            # Each fixed regime pair has a convex feasible set and a concave
            # two-control objective. Its optimized continuation is concave in a1.
            lo = max(BigFloat(first_branch.lower),
                (BigFloat(second_branch.lower) - BigFloat(first_branch.intercept)) /
                BigFloat(first_branch.slope))
            hi = min(BigFloat(first_branch.upper), cash)
            hi > lo || continue
            next_x(a) = BigFloat(first_branch.intercept) + BigFloat(first_branch.slope) * a
            objective(a) = utility(cash - a) + BigFloat(BETA) *
                _reference_second(next_x(a), second_branch).value
            a = maximize_concave(objective, lo, hi)
            push!(results, (saving=a, consumption=cash - a, value=objective(a)))
        end
        _float_reference(choose_reference(results))
    end
end

function check_collection(solution, t)
    pieces = solution.policy[t]
    first(pieces).lower == solution.lower || error("wrong lower endpoint")
    last(pieces).upper == solution.upper[t] || error("wrong upper endpoint")
    for (i, p) in enumerate(pieces)
        p.lower < p.upper || error("degenerate policy piece")
        for x in (p.lower, p.upper)
            c = consumption(p, x)
            0 < c <= x + coordinate_tolerance(x) || error("infeasible policy at $x")
        end
        if i > 1
            previous = pieces[i - 1]
            previous.upper == p.lower || error("gap or overlap in policy collection")
            consumption(p, p.lower) <= consumption(previous, p.lower) + 1e-10 ||
                error("upward consumption jump at $(p.lower)")
        end
    end
    return true
end

function verify(solution=solve(); grid_points=801)
    rows = NamedTuple[]
    for t in 1:2
        check_collection(solution, t)
        xs = collect(range(solution.lower, solution.upper[t]; length=grid_points))
        t == 1 && push!(xs, 10.0)
        for piece in solution.policy[t][2:end]
            join = piece.lower
            append!(xs, [join - 1e-7, join, join + 1e-7])
        end
        sort!(unique!(xs))
        for x in xs
            reference = t == 1 ? reference_first(x; transitions=solution.transitions) :
                reference_second(x; transitions=solution.transitions)
            ptd_c = consumption(policy_piece(solution, t, x), x)
            ptd_v = value(solution, t, x)
            push!(rows, (; date=t, cash=x, ptd_consumption=ptd_c,
                reference_consumption=reference.consumption, ptd_value=ptd_v,
                reference_value=reference.value))
            abs(ptd_v - reference.value) <= 2e-11 || error("value check failed at date $t, x=$x")
        end
    end
    return rows
end

function scientific_tex(x)
    mantissa, exponent = split(@sprintf("%.2e", x), 'e')
    return mantissa * raw"\times10^{" * string(parse(Int, exponent)) * "}"
end

function verification_summary(solution, rows)
    return map(1:2) do t
        sample = filter(r -> r.date == t, rows)
        (; date=t, states=length(sample), pieces=length(solution.policy[t]),
            max_value_error=maximum(abs(r.ptd_value - r.reference_value) for r in sample),
            max_consumption_error=maximum(abs(r.ptd_consumption - r.reference_consumption) for r in sample))
    end
end

"""Save pointwise verification, summaries, and calibration in one flat JLD2 artifact."""
function save_verification(solution, rows; root=PROJECT)
    path = Ptd._validate_data_output(joinpath(root, "data", "multibranch-verification.jld2"))
    fields(object) = NamedTuple{propertynames(object)}(Tuple(getproperty(object, name)
        for name in propertynames(object)))
    artifact = (; schema="ptd-multibranch-verification-v1", beta=BETA,
        root_rtol=ROOT_RTOL, reference_precision=REFERENCE_PRECISION,
        lower=solution.lower, upper=solution.upper,
        transitions=map(branches -> fields.(branches), solution.transitions),
        policy=map(pieces -> fields.(pieces), solution.policy),
        rows, summary=verification_summary(solution, rows))
    mkpath(dirname(path))
    JLD2.jldsave(path; artifact)
    return path
end

"""
    render(solution, rows; root=PROJECT, table_directory=Ptd.tables_dir(),
        graph_directory=Ptd.graphs_dir())

Write the Appendix B verification table and policy figure, and save the
verification artifact.

# Arguments
- `solution::Solution`: the solved appendix example returned by `solve()`.
- `rows::Vector{NamedTuple}`: the pointwise checks returned by
  `verify(solution)`.
- `root::AbstractString`: project folder whose `data/` receives
  `multibranch-verification.jld2`; the default is this package's root.
- `table_directory::AbstractString`: folder for
  `multibranch_verification.tex`; the default `Ptd.tables_dir()` is
  `Tables/` in the active project.
- `graph_directory::AbstractString`: folder for `multibranch_model_b.png`;
  the default `Ptd.graphs_dir()` is `Graphs/` in the active project.

# Returns
- `NamedTuple` with fields `table` (path of the LaTeX table), `graph` (path
  of the figure without its `.png` extension) and `artifact` (path of the
  JLD2 artifact).
"""
function render(solution, rows; root=PROJECT,
    table_directory=Ptd.tables_dir(), graph_directory=Ptd.graphs_dir())

    table = Ptd._validate_data_output(
        joinpath(table_directory, "multibranch_verification.tex"))
    graph = Ptd._validate_data_output(
        joinpath(graph_directory, "multibranch_model_b"))
    artifact = save_verification(solution, rows; root)
    mkpath(dirname(table))
    mkpath(dirname(graph))
    open(table, "w") do io
        println(io, raw"\begin{tabular}{rrrrr}")
        println(io, raw"\toprule")
        println(io, raw"Date & States & Pieces & Max. $|v-v^{\mathrm{ref}}|$ & Max. $|c-c^{\mathrm{ref}}|$ \\\\")
        println(io, raw"\midrule")
        for summary in verification_summary(solution, rows)
            Printf.format(io, Printf.Format(raw"%d & %d & %d & $%s$ & $%s$"),
                summary.date, summary.states, summary.pieces,
                scientific_tex(summary.max_value_error), scientific_tex(summary.max_consumption_error))
            println(io, raw" \\\\")
        end
        println(io, raw"\bottomrule")
        println(io, raw"\end{tabular}")
    end
    get!(ENV, "GKSwstype", "100")
    @eval import Plots
    Base.invokelatest() do
        panels = []
        piece_colors = ["#0072B2", "#009E73", "#CC79A7", "#332288",
            "#882255", "#44AA99", "#8C6D31"]
        for t in 1:2
            # Show the switches clearly; verification retains the full domain.
            display_upper = min(solution.upper[t], (12.0, 8.0)[t])
            # The 10-inch canvas prints 6.3 inches wide, so these font sizes
            # print at roughly 7-9 pt.
            panel = Plots.plot(; xlabel="Cash", ylabel="Consumption", title="Date $t",
                xlims=(0.0, display_upper),
                legend=:topleft, grid=false, foreground_color_legend=:transparent,
                titlefontsize=14, guidefontsize=13, tickfontsize=11,
                legendfontsize=11,
                left_margin=6 * Plots.mm, bottom_margin=6 * Plots.mm)
            for (i, piece) in enumerate(solution.policy[t])
                piece.lower < display_upper || continue
                xs = range(piece.lower, min(piece.upper, display_upper); length=30)
                Plots.plot!(panel, xs, consumption.(Ref(piece), xs);
                    color=piece_colors[mod1(i, length(piece_colors))], linewidth=2.5,
                    label=i == 1 ? "PTD (colored pieces)" : "")
            end
            visible = filter(r -> r.date == t && r.cash <= display_upper, rows)
            # Use the same marker count despite different verification domains.
            indices = round.(Int, range(1, length(visible); length=min(61, length(visible))))
            sampled = visible[indices]
            Plots.scatter!(panel, getfield.(sampled, :cash), getfield.(sampled, :reference_consumption);
                color=:darkorange, markersize=2.5, markershape=:xcross, label="Independent optimizer")
            push!(panels, panel)
        end
        figure = Plots.plot(panels...; layout=(1, 2), size=(1000, 420))
        Plots.savefig(figure, graph * ".png")
    end

    return (; table, graph, artifact)
end

function main()
    solution = solve()
    rows = verify(solution)
    output = render(solution, rows)
    at_ten = reference_first(10.0; transitions=solution.transitions)
    @printf("Appendix example at x=10: c=%.10f, v=%.10f\n", at_ten.consumption, at_ten.value)
    for t in 1:2
        sample = filter(r -> r.date == t, rows)
        @printf("Date %d: %d ordered, abutting pieces; %d states; max value error %.3e\n",
            t, length(solution.policy[t]), length(sample),
            maximum(abs(r.ptd_value - r.reference_value) for r in sample))
    end
    println("Appendix outputs: ", output)
    return (; solution, rows, output)
end

end # module

abspath(PROGRAM_FILE) == abspath(@__FILE__) && MultiBranchExample.main()
