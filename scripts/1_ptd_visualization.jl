"""Generate the Section 3.1 PTD figures and the Section 3.3 policy figure."""

ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using DrWatson
@quickactivate "Ptd"

using Ptd
import Ptd: DcEgm
using LaTeXStrings, Plots, Plots.PlotMeasures


const Colors = palette(:tab10, 10)
const Markers = [:none, :rect, :diamond, :star5, :cross]
const LineStyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]

"""
    paper_plot_style!(; grid=false, linewidth=2.0, markerstrokewidth=0.0,
        legendfontsize=16, left_margin=5mm, bottom_margin=5mm,
        legend=:bottomright, xguidefontsize=17, yguidefontsize=17,
        tickfontsize=15, titlefontsize=19)

Apply the common plotting style used by the deterministic paper figures.

The figures are 12 inches wide at Plots' 100 pixels per inch and the paper
prints them about 5.6 inches wide, so text is set about 2.1 times its
printed size: the default font sizes print at roughly 7-9 pt.

# Arguments
- `grid::Bool`: whether panels draw grid lines.
- `linewidth::Real`: default line width, in pixels.
- `markerstrokewidth::Real`: default marker outline width, in pixels.
- `legendfontsize::Integer`: legend font size, in points.
- `left_margin`, `bottom_margin`: extra panel margins, as
  `Plots.PlotMeasures` lengths such as `5mm`.
- `legend::Symbol`: default legend position.
- `xguidefontsize::Integer`, `yguidefontsize::Integer`: axis-label font
  sizes, in points.
- `tickfontsize::Integer`: tick-label font size, in points.
- `titlefontsize::Integer`: panel-title font size, in points.

# Returns
- `nothing`; the style applies to every plot created afterwards.
"""
function paper_plot_style!(; grid=false, linewidth=2.0, markerstrokewidth=0.0,
    legendfontsize=16, left_margin=5mm, bottom_margin=5mm,
    legend=:bottomright, xguidefontsize=17, yguidefontsize=17,
    tickfontsize=15, titlefontsize=19)

    default(grid=grid, linewidth=linewidth,
        markerstrokewidth=markerstrokewidth,
        legendfontsize=legendfontsize, left_margin=left_margin,
        bottom_margin=bottom_margin, legend=legend,
        xguidefontsize=xguidefontsize, yguidefontsize=yguidefontsize,
        tickfontsize=tickfontsize, titlefontsize=titlefontsize)

    return nothing
end

# Plots for main ideas of solving the deterministic model

"""Draw the policy and value panels for the terminal period `T`."""
function terminal_period_figure(agent::DeterministicProblem; save_path="")
    paper_plot_style!()

    # (a).policy function
    p1 = plot(legend=:bottomright)
    xSup = 15.0
    xlims!(p1, (0, xSup))
    ylims!(p1, (0, xSup))
    xticks!(p1, [0:2:xSup-1;])
    xlabel!(p1, L"x_{T}")
    ylabel!(p1, L"c_{T}")
    plot!(p1, [0.0, xSup], [0.0, xSup], linecolor=Colors[1],
        label=L"c^{0}_{T}(x_{T})")
    title!(p1, "(a) policy function")

    # (b).value function
    xGrid0 = range(0.01, xSup, 2000)
    p2 = plot(legend=:bottomright)
    xSup = 15.0
    xlims!(p2, (0, xSup))
    xticks!(p2, [0:2:xSup-1;])
    xlabel!(p2, L"x_{T}")
    ylabel!(p2, L"v_{T}")
    plot!(p2, xGrid0, x -> v_x(agent, x, i=1, t=50), linecolor=Colors[4],
        label=L"v_{T}(x_{T})")
    title!(p2, "(b) value function")

    p = plot(p1, p2, layout=(1, 2), size=(1200, 600))
    if !isempty(save_path)
        Ptd._validate_data_output(save_path)
        savefig(p, save_path)
    end
    p
end

"""Draw the policy-segment and value-segment panels for period `T - 1`."""
function penultimate_period_figure(agent::DeterministicProblem; save_path="")

    paper_plot_style!()

    # (a).policy candidates
    p1 = plot(legend=:topleft)
    ps1 = descendant_operator(agent, i=1, t=50)[1]
    xSup = 15.0
    kink1 = agent.policy[1, 49][1].xEnd # kink

    xlims!(p1, (0, xSup))
    ylims!(p1, (0, xSup))
    xticks!(p1, [0:2:xSup-1;])
    xlabel!(p1, L"x_{T-1}")
    ylabel!(p1, L"c_{T-1}")

    plot!(p1, [0.0, xSup], [0.0, xSup],
        linecolor=Colors[2], label=L"c^{0}_{T-1}(x_{T-1})")
    plot!(p1, [ps1.xStart, xSup], [c_x(ps1, ps1.xStart), c_x(ps1, xSup)],
        linecolor=Colors[1], label=L"c^{1}_{T-1}(x_{T-1})")
    scatter!(p1, [ps1.xStart,], [c_x(ps1, ps1.xStart),],
        label=L"z_T=x_{\mathrm{floor}}",
        markercolor=:purple)
    # 45 degree arc
    center = (0, 0)
    radius = 0.8
    theta1_rad = 0
    theta2_rad = π / 4
    t = range(theta1_rad, theta2_rad, length=100)
    x = center[1] .+ radius .* cos.(t)
    y = center[2] .+ radius .* sin.(t)
    plot!(p1, x, y, linecolor="gray", label="")
    annotate!(p1, 1.4, 0.5, text(L"45^∘"))
    title!(p1, "(a) policy segments")

    # (b).value candidates
    p2 = plot(legend=:bottomright, legend_column=2)
    xGrid0 = range(0.01, xSup, 1000)
    xGrid1 = range(ps1.xStart, xSup, 1000)
    v0(x) = u(agent, x) + agent.β * v_x(agent, agent.xFloor; i=1, t=50)

    function v1(x)
        zNext = zNext_a(agent, x - c_x(ps1, x); i=1, t=49)
        return u(agent, c_x(ps1, x)) + agent.β * v_x(agent, zNext, i=1, t=50)
    end

    xticks!(p2, [0:2:xSup-1;])
    xlabel!(p2, L"x_{T-1}")
    ylabel!(p2, L"v_{T-1}")

    plot!(p2, xGrid0, v0, linecolor=Colors[2], label=L"v^{0}_{T-1}(x_{T-1})")
    plot!(p2, xGrid1, v1, linecolor=Colors[1], label=L"v^{1}_{T-1}(x_{T-1})")
    scatter!(p2, [xGrid1[1]], [v1(xGrid1[1])],
        label=L"z_T=x_{\mathrm{floor}}", markercolor=:purple)
    scatter!(p2, [kink1], [v_x(agent, kink1; i=1, t=49)],
        label="kink", markercolor=:black)
    title!(p2, "(b) value segments")

    # (c).policy function
    p3 = plot(legend=:topleft, legend_column=-1)
    plot!(p3, xGrid0[xGrid0.<=kink1], x -> c_x(agent, x, i=1, t=49),
        linecolor=Colors[2],
        label=L"c^{\star 0}_{T-1}(x_{T-1})")
    plot!(p3, xGrid0[xGrid0.>=kink1], x -> c_x(agent, x, i=1, t=49),
        linecolor=Colors[1],
        label=L"c^{\star 1}_{T-1}(x_{T-1})")
    plot!(p3, [kink1, kink1], [kink1, c_x(ps1, kink1)], linestyle=:dash,
        linecolor=:grey, label=nothing)
    xticks!(p3, [0:2:xSup-1;])
    xlims!(p3, (0, xSup))
    ylims!(p3, (0, xSup))
    xlabel!(p3, L"x_{T-1}")
    ylabel!(p3, L"c_{T-1}")
    title!(p3, "(c) policy function")

    # (d).value function
    p4 = plot(legend=:bottomright, legend_column=2)
    plot!(p4, xGrid0, x -> v_x(agent, x, i=1, t=49),
        linecolor=Colors[4], label=L"v_{T-1}(x_{T-1})")
    scatter!(p4, [kink1,], [v_x(agent, kink1; i=1, t=49),], label="kink", markercolor=:black)
    xticks!(p4, [0:2:xSup-1;])
    xlabel!(p4, L"x_{T-1}")
    ylabel!(p4, L"v_{T-1}")
    title!(p4, "(d) value function")

    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1200))
    if !isempty(save_path)
        Ptd._validate_data_output(save_path)
        savefig(p, save_path)
    end
    p
end


"""Draw the policy-segment and value-segment panels for period `T - 2`."""
function antepenultimate_period_figure(agent::DeterministicProblem; save_path="")

    # (a) three candidate policy functions
    p1 = plot(legend=:topleft)
    ps1, ps2 = descendant_operator(agent, i=1, t=49)
    xSup = 20.0
    kink1, kink2 = agent.policy[1, 48][1].xEnd, agent.policy[1, 48][2].xEnd # kink

    xlims!(p1, (0, xSup))
    ylims!(p1, (0, xSup))
    xticks!(p1, [0:4:xSup;])

    xlabel!(p1, L"x_{T-2}")
    ylabel!(p1, L"c_{T-2}")

    plot!(p1, [0.0, xSup], [0.0, xSup],
        linecolor=Colors[3], label=L"c^{0}_{T-2}(x_{T-2})")
    plot!(p1, [ps1.xStart, ps1.xEnd],
        [c_x(ps1, ps1.xStart), c_x(ps1, ps1.xEnd)],
        linecolor=Colors[2],
        label=L"c^{1}_{T-2}(x_{T-2})")
    plot!(p1, [ps2.xStart, xSup],
        [c_x(ps2, ps2.xStart), c_x(ps2, xSup)],
        linecolor=Colors[1],
        label=L"c^{2}_{T-2}(x_{T-2})")

    scatter!(p1, [ps1.xStart,], [c_x(ps1, ps1.xStart),],
        label=L"z_{T-1}=x_{\mathrm{floor}}", markercolor=:purple)
    # 45 degree arc
    center = (0, 0)
    radius = 0.8
    theta1_rad = 0
    theta2_rad = π / 4
    t = range(theta1_rad, theta2_rad, length=100)
    x = center[1] .+ radius .* cos.(t)
    y = center[2] .+ radius .* sin.(t)
    plot!(p1, x, y, linecolor="gray", label="")
    annotate!(p1, 1.4, 0.5, text(L"45^∘", 12))
    title!(p1, "(a) policy segments")

    # (b) three candidate value functions
    p2 = plot(legend=:bottomright, legend_column=2)
    xGrid0 = range(0.01, xSup, 1000)
    xGrid1 = range(ps1.xStart, ps1.xEnd, 1000)
    xGrid2 = range(ps2.xStart, xSup, 1000)

    v0(x) = u(agent, x) + agent.β * v_x(agent, agent.xFloor, i=1, t=49)

    function v1(x)
        zNext = zNext_a(agent, x - c_x(ps1, x), i=1, t=48)
        return u(agent, c_x(ps1, x)) + agent.β * v_x(agent, zNext, i=1, t=49)
    end

    function v2(x)
        xNext = xNext_a(agent, x - c_x(ps2, x), i=1, t=48)
        return u(agent, c_x(ps2, x)) + agent.β * v_x(agent, xNext, i=1, t=49)
    end

    xticks!(p2, [0:4:xSup;])
    xlabel!(p2, L"x_{T-2}")
    ylabel!(p2, L"v_{T-2}")

    plot!(p2, xGrid0, v0, linecolor=Colors[3], label=L"v^{0}_{T-2}(x_{T-2})")
    plot!(p2, xGrid1, v1, linecolor=Colors[2], label=L"v^{1}_{T-2}(x_{T-2})")
    plot!(p2, xGrid2, v2, linecolor=Colors[1], label=L"v^{2}_{T-2}(x_{T-2})")
    scatter!(p2, [ps1.xStart], [v1(ps1.xStart)],
        label=L"z_{T-1}=x_{\mathrm{floor}}", markercolor=:purple)
    scatter!(p2, [kink1], [v_x(agent, kink1; i=1, t=48)],
        label="kinks", markercolor=:black)
    scatter!(p2, [kink2], [v_x(agent, kink2; i=1, t=48)],
        label=nothing, markercolor=:black)
    title!(p2, "(b) value segments")

    # policy function at T-2
    p3 = plot(legend=:topleft, legend_column=2)

    xlims!(p3, (0, xSup))
    ylims!(p3, (0, xSup))
    xticks!(p3, [0:4:xSup;])

    xlabel!(p3, L"x_{T-2}")
    ylabel!(p3, L"c_{T-2}")

    plot!(p3, [0.0, kink1], [0.0, kink1],
        linecolor=Colors[3], label=L"c^{\star 0}_{T-2}(x_{T-2})")
    plot!(p3, [kink1, kink2],
        [c_x(ps1, kink1), c_x(ps1, kink2)],
        linecolor=Colors[2],
        label=L"c^{\star 1}_{T-2}(x_{T-2})")
    plot!(p3, [kink2, xSup],
        [c_x(ps2, kink2), c_x(ps2, xSup)],
        linecolor=Colors[1],
        label=L"c^{\star 2}_{T-2}(x_{T-2})")

    plot!(p3, [kink1, kink1], [kink1, c_x(ps1, kink1)],
        linestyle=:dash, linecolor=:grey, label=nothing)
    plot!(p3, [kink2, kink2], [c_x(ps1, kink2), c_x(ps2, kink2)],
        linestyle=:dash, linecolor=:grey, label=nothing)

    title!(p3, "(c) policy function")

    # value function
    p4 = plot(legend=:bottomright, legend_column=2)
    plot!(p4, xGrid0, map(x -> v_x(agent, x, i=1, t=48), xGrid0),
        linecolor=Colors[4], label=L"v_{T-2}(x_{T-2})")
    scatter!(p4, [kink1], [v_x(agent, kink1; i=1, t=48),],
        label="kinks", markercolor=:black)
    scatter!(p4, [kink2], [v_x(agent, kink2; i=1, t=48),],
        label=nothing, markercolor=:black)
    xticks!(p4, [0:4:xSup;])
    xlabel!(p4, L"x_{T-2}")
    ylabel!(p4, L"v_{T-2}")
    title!(p4, "(d) value function")

    p = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 1200))
    if !isempty(save_path)
        Ptd._validate_data_output(save_path)
        savefig(p, save_path)
    end
    p
end

# Plot policy functions for deterministic model.

"""Draw the period-`t` deterministic PTD policy on `[0, xMax]`."""
function deterministic_policy_plot(
    agent::DeterministicProblem; xMax::Float64, t::Int=1)

    paper_plot_style!()
    p = plot(legend=:topright)
    deterministic_ptd!(agent)

    for po in agent.policy[1, t]
        if po.xStart < xMax
            x1, x2 = po.xStart, min(xMax, po.xEnd)
            y1 = c_x(po, x1)
            y2 = c_x(po, x2)
            label = po.xStart == 0.0 ? "PTD" : ""
            plot!(p, [x1, x2], [y1, y2], linecolor=Colors[1], label=label)
        else
            break
        end
    end

    xlabel!(p, L"x_{%$t}")
    ylabel!(p, L"c_{%$t}")
    title!(p, "(a) policy function at " * L"t = %$t" * " (PTD)")
    return p
end


"""
Draw period-`t` policies produced by the pure-Julia DC-EGM implementation on
`[0, xMax]` in the deterministic model.
"""
function deterministic_policy_plot(
    agents::Vector{DcEgm.AgentD}; xMax::Float64, t::Int=1, gap=0.01)

    paper_plot_style!()
    p = plot(legend=:topright)
    xVec = collect(0.0:gap:xMax)

    for (i, agent) in enumerate(agents)
        DcEgm.egm!(agent)
        plot!(p, xVec, DcEgm.c_x(agent, xVec, t=t), linecolor=Colors[i+1],
            linestyle=LineStyles[i], label="DC-EGM$i")
    end

    xlabel!(p, L"x_{%$t}")
    ylabel!(p, L"c_{%$t}")
    title!(p, "(b) policy function at " * L"t = %$t" * " (DC-EGM)")
    return p
end

"""Write the three Section 3.1 figures to `graph_directory`."""
function ptd_visualization_outputs(;
    graph_directory::AbstractString=Ptd.graphs_dir())
    Ptd._validate_data_output(joinpath(graph_directory, "policy_value_T.png"))
    mkpath(graph_directory)
    agent = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0,
        incomes=fill(1.0, (1, 50)))
    deterministic_ptd!(agent)
    paths = (
        joinpath(graph_directory, "policy_value_T.png"),
        joinpath(graph_directory, "policy_value_T-1.png"),
        joinpath(graph_directory, "policy_value_T-2.png"),
    )
    terminal_period_figure(agent; save_path=paths[1])
    penultimate_period_figure(agent; save_path=paths[2])
    antepenultimate_period_figure(agent; save_path=paths[3])
    return paths
end

"""
    deterministic_policy_comparison(; graph_directory=Ptd.graphs_dir())

Write the Section 3.3 figure comparing PTD with the four DC-EGM grids.

The left panel is the PTD consumption policy at date 1; the right panel shows
DC-EGM1 to DC-EGM4 on the four asset grids of Table 1. The package keeps
those grids in `Ptd.FixedCalibrationTiming.deterministic_timing_grids()`,
and test 31 checks that they equal `deterministic_asset_grids()` in
`scripts/2_deterministic.jl`. Both panels solve their models afresh.

# Arguments
- `graph_directory::AbstractString`: folder for `d_policies_ptd_egm.png`;
  the default `Ptd.graphs_dir()` is `Graphs/` in the active project.

# Returns
- `String`: the absolute path of the written PNG file.
"""
function deterministic_policy_comparison(;
    graph_directory::AbstractString=Ptd.graphs_dir())

    path = Ptd._validate_data_output(
        joinpath(graph_directory, "d_policies_ptd_egm.png"))
    mkpath(dirname(path))
    # DC-EGM1 to DC-EGM4 use the package's copy of the Table 1 grids.
    asset_grids = collect(values(
        Ptd.FixedCalibrationTiming.deterministic_timing_grids()))
    agent = DeterministicProblem(β=0.98, γ=1.0, r=0.1, xFloor=3.0,
        incomes=fill(1.0, (1, 50)))
    agents_egm = [DcEgm.build_agent(model="deterministic", β=agent.β,
        γ=agent.γ, r=agent.r, xFloor=agent.xFloor, tMax=agent.tMax,
        a=grid, incomes=agent.incomes[1, :]) for grid in asset_grids]
    comparison = plot(
        deterministic_policy_plot(agent; xMax=50.0, t=1),
        deterministic_policy_plot(agents_egm; xMax=50.0, t=1),
        layout=(1, 2), size=(1200, 600))
    savefig(comparison, path)

    return path
end

if abspath(PROGRAM_FILE) == @__FILE__
    foreach(println, ptd_visualization_outputs())
    println(deterministic_policy_comparison())
end
