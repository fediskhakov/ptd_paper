using Test
using JLD2
using SHA

# The grid-refinement footnote in Section 4.4.3 cites the saved 17-, 18-
# and 19-node G2EGM runs at initial cash 21.25. Every table here goes to a
# temporary folder, never to the package's Tables/ folder.
@testset "grid-refinement footnote reproduced from saved artifacts" begin
    H = Ptd.HSZ
    baseline = H.default_hsz_utility_artifact_path()
    shipped = H.paper_hsz_grid_refinement_paths()
    data = joinpath(pkgdir(Ptd), "data")
    @test shipped == [joinpath(data, "hsz-grid-refinement.jld2"),
        joinpath(data, "hsz-grid-refinement-19.jld2")]
    row_end = " \\\\"
    mktempdir() do directory
        timed_path = joinpath(directory, "timed.tex")
        timed = H.compose_hsz_grid_refinement(baseline, shipped;
            table_path=timed_path, include_timings=true)
        @test timed.table_path == timed_path
        @test timed.family_size == 30
        @test [row.nodes for row in timed.rows] ==
            repeat([17, 18, 19]; inner=5)
        @test [row.initial_cash for row in timed.rows] ==
            repeat([5.5, 10.75, 16.0, 21.25, 26.5], 3)
        @test all(row -> row.n_paths == 100, timed.rows)

        # The footnote's numbers: floor-transfer paths and initial G2EGM
        # consumption at x1 = 21.25 for 17, 18 and 19 nodes.
        footnote = filter(row -> row.initial_cash == 21.25, timed.rows)
        @test [row.nodes for row in footnote] == [17, 18, 19]
        @test [row.g2egm_floor_paths for row in footnote] == [26, 73, 17]
        @test [round(row.initial_consumption; digits=3)
            for row in footnote] == [1.188, 2.635, 1.029]
        lines = readlines(timed_path)
        footnote_lines = [
            "17 & 21.25 & 1.188 & 26/100 & " *
                raw"$-2.7\;[-4.42,\,-0.941]$ & $-2.7\;[-4.42,\,-0.942]$",
            "18 & 21.25 & 2.635 & 73/100 & " *
                raw"$-9.67\;[-12.2,\,-7.03]$ & $-9.67\;[-12.2,\,-7.03]$",
            "19 & 21.25 & 1.029 & 17/100 & " *
                raw"$-1.4\;[-2.56,\,-0.224]$ & $-1.4\;[-2.56,\,-0.224]$"]
        @test all(line -> line * row_end in lines, footnote_lines)
        timing_lines = [
            raw"\multicolumn{6}{l}{Baseline paths reused; " *
                "timing not remeasured.}",
            raw"\multicolumn{6}{l}{Saved 18-node run: construction " *
                "727.676 s; replay (500 paths) 0.006 s.}",
            raw"\multicolumn{6}{l}{Saved 19-node run: construction " *
                "696.530 s; replay (500 paths) 0.007 s.}"]
        @test filter(line -> startswith(line, raw"\multicolumn"), lines) ==
            timing_lines .* row_end

        # The header names the inputs inside the package by relative path
        # and SHA-256, never by an absolute path.
        text = read(timed_path, String)
        for (label, path) in (("Baseline", baseline),
            ("Grid 18", first(shipped)), ("Grid 19", last(shipped)))
            relative = "data/" * basename(path)
            digest = bytes2hex(SHA.sha256(read(path)))
            sha_label = label == "Baseline" ? label : label * " artifact"
            @test occursin("% $label artifact: $relative\n", text)
            @test occursin("% $sha_label SHA-256: $digest\n", text)
        end
        @test occursin("\n% Bonferroni family size: 30;", text)
        @test !occursin(pkgdir(Ptd), text)
        @test !occursin(directory, text)

        # Accuracy rows only by default, identical to the timed table's.
        accuracy_path = joinpath(directory, "accuracy.tex")
        accuracy = H.compose_hsz_grid_refinement(baseline, shipped;
            table_path=accuracy_path)
        @test accuracy.family_size == 30
        @test accuracy.rows == timed.rows
        @test readlines(accuracy_path) ==
            filter(line -> !startswith(line, raw"\multicolumn"), lines)
    end
end

@testset "grid-refinement composition rejects ambiguous inputs" begin
    H = Ptd.HSZ
    baseline = H.default_hsz_utility_artifact_path()
    shipped = H.paper_hsz_grid_refinement_paths()
    mktempdir() do directory
        table_path = joinpath(directory, "table.tex")
        @test_throws ArgumentError H.compose_hsz_grid_refinement(baseline,
            String[]; table_path)
        missing_path = joinpath(directory, "missing", "refinement.jld2")
        @test_throws ArgumentError H.compose_hsz_grid_refinement(baseline,
            [missing_path]; table_path)
        @test !ispath(dirname(missing_path))

        # A finer grid may come from only one artifact.
        copy_path = joinpath(directory, "hsz-grid-refinement-copy.jld2")
        cp(first(shipped), copy_path)
        @test_throws ArgumentError H.compose_hsz_grid_refinement(baseline,
            [first(shipped), copy_path]; table_path)
        @test !ispath(table_path)

        # Archived timings are shown together only when every artifact
        # records the same Julia and BLAS thread counts.
        artifact = JLD2.load(last(shipped), "artifact")
        artifact["julia_threads"] = 8
        threads_path = joinpath(directory, "refinement-8-threads.jld2")
        JLD2.jldsave(threads_path; artifact, payload_fingerprint=
            Ptd.stable_fingerprint(H._utility_record(artifact)))
        @test_throws ArgumentError H.compose_hsz_grid_refinement(baseline,
            [first(shipped), threads_path]; table_path,
            include_timings=true)
        @test !ispath(table_path)
        mixed = H.compose_hsz_grid_refinement(baseline,
            [first(shipped), threads_path]; table_path)
        @test [row.g2egm_floor_paths for row in mixed.rows
            if row.initial_cash == 21.25] == [26, 73, 17]
        # Inputs outside the package appear by file name only.
        @test occursin("% Grid 19 artifact: refinement-8-threads.jld2\n",
            read(table_path, String))
        @test sort(readdir(directory)) == ["hsz-grid-refinement-copy.jld2",
            "refinement-8-threads.jld2", "table.tex"]
    end
end
