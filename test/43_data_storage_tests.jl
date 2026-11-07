using Test
using JLD2

@testset "flat JLD2 data output paths" begin
    mktempdir() do root
        data = joinpath(root, "data")
        mkpath(data)
        validate(path) = Ptd._validate_data_output(path; data_root=data)
        @test validate(joinpath(data, "result.jld2")) == joinpath(data, "result.jld2")
        @test_throws ArgumentError validate(joinpath(data, "result.jld"))
        table = joinpath(root, "Tables", "table.tex")
        @test validate(table) == table
        for path in (data, joinpath(data, "notes.txt"), joinpath(data, "plot.png"),
            joinpath(data, "nested", "result.jld2"), joinpath(data, "run.lock"))
            @test_throws ArgumentError validate(path)
        end
        @test isempty(readdir(data))

        destination = joinpath(data, "result.jld2")
        temporary = Ptd._temporary_jld_path(destination)
        @test dirname(temporary) == data
        @test endswith(temporary, ".jld2")
        @test isfile(temporary)
        @test all(path -> isfile(path) && endswith(path, ".jld2"), readdir(data; join=true))
        JLD2.jldsave(temporary; values=[1.0, 2.0])
        Base.rename(temporary, destination)
        @test JLD2.load(destination, "values") == [1.0, 2.0]
        @test readdir(data) == ["result.jld2"]
    end
end

@testset "paper data directory contains only flat JLD2 results" begin
    paths = readdir(joinpath(pkgdir(Ptd), "data"); join=true)
    @test !isempty(paths)
    @test all(path -> isfile(path) && lowercase(splitext(path)[2]) == ".jld2", paths)
end

@testset "exhibit folders sit at the top of the active project" begin
    # The README runs the scripts as `julia --project=. scripts/...` from
    # the package root. A fresh process started that way must place the
    # exhibits in the package's Tables/ and Graphs/ folders, so the helpers
    # must follow the active project when called, not when precompiled.
    root = pkgdir(Ptd)
    code = "using Ptd; println(Ptd.tables_dir()); println(Ptd.graphs_dir())"
    julia = Base.julia_cmd()
    folders = readlines(`$julia --startup-file=no --project=$root -e $code`)
    @test folders == [joinpath(root, "Tables"), joinpath(root, "Graphs")]
    table = Ptd.tables_dir("table.tex")
    @test table == joinpath(Ptd.tables_dir(), "table.tex")
    @test Ptd.graphs_dir("figure.png") ==
        joinpath(Ptd.graphs_dir(), "figure.png")
    # Exhibits stay outside data/, which holds only flat JLD2 results.
    @test Ptd._validate_data_output(table) == table
end

@testset "saved provenance paths name no local folders" begin
    root = pkgdir(Ptd)
    @test Ptd.portable_path(joinpath(root, "data", "iid-contest.jld2")) ==
        "data/iid-contest.jld2"
    @test Ptd.portable_path(root) == "."
    mktempdir() do directory
        outside = joinpath(directory, "nested", "accuracy.jld2")
        # A file outside the package keeps only its name.
        @test Ptd.portable_path(outside) == "accuracy.jld2"
        @test Ptd.portable_path(outside; root=directory) ==
            "nested/accuracy.jld2"
        # A folder whose name only starts like the root lies outside it.
        @test Ptd.portable_path(directory * "-other/accuracy.jld2";
            root=directory) == "accuracy.jld2"
        # A relative path is resolved against the working directory.
        cd(directory) do
            @test Ptd.portable_path(joinpath("nested", "accuracy.jld2");
                root=directory) == "nested/accuracy.jld2"
            @test Ptd.portable_path(joinpath("..", "accuracy.jld2");
                root=directory) == "accuracy.jld2"
        end
    end
    # Symbolic links are resolved first. On macOS, mktempdir() returns a
    # path below /var, a link to /private/var, and pwd() the physical one.
    mktempdir() do directory
        physical = joinpath(directory, "physical")
        mkpath(joinpath(physical, "data"))
        linked = joinpath(directory, "linked")
        symlink(physical, linked; dir_target=true)
        file = joinpath("data", "accuracy.jld2")
        @test Ptd.portable_path(joinpath(linked, file); root=physical) ==
            "data/accuracy.jld2"
        @test Ptd.portable_path(joinpath(physical, file); root=linked) ==
            "data/accuracy.jld2"
        cd(linked) do
            @test Ptd.portable_path(file; root=linked) == "data/accuracy.jld2"
        end
        @test Ptd.portable_path(linked; root=physical) == "."
        @test Ptd.portable_path(joinpath(directory, "other", "run.jld2");
            root=linked) == "run.jld2"
    end
end
