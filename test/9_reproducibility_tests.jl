# Tests for src/simulation/common/reproducibility.jl and source-order file 9.
using Test
using Dates
using JLD2
using Ptd

# Saved artifacts and run-context checks rely on this exact field layout.
const GIT_METADATA_FIELDS = (:repository, :project_root, :scope, :commit,
    :branch, :detached_head, :dirty, :status, :status_entries)

@testset "Stable simulation fingerprints and metadata" begin
    first_config = Dict(:gamma => 2.0, :beta => 0.98,
        :nested => (nodes=15, panels=(:baseline, :high_variance)))
    reordered = Dict(:nested => (nodes=15, panels=(:baseline, :high_variance)),
        :beta => 0.98, :gamma => 2.0)
    changed = merge(first_config, Dict(:gamma => 2.1))

    @test stable_fingerprint(first_config) == stable_fingerprint(reordered)
    @test stable_fingerprint(first_config) != stable_fingerprint(changed)
    @test length(stable_fingerprint(first_config)) == 64
    @test configuration_metadata(first_config).fingerprint_schema ==
        FINGERPRINT_SCHEMA_VERSION
    @test configuration_metadata(first_config).configuration_fingerprint ==
        stable_fingerprint(first_config)

    # Named configuration field order and unordered set iteration do not affect
    # an artifact key. Shape, scalar type, and exact IEEE bits do affect it.
    first_named = (gamma=2.0, beta=0.98)
    reordered_named = (beta=0.98, gamma=2.0)
    @test stable_fingerprint(first_named) == stable_fingerprint(reordered_named)
    @test canonical_token(Set([:b, :a])) == canonical_token(Set([:a, :b]))
    @test canonical_token(reshape([1, 2], 2, 1)) != canonical_token([1, 2])
    @test canonical_token(Float32(0.5)) != canonical_token(Float64(0.5))
    @test canonical_token(-0.0) != canonical_token(0.0)

    # Length framing makes nested values unambiguous even when strings contain
    # punctuation used by the human-readable encoding.
    @test canonical_token(("a|b", "c=d")) !=
        canonical_token(("a", "b|c=d"))
    @test_throws ArgumentError canonical_token(Inf)
    @test_throws ArgumentError canonical_token(BigFloat("1.1"))
    @test_throws ArgumentError canonical_token(Ref(1))

    fixed_time = DateTime(2026, 8, 4, 12, 30, 15, 250)
    runtime = runtime_metadata(timestamp=fixed_time)
    @test runtime.timestamp_utc == "2026-08-04T12:30:15.250Z"
    @test runtime.julia_version == string(VERSION)
    @test runtime.threads == Threads.nthreads()

    packages = package_version_metadata()
    @test !isempty(packages)
    @test issorted(packages; by=entry -> (entry.name, entry.uuid))
    @test all(entry -> !isempty(entry.name) && !isempty(entry.uuid) &&
        !isempty(entry.version), packages)
    @test any(entry -> entry.name == "Ptd", packages)

    # Exercise scoped git metadata in an isolated repository, making this test
    # independent of the checkout or staged-index directory that runs it.
    mktempdir() do repository
        project_root = joinpath(repository, "Ptd")
        mkpath(project_root)
        write(joinpath(project_root, "tracked.txt"), "project\n")
        write(joinpath(repository, "outside.txt"), "outside\n")
        run(`git init -q $repository`)
        run(Cmd(`git add Ptd/tracked.txt outside.txt`; dir=repository))
        # A global commit-signing setting must not break this scratch commit.
        git_options = ["-c", "user.name=Replication-Test",
            "-c", "user.email=test@example.invalid",
            "-c", "commit.gpgsign=false"]
        run(Cmd(`git $git_options commit -q -m initial`; dir=repository))

        clean = git_metadata(project_root)
        @test keys(clean) == GIT_METADATA_FIELDS
        @test clean.project_root == "."
        @test clean.repository == ".."
        @test clean.scope == "Ptd"
        @test !clean.dirty
        @test isempty(clean.status_entries)
        @test clean.commit ==
            readchomp(Cmd(`git rev-parse HEAD`; dir=repository))
        @test clean.branch == Ptd.UNRECORDED_BRANCH
        @test !clean.detached_head
        # Nothing saved reveals where the repository lives on this machine.
        @test !any(value -> value isa AbstractString &&
            occursin(basename(repository), value), values(clean))
        run(Cmd(`git checkout -q --detach`; dir=repository))
        detached = git_metadata(project_root)
        @test detached.detached_head
        @test detached.branch == Ptd.UNRECORDED_BRANCH
        @test detached.commit == clean.commit

        # A change outside Ptd does not contaminate scoped project provenance.
        write(joinpath(repository, "outside.txt"), "changed outside\n")
        @test !git_metadata(project_root).dirty

        write(joinpath(project_root, "tracked.txt"), "changed project\n")
        dirty = git_metadata(project_root)
        @test dirty.dirty
        @test dirty.status_entries isa Vector{String}
        @test any(entry -> occursin("Ptd/tracked.txt", entry),
            dirty.status_entries)

        # git refuses a repository that another user owns. That checkout is
        # versioned and here dirty, so git_metadata must fail instead of
        # recording an unversioned, clean copy. git's test variable below
        # makes the owner look foreign; a git without it, or a
        # safe.directory setting, reads the repository as usual.
        foreign_owner = "GIT_TEST_ASSUME_DIFFERENT_OWNER" => "1"
        git_refuses = withenv(foreign_owner) do
            probe = Cmd(`git rev-parse --show-toplevel`; dir=project_root)
            !success(pipeline(probe; stderr=devnull))
        end
        if git_refuses
            @test_throws "dubious ownership" withenv(
                () -> git_metadata(project_root), foreign_owner)
        else
            @test withenv(() -> git_metadata(project_root),
                foreign_owner).dirty
        end

        metadata = simulation_metadata(
            first_config, project_root; timestamp=fixed_time)
        @test metadata.configuration.configuration_fingerprint ==
            stable_fingerprint(first_config)
        @test metadata.source.dirty
        @test metadata.runtime.timestamp_utc == runtime.timestamp_utc
        @test metadata.packages == packages

        # DrWatson's tagsave/produce_or_load artifacts use JLD2 underneath. A
        # direct round trip catches provenance fields with unsupported types.
        artifact = joinpath(repository, "metadata.jld2")
        JLD2.jldsave(artifact; metadata)
        restored = JLD2.load(artifact, "metadata")
        @test restored == metadata
    end
end

@testset "Git metadata outside a git work tree" begin
    configuration = (nodes=15, panels=(:baseline,))
    fixed_time = DateTime(2026, 8, 4, 12, 30, 15, 250)
    mktempdir() do directory
        write(joinpath(directory, "tracked.txt"), "project\n")
        # An unpacked GitHub or Zenodo ZIP copy has no .git folder. Stop
        # git's search at this folder, so no enclosing repository is found.
        outside_git = "GIT_CEILING_DIRECTORIES" => dirname(directory)
        unversioned = withenv(outside_git) do
            git_metadata(directory)
        end
        @test keys(unversioned) == GIT_METADATA_FIELDS
        @test unversioned.commit == Ptd.UNVERSIONED_COMMIT
        @test unversioned.branch == Ptd.UNRECORDED_BRANCH
        @test unversioned.repository == ""
        @test unversioned.project_root == unversioned.scope == "."
        @test !unversioned.dirty
        @test !unversioned.detached_head
        @test isempty(unversioned.status)
        @test unversioned.status_entries == String[]

        # Without a git executable the record is the same.
        without_git = withenv("PATH" => directory) do
            @test isnothing(Sys.which("git"))
            git_metadata(directory)
        end
        @test without_git == unversioned

        # Artifact writers save this record through simulation_metadata,
        # and JLD2 stores it unchanged.
        metadata = withenv(outside_git) do
            simulation_metadata(configuration, directory;
                timestamp=fixed_time)
        end
        @test metadata.source == unversioned
        artifact = joinpath(directory, "metadata.jld2")
        JLD2.jldsave(artifact; metadata)
        @test JLD2.load(artifact, "metadata") == metadata
    end
end

@testset "Run locks stay outside scientific data" begin
    mktempdir() do directory
        root = joinpath(directory, "data")
        mkpath(root)
        alias = joinpath(directory, "data-alias")
        symlink(root, alias; dir_target=true)
        destination = joinpath(root, "result.jld2")
        path = Ptd._runtime_lock_path(destination, :ar1_exhibit)
        @test dirname(path) != root
        @test path == Ptd._runtime_lock_path(
            joinpath(alias, "result.jld2"), :ar1_exhibit)
        @test path != Ptd._runtime_lock_path(
            joinpath(root, "other.jld2"), :ar1_exhibit)
        @test path != Ptd._runtime_lock_path(destination, :iid_contest)

        # Different runs targeting the same output root still exclude each
        # other, and a failed contender must not remove the owner's lock.
        for runner in (Ptd.IID, Ptd.AR1)
            lock_path = runner._lock_path(root, "first-run")
            @test lock_path == runner._lock_path(root, "second-run")
            @test runner._with_run_lock(root, "first-run") do
                @test isempty(readdir(root))
                @test isfile(joinpath(lock_path, "owner.txt"))
                @test_throws ArgumentError runner._with_run_lock(
                    () -> nothing, alias, "second-run")
                @test isdir(lock_path)
                :completed
            end === :completed
            @test !ispath(lock_path)
            @test_throws ErrorException runner._with_run_lock(
                () -> error("deliberate failure"), root, "first-run")
            @test !ispath(lock_path)
        end

        Ptd.AR1._exhibit_with_run_lock(destination) do
            @test isempty(readdir(root))
            @test isfile(joinpath(path, "owner.txt"))
            @test_throws ArgumentError Ptd.AR1._exhibit_with_run_lock(
                () -> nothing, destination)
            @test isdir(path)
        end
        @test !ispath(path)
        @test_throws ErrorException Ptd.AR1._exhibit_with_run_lock(
            () -> error("deliberate failure"), destination)
        @test !ispath(path)

        # Preserve invalid evidence without leaving non-JLD files in data.
        write(destination, "invalid artifact")
        quarantined = Ptd.AR1._quarantine_artifact(destination)
        try
            @test isempty(readdir(root))
            @test basename(quarantined) == "result.jld2"
            @test read(quarantined, String) == "invalid artifact"
        finally
            rm(dirname(quarantined); recursive=true)
        end
    end
end
