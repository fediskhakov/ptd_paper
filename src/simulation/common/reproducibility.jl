using Dates
using Pkg
using SHA

"""
Version of the canonical configuration encoding.

The schema label is included in every digest, so a deliberate future encoding
change cannot silently collide with an artifact produced under this schema.
"""
const FINGERPRINT_SCHEMA_VERSION = "ptd-config-v1"

"""Length-frame a tag and its components into an unambiguous UTF-8 token."""
function _frame(tag::AbstractString, components)
    tokens = String[String(component) for component in components]
    output = IOBuffer()
    write(output, string(ncodeunits(tag)), ":", tag, ":", string(length(tokens)), ":")
    for token in tokens
        write(output, string(ncodeunits(token)), ":", token)
    end
    return String(take!(output))
end

"""
    package_version_metadata()

Return the complete resolved Julia environment as a sorted, JLD2-safe vector.
The manifest UUID is retained so packages with the same display name cannot be
confused. Standard libraries whose dependency record has no independent version
use the running Julia version explicitly.
"""
function package_version_metadata()
    packages = NamedTuple{(:name, :uuid, :version),Tuple{String,String,String}}[]
    for (uuid, information) in Pkg.dependencies()
        name = information.name
        isnothing(name) && continue
        version = isnothing(information.version) ? VERSION : information.version
        push!(packages, (;
            name=String(name),
            uuid=string(uuid),
            version=string(version),
        ))
    end
    package_id = Base.PkgId(@__MODULE__)
    if all(entry -> entry.uuid != string(package_id.uuid), packages)
        push!(packages, (;
            name=String(package_id.name),
            uuid=string(package_id.uuid),
            version=string(pkgversion(@__MODULE__)),
        ))
    end
    sort!(packages; by=entry -> (entry.name, entry.uuid))
    return packages
end

"""Encode a supported value canonically for stable scientific fingerprints."""
canonical_token(::Nothing) = _frame("nothing", ())
canonical_token(::Missing) = _frame("missing", ())
canonical_token(value::Bool) = _frame("bool", (value ? "1" : "0",))

# Integer width is deliberately ignored: an explicit count has the same
# configuration meaning on 32- and 64-bit Julia. Bool remains a separate case.
canonical_token(value::Integer) = _frame("integer", (string(value),))
canonical_token(value::Char) = _frame("char", (string(value),))
canonical_token(value::Symbol) = _frame("symbol", (String(value),))
canonical_token(value::AbstractString) = _frame("string", (String(value),))
canonical_token(value::Date) = _frame("date", (string(Dates.value(value)),))
canonical_token(value::DateTime) =
    _frame("datetime", (string(Dates.value(value)),))
canonical_token(value::VersionNumber) =
    _frame("version", (string(value),))

function _float_bits(value::Float16)
    return string(reinterpret(UInt16, value); base=16, pad=4)
end

function _float_bits(value::Float32)
    return string(reinterpret(UInt32, value); base=16, pad=8)
end

function _float_bits(value::Float64)
    return string(reinterpret(UInt64, value); base=16, pad=16)
end

function canonical_token(value::T) where {T<:Union{Float16,Float32,Float64}}
    isfinite(value) || throw(ArgumentError(
        "Stable configuration fingerprints require finite floats."))
    return _frame(lowercase(string(T)), (_float_bits(value),))
end

# BigFloat depends on an ambient precision and rounding context. Silently
# converting it to Float64 would make distinct configurations collide.
function canonical_token(value::AbstractFloat)
    throw(ArgumentError(
        "Unsupported floating-point type $(typeof(value)); convert explicit " *
        "configuration values to Float16, Float32, or Float64."))
end

canonical_token(value::Pair) = _frame("pair", (
    canonical_token(first(value)), canonical_token(last(value))))

function canonical_token(value::Tuple)
    return _frame("tuple", (canonical_token(item) for item in value))
end

function canonical_token(value::NamedTuple)
    entries = String[]
    for name in keys(value)
        push!(entries, _frame("field", (
            canonical_token(Symbol(name)),
            canonical_token(getproperty(value, name)),
        )))
    end
    # Field order is irrelevant for an explicit named configuration, just as
    # insertion order is irrelevant for a dictionary configuration.
    sort!(entries)
    return _frame("named_tuple", entries)
end

function canonical_token(value::AbstractDict)
    entries = [_frame("entry", (
        canonical_token(key), canonical_token(item)))
        for (key, item) in pairs(value)]
    sort!(entries)
    return _frame("dictionary", entries)
end

function canonical_token(value::AbstractSet)
    entries = sort!([canonical_token(item) for item in value])
    return _frame("set", entries)
end

function canonical_token(value::AbstractArray)
    dimensions = canonical_token(Tuple(size(value)))
    entries = String[dimensions]
    append!(entries, (canonical_token(item) for item in value))
    return _frame("array", entries)
end

function canonical_token(value)
    throw(ArgumentError(
        "Unsupported configuration value of type $(typeof(value)). " *
        "Convert configurations explicitly to named tuples, dictionaries, " *
        "arrays, tuples, sets, or supported scalar values before fingerprinting."))
end

"""Session-independent SHA-256 fingerprint of an explicit configuration."""
function stable_fingerprint(configuration)
    payload = _frame("fingerprint", (
        FINGERPRINT_SCHEMA_VERSION,
        canonical_token(configuration),
    ))
    return bytes2hex(sha256(Vector{UInt8}(codeunits(payload))))
end

"""Return a stable temporary lock location for an output destination.

Processes writing the same destination share a lock, including when they use
different spellings of a symlinked path. Lock directories and owner records
stay outside the scientific data directory.
"""
function _runtime_lock_path(destination::AbstractString, namespace::Symbol)
    ancestor = abspath(destination)
    suffix = String[]
    while !ispath(ancestor)
        push!(suffix, basename(ancestor))
        ancestor = dirname(ancestor)
    end
    canonical_destination = joinpath(realpath(ancestor), reverse(suffix)...)
    identity = stable_fingerprint((; namespace,
        destination=canonical_destination))
    return joinpath(tempdir(), "ptd-run-locks", string(namespace, "-", identity, ".lock"))
end

"""Compact, immutable, JLD2-safe identity fields for a configuration artifact."""
function configuration_metadata(configuration)
    token = canonical_token(configuration)
    return (;
        fingerprint_schema=FINGERPRINT_SCHEMA_VERSION,
        configuration_fingerprint=stable_fingerprint(configuration),
        canonical_configuration=token,
    )
end

function _git_read(arguments::Cmd, directory::AbstractString)
    return String(readchomp(Cmd(arguments; dir=directory)))
end

"""
Branch name that `git_metadata` saves instead of the local branch.

A branch name describes one developer's checkout, not the source code, so
it is never written to an artifact. The commit identifies the source.
"""
const UNRECORDED_BRANCH = "unrecorded"

"""
Commit that `git_metadata` saves when the package has no git history.

This happens in an unpacked GitHub or Zenodo ZIP copy, or when the `git`
executable is missing. The SHA-256 input trees saved by the runners still
identify every source file.
"""
const UNVERSIONED_COMMIT = "unversioned"

"""
    _git_work_tree_root(directory)

Find the root of the git work tree that contains `directory`.

Only git's own verdict that `directory` is not a git repository counts as
a copy without git history. Any other failure stops with git's message:
for example, git refuses to read a repository that another user owns
("dubious ownership"), which is common in containers and on shared
drives. Such a checkout has a commit and may have uncommitted changes, so
treating it as unversioned would skip the runners' refusal of a dirty
tree. git runs with `LC_ALL=C`, so its messages are in English.

# Arguments
- `directory::AbstractString`: existing directory where the search
  starts.

# Returns
- `Union{Nothing,String}`: the work-tree root as `git` reports it, an
  absolute path with symbolic links resolved; or `nothing` when the `git`
  executable is missing or git reports that `directory` is not inside a
  git repository. Any other git failure throws an `ErrorException` that
  quotes git's message.
"""
function _git_work_tree_root(directory::AbstractString)

    isnothing(Sys.which("git")) && return nothing
    command = addenv(Cmd(`git rev-parse --show-toplevel`; dir=directory),
        "LC_ALL" => "C")
    output = IOBuffer()
    errors = IOBuffer()
    found = success(pipeline(command; stdout=output, stderr=errors))
    message = strip(String(take!(errors)))
    unversioned = occursin(r"^fatal: not a git repository"im, message)
    found || unversioned || error(
        "git cannot read the repository that contains $directory: $message")
    root = found ? String(chomp(String(take!(output)))) : nothing

    return root
end

"""
    git_metadata(project_root)

Return the git identity of the source tree at `project_root`.

The record holds nothing specific to one machine, so it can be saved in
public artifacts. Its paths are relative, and the local branch name is
replaced by `UNRECORDED_BRANCH`. The commit is the true `HEAD` hash,
because the runners compare it before and after a run. Changes elsewhere
in a containing repository do not mark the project dirty.

Outside a git work tree, for example in an unpacked GitHub or Zenodo ZIP
copy, or when the `git` executable is missing, the function does not
throw. It records the commit as `UNVERSIONED_COMMIT`, `dirty` as `false`
and an empty status. Such a copy has no commit that could identify it.
The runners still save the SHA-256 digest of every computational input
file, and those input trees pin the exact source. When git finds a
repository but refuses to read it, for example one that another user
owns, the function throws an `ErrorException` with git's message instead
(see `_git_work_tree_root`).

# Arguments
- `project_root::AbstractString`: existing package directory, usually
  `PROJECT_ROOT`.

# Returns
- `NamedTuple` of JLD2-safe values with fields:
  - `repository::String`: the git work-tree root relative to
    `project_root` (`"."` when the package is the repository root, `".."`
    when it is a direct subfolder), or `""` outside a work tree;
  - `project_root::String`: always `"."`, the package root itself;
  - `scope::String`: `project_root` relative to the work-tree root, the
    part of the repository that `git status` checks (`"."` outside a
    work tree);
  - `commit::String`: the full `HEAD` commit hash, or
    `UNVERSIONED_COMMIT` outside a work tree;
  - `branch::String`: always `UNRECORDED_BRANCH`;
  - `detached_head::Bool`: whether `HEAD` is detached (`false` outside a
    work tree);
  - `dirty::Bool`: whether any file below `scope`, untracked files
    included, differs from `HEAD` (`false` outside a work tree);
  - `status::String`: `status_entries` joined by newlines;
  - `status_entries::Vector{String}`: `git status --porcelain=v1` entries
    for `scope` (empty outside a work tree).
"""
function git_metadata(project_root::AbstractString)

    root = abspath(project_root)
    isdir(root) || throw(ArgumentError("Project root does not exist: $root"))
    work_tree = _git_work_tree_root(root)
    if isnothing(work_tree)
        return (;
            repository="",
            project_root=".",
            scope=".",
            commit=UNVERSIONED_COMMIT,
            branch=UNRECORDED_BRANCH,
            detached_head=false,
            dirty=false,
            status="",
            status_entries=String[],
        )
    end
    # git reports the work tree with symbolic links resolved.
    resolved_root = realpath(root)
    resolved_work_tree = realpath(work_tree)
    scope = relpath(resolved_root, resolved_work_tree)
    commit = _git_read(`git rev-parse HEAD`, work_tree)
    branch = _git_read(`git branch --show-current`, work_tree)
    status_options = `--porcelain=v1 -z --untracked-files=all`
    raw_status = read(Cmd(`git status $status_options -- $scope`;
        dir=work_tree), String)
    status_entries = String.(split(raw_status, '\0'; keepempty=false))
    metadata = (;
        repository=relpath(resolved_work_tree, resolved_root),
        project_root=".",
        scope,
        commit,
        branch=UNRECORDED_BRANCH,
        detached_head=isempty(branch),
        dirty=!isempty(status_entries),
        status=join(status_entries, '\n'),
        status_entries,
    )

    return metadata
end

"""Portable runtime fields saved alongside every raw/processed artifact."""
function runtime_metadata(; timestamp::DateTime=now(UTC))
    return (;
        timestamp_utc=Dates.format(timestamp, dateformat"yyyy-mm-ddTHH:MM:SS.sss") * "Z",
        julia_version=string(VERSION),
        threads=Threads.nthreads(),
        word_size=Sys.WORD_SIZE,
        kernel=string(Sys.KERNEL),
        machine=Sys.MACHINE,
        cpu_name=Sys.CPU_NAME,
    )
end

"""
    simulation_metadata(configuration, project_root; timestamp=now(UTC))

Return configuration identity, source state, and runtime provenance as a
NamedTuple of JLD2-safe values.
"""
function simulation_metadata(configuration,
                              project_root::AbstractString;
                              timestamp::DateTime=now(UTC))
    return (;
        configuration=configuration_metadata(configuration),
        source=git_metadata(project_root),
        runtime=runtime_metadata(; timestamp=timestamp),
        packages=package_version_metadata(),
    )
end
