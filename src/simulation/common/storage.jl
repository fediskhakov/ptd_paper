using DrWatson: projectdir

"""
    tables_dir(parts...)

Return a path in `Tables/`, the folder for the generated LaTeX tables.

The folder sits at the top level of the active Julia project, the way
DrWatson's own `plotsdir` wraps `projectdir`: the result is
`DrWatson.projectdir("Tables", parts...)`, and `projectdir()` is
`dirname(Base.active_project())`. Every paper script starts with
`@quickactivate "Ptd"`, which activates this package unless it is already
active (for example through `--project`), so the scripts write their
tables to `<package>/Tables`. Under any other active project the path
points into that project instead: `Pkg.test`, for example, starts in a
temporary environment, so tests always pass explicit output paths. The
path is computed at each call, never fixed at precompilation. The folder is
not created here; every writer creates its own destination folder.

# Arguments
- `parts::AbstractString...`: optional path components below `Tables/`,
  usually one file name such as `"iid_contest_summary.tex"`. With none,
  the path of the folder itself is returned.

# Returns
- `String`: the absolute path `<active project>/Tables/<parts...>`.
"""
function tables_dir(parts::AbstractString...)

    path = projectdir("Tables", parts...)

    return path
end

"""
    graphs_dir(parts...)

Return a path in `Graphs/`, the folder for the generated PNG figures.

The folder sits at the top level of the active Julia project, the way
DrWatson's own `plotsdir` wraps `projectdir`: the result is
`DrWatson.projectdir("Graphs", parts...)`, and `projectdir()` is
`dirname(Base.active_project())`. Every paper script starts with
`@quickactivate "Ptd"`, which activates this package unless it is already
active (for example through `--project`), so the scripts write their
figures to `<package>/Graphs`. Under any other active project the path
points into that project instead: `Pkg.test`, for example, starts in a
temporary environment, so tests always pass explicit output paths. The
path is computed at each call, never fixed at precompilation. The folder is
not created here; every writer creates its own destination folder.

# Arguments
- `parts::AbstractString...`: optional path components below `Graphs/`,
  usually one file name such as `"hsz_policy_paths.png"`. With none, the
  path of the folder itself is returned.

# Returns
- `String`: the absolute path `<active project>/Graphs/<parts...>`.
"""
function graphs_dir(parts::AbstractString...)

    path = projectdir("Graphs", parts...)

    return path
end

"""
    _resolve_symbolic_links(path)

Resolve the symbolic links in the existing leading part of `path`.

`realpath` accepts only existing paths, but `portable_path` also names
files that do not exist yet. This helper resolves the longest leading
part of `path` that exists and appends the remaining components as they
are.

# Arguments
- `path::AbstractString`: absolute path; it need not exist.

# Returns
- `String`: the physical spelling of `path`, for example
  `/private/var/folders/x/run.jld2` for `/var/folders/x/run.jld2` on
  macOS, where `/var` links to `/private/var`; `path` itself when no
  leading part of it exists.
"""
function _resolve_symbolic_links(path::AbstractString)

    existing = String(path)
    pending = String[]
    while !ispath(existing) && dirname(existing) != existing
        name = basename(existing)
        isempty(name) || pushfirst!(pending, name)
        existing = dirname(existing)
    end
    resolved = ispath(existing) ?
        joinpath(realpath(existing), pending...) : String(path)

    return resolved
end

"""
    portable_path(path; root=PROJECT_ROOT)

Return a spelling of `path` that is safe to save in an artifact.

Saved results must not reveal the folders of the machine that made them.
A file inside the package is therefore recorded relative to the package
root, and any other file by its name alone. The SHA-256 digest saved next
to the path identifies the file; the path only says where the package
keeps it. A relative `path` is resolved against the working directory,
as the scripts resolve their command-line arguments. Symbolic links in
`path` and `root` are resolved before they are compared, so a file that
is reached through a link, such as macOS's `/var` for `/private/var`,
still counts as inside the package.

# Arguments
- `path::AbstractString`: file path, absolute or relative to the working
  directory; it need not exist.
- `root::AbstractString`: folder that a relative result refers to;
  defaults to `PROJECT_ROOT`, the package root.

# Returns
- `String`: the path relative to `root` with `/` separators, such as
  `"data/iid-contest.jld2"`, when `path` lies inside `root`; otherwise
  the file name alone, as given in `path`, such as `"iid-contest.jld2"`.
"""
function portable_path(path::AbstractString;
                       root::AbstractString=PROJECT_ROOT)

    absolute = _resolve_symbolic_links(abspath(path))
    relative = relpath(absolute, _resolve_symbolic_links(abspath(root)))
    components = splitpath(relative)
    inside = !isabspath(relative) && first(components) != ".."
    portable = inside ? join(components, '/') : basename(abspath(path))

    return portable
end

"""Require outputs inside the project's data directory to be flat JLD2 files."""
function _validate_data_output(path::AbstractString; data_root=joinpath(PROJECT_ROOT, "data"))
    destination = abspath(path)
    root = abspath(data_root)
    relative = relpath(destination, root)
    if relative == "." || first(splitpath(relative)) != ".."
        dirname(destination) == root &&
            lowercase(splitext(destination)[2]) == ".jld2" ||
            throw(ArgumentError("data/ accepts only top-level .jld2 files: $destination"))
    end
    return destination
end

"""Reserve a flat JLD2 staging file beside its destination for atomic publication."""
function _temporary_jld_path(destination::AbstractString)
    destination = _validate_data_output(destination)
    mkpath(dirname(destination))
    path = tempname(dirname(destination); cleanup=false, suffix=".jld2")
    flags = Base.Filesystem.JL_O_RDWR | Base.Filesystem.JL_O_CREAT | Base.Filesystem.JL_O_EXCL
    close(Base.Filesystem.open(path, flags, 0o600))
    return path
end
