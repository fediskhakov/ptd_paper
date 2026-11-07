# Final IID contest artifact construction and authentication.
using Dates
using DrWatson: datadir

using ..Ptd: build_contest_artifact,
             load_contest_artifact,
             publish_contest_artifact

export IID_CONTEST_ARTIFACT_SCHEMA_VERSION,
       iid_contest_path,
       load_iid_contest,
       publish_iid_contest

include("legacy_artifact.jl")

const IID_CONTEST_ARTIFACT_SCHEMA_VERSION = "ptd-iid-contest-v3"

"""Return the canonical final IID contest path below `root`."""
function iid_contest_path(root::AbstractString,
                                run_fingerprint::AbstractString="")
    return joinpath(abspath(root), "iid-contest.jld2")
end

"""Load and authenticate a final IID contest artifact."""
function load_iid_contest(path::AbstractString)
    isfile(path) || throw(ArgumentError("contest artifact does not exist: $path"))
    artifact = JLD2.load(path)
    metadata = _processing_field(artifact, :metadata)
    if String(_processing_field(metadata, :artifact_schema)) ==
            PROCESSED_CONTEST_SCHEMA_VERSION
        return _load_legacy_iid_contest(artifact)
    end
    return load_contest_artifact(path;
        expected_schema=IID_CONTEST_ARTIFACT_SCHEMA_VERSION)
end

"""
    publish_iid_contest(config, welfare, timing; output_path, context, ...)

Assemble and atomically publish the sole IID contest artifact after all
requested contest slots have been attempted, retaining any failures. No intermediate simulation file is
created.
"""
function publish_iid_contest(config::IIDContestRunConfig, welfare, timing;
    output_path::AbstractString=iid_contest_path(datadir()),
    context=nothing,
    confidence_level::Real=0.95,
    completed_at::DateTime=now(UTC),
    compress::Bool=true)

    artifact = build_contest_artifact(
        artifact_schema=IID_CONTEST_ARTIFACT_SCHEMA_VERSION,
        scientific_configuration=scientific_configuration(config),
        welfare=welfare,
        timing=timing,
        methods=config.methods,
        panels=keys(config.panels),
        n_paths=config.n_paths,
        confidence_level=confidence_level,
        completed_at=completed_at,
    )
    path = publish_contest_artifact(output_path, artifact; compress)
    loaded = load_iid_contest(path)
    return (;
        status=:computed,
        path,
        metadata=loaded["metadata"],
        artifact=loaded,
    )
end
