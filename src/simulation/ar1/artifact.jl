# Final AR(1) contest-welfare artifact construction and authentication.
using Dates
using DrWatson: datadir

using ..Ptd: build_contest_artifact,
             load_contest_artifact,
             publish_contest_artifact

export AR1_CONTEST_ARTIFACT_SCHEMA_VERSION,
       ar1_contest_welfare_path,
       load_ar1_contest_welfare,
       publish_ar1_contest

const AR1_CONTEST_ARTIFACT_SCHEMA_VERSION = "ptd-ar1-contest-v3"

"""Return the canonical final AR(1) contest path below `root`."""
function ar1_contest_welfare_path(root::AbstractString,
                                    run_fingerprint::AbstractString="")
    return joinpath(abspath(root), "ar1-contest-welfare.jld2")
end

"""Load and authenticate a final AR(1) contest artifact."""
function load_ar1_contest_welfare(path::AbstractString)
    return load_contest_artifact(path;
        expected_schema=AR1_CONTEST_ARTIFACT_SCHEMA_VERSION)
end

"""
    publish_ar1_contest(config, welfare, timing; output_path, context, ...)

Assemble and atomically publish the sole AR(1) contest artifact after all
requested contest slots have been attempted, retaining any failures. No intermediate simulation file is
created.
"""
function publish_ar1_contest(config::AR1ContestRunConfig, welfare, timing;
    output_path::AbstractString=ar1_contest_welfare_path(datadir()),
    context=nothing,
    confidence_level::Real=0.95,
    completed_at::DateTime=now(UTC),
    compress::Bool=true)

    artifact = build_contest_artifact(
        artifact_schema=AR1_CONTEST_ARTIFACT_SCHEMA_VERSION,
        scientific_configuration=scientific_configuration(config),
        welfare=welfare,
        timing=timing,
        methods=config.methods,
        panels=(AR1_PANEL,),
        n_paths=config.n_paths,
        confidence_level=confidence_level,
        completed_at=completed_at,
    )
    path = publish_contest_artifact(output_path, artifact; compress)
    loaded = load_ar1_contest_welfare(path)
    return (;
        status=:computed,
        path,
        metadata=loaded["metadata"],
        artifact=loaded,
    )
end
