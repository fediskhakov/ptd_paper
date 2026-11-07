# Read-only compatibility with the archived IID processed-v2 format.
# Retains the original validation and terminology normalization of the
# former src/simulation/iid/processing.jl, removed when contest chunks were
# replaced by complete runs. No chunk runner or writer is restored.
using JLD2
using SHA
using ..Ptd: ContestSummary, PairedComparison, aggregate_outcomes, stable_fingerprint

const PROCESSED_CONTEST_SCHEMA_VERSION = "ptd-iid-contest-processed-v2"
const PROCESSED_AUTHENTICATION_SCHEMA_VERSION =
    "ptd-iid-contest-processed-authentication-v2"
const SOURCE_CATALOG_SCHEMA_VERSION = "ptd-iid-contest-source-catalog-v1"
const UTILITY_PAYLOAD_SCHEMA_VERSION = "ptd-iid-contest-utilities-v1"

function _processing_has_direct_field(container, name::Symbol)
    return container isa NamedTuple ? hasproperty(container, name) :
        container isa AbstractDict &&
        (haskey(container, name) || haskey(container, String(name)))
end

function _processing_field(container, name::Symbol, location::AbstractString)
    legacy_name = name === :simulation_metadata ? :replication_metadata :
        name === :simulation_metadata_fingerprint ?
            :replication_metadata_fingerprint : nothing
    candidates = isnothing(legacy_name) ? (name,) : (name, legacy_name)
    if container isa NamedTuple
        for candidate in candidates
            hasproperty(container, candidate) &&
                return getproperty(container, candidate)
        end
    elseif container isa AbstractDict
        for candidate in candidates
            haskey(container, candidate) && return container[candidate]
            haskey(container, String(candidate)) &&
                return container[String(candidate)]
        end
    end
    throw(ArgumentError("$location is missing required field $name"))
end

_processing_field(container, name::Symbol) = _processing_field(container, name, "artifact")

_iid_paper_name(value) = value
_iid_paper_name(value::Symbol) = value === :sceq_ptd ? :ptd_sceq :
    value === :oa_ptd ? :ptd_osa :
    value === :oa_options ? :ptd_osa_options :
    value === :replication_metadata ? :simulation_metadata :
    value === :replication_metadata_fingerprint ?
        :simulation_metadata_fingerprint : value
_iid_paper_name(value::String) =
    value == "SCEQ-PTD" ? "PTD-SCEQ" :
    value == "OA-PTD" ? "PTD-OSA" :
    value == "sceq-ptd-conditional-mean-v1" ?
        "ptd-sceq-conditional-mean-v1" :
    value == "oa-ptd-certified-five-node-v1" ?
        "ptd-osa-certified-five-node-v1" :
    value == "replication_metadata" ? "simulation_metadata" :
    value == "replication_metadata_fingerprint" ?
        "simulation_metadata_fingerprint" : value

function _iid_paper_name(value::NamedTuple)
    names = Tuple(_iid_paper_name.(keys(value)))
    return NamedTuple{names}(Tuple(_iid_paper_name.(values(value))))
end

_iid_paper_name(value::Tuple) = map(_iid_paper_name, value)
_iid_paper_name(value::AbstractRange) = value
_iid_paper_name(value::AbstractArray) = map(_iid_paper_name, value)
_iid_paper_name(value::AbstractDict) = Dict(
    _iid_paper_name(key) => _iid_paper_name(item)
    for (key, item) in pairs(value))

function _iid_paper_named_artifact(artifact::AbstractDict)
    normalized = Dict{String,Any}(_iid_paper_name(String(name)) =>
        _iid_paper_name(value)
        for (name, value) in pairs(artifact))
    scientific = normalized["scientific_configuration"]
    summaries = normalized["contest_summaries"]
    aggregates = normalized["panel_aggregates"]
    source_catalog = normalized["source_catalog"]
    timings = normalized["timings"]
    simulation = normalized["simulation_metadata"]
    scientific_fingerprint_value = stable_fingerprint(scientific)
    summaries_fingerprint = stable_fingerprint(summaries)
    aggregates_fingerprint = stable_fingerprint(aggregates)
    source_catalog_fingerprint =
        _source_catalog_fingerprint(source_catalog, timings)
    simulation_metadata_fingerprint = stable_fingerprint(simulation)
    metadata = merge(normalized["metadata"], (;
        scientific_fingerprint=scientific_fingerprint_value,
        summaries_fingerprint,
        aggregates_fingerprint,
        source_catalog_fingerprint,
        simulation_metadata_fingerprint))
    result_fingerprint = stable_fingerprint((;
        utilities_fingerprint=metadata.utilities_fingerprint,
        summaries_fingerprint,
        aggregates_fingerprint))
    metadata = merge(metadata, (; result_fingerprint))
    metadata = merge(metadata, (authentication_fingerprint=
        _processed_authentication(metadata),))
    normalized["metadata"] = metadata
    return normalized
end

function _stored_comparison(value, location::AbstractString)
    comparison = PairedComparison(
        Symbol(_processing_field(value, :method_a, location)),
        Symbol(_processing_field(value, :method_b, location)),
        Int(_processing_field(value, :n_paths, location)),
        Float64(_processing_field(value, :confidence_level, location)),
        Float64(_processing_field(value, :mean_difference, location)),
        Float64(_processing_field(value, :standard_error, location)),
        Float64(_processing_field(value, :ci_lower, location)),
        Float64(_processing_field(value, :ci_upper, location)),
    )
    comparison.method_a != comparison.method_b || throw(ArgumentError(
        "$location compares one method with itself"))
    comparison.n_paths >= 2 || throw(ArgumentError(
        "$location has fewer than two paired paths"))
    0.0 < comparison.confidence_level < 1.0 || throw(ArgumentError(
        "$location has an invalid confidence level"))
    all(isfinite, (comparison.mean_difference, comparison.standard_error,
        comparison.ci_lower, comparison.ci_upper)) || throw(ArgumentError(
            "$location has a nonfinite statistic"))
    comparison.standard_error >= 0.0 || throw(ArgumentError(
        "$location has a negative standard error"))
    comparison.ci_lower <= comparison.ci_upper || throw(ArgumentError(
        "$location has a reversed confidence interval"))
    return comparison
end

function _stored_summary(value, index::Int)
    location = "contest_summaries[$index]"
    methods = Symbol.(collect(_processing_field(value, :method_ids, location)))
    expected = Float64.(collect(_processing_field(value, :expected_utility, location)))
    point_winners = Symbol.(collect(_processing_field(value, :point_winners, location)))
    comparisons = [_stored_comparison(comparison,
        "$location.comparisons[$comparison_index]")
        for (comparison_index, comparison) in
            enumerate(_processing_field(value, :comparisons, location))]
    point_winner_value = _processing_field(value, :point_winner, location)
    resolved_winner_value = _processing_field(value, :resolved_winner, location)
    summary = ContestSummary(
        Int(_processing_field(value, :contest_id, location)),
        Symbol(_processing_field(value, :panel, location)),
        methods,
        Int(_processing_field(value, :n_paths, location)),
        Float64(_processing_field(value, :confidence_level, location)),
        expected,
        point_winners,
        isnothing(point_winner_value) ? nothing : Symbol(point_winner_value),
        isnothing(resolved_winner_value) ? nothing :
            Symbol(resolved_winner_value),
        Symbol(_processing_field(value, :classification, location)),
        comparisons,
    )
    summary.contest_id > 0 || throw(ArgumentError(
        "$location has a nonpositive contest identifier"))
    summary.panel in (:baseline, :high_variance) || throw(ArgumentError(
        "$location has an unknown panel"))
    length(summary.method_ids) >= 2 &&
        length(unique(summary.method_ids)) == length(summary.method_ids) ||
        throw(ArgumentError("$location has invalid method identifiers"))
    length(summary.expected_utility) == length(summary.method_ids) &&
        all(isfinite, summary.expected_utility) || throw(ArgumentError(
            "$location has invalid expected utilities"))
    summary.n_paths >= 2 || throw(ArgumentError(
        "$location has fewer than two paths"))
    0.0 < summary.confidence_level < 1.0 || throw(ArgumentError(
        "$location has an invalid confidence level"))
    summary.classification in (:resolved, :mc_unresolved, :point_tie) ||
        throw(ArgumentError("$location has an invalid classification"))
    return summary
end

function _aggregates_from_stored_summaries(stored_summaries,
                                           methods::Vector{Symbol},
                                           panels::Vector{Symbol},
                                           outer_design,
                                           n_paths::Int,
                                           confidence_level::Float64)
    design_by_id = Dict{Int,Any}()
    for (index, entry) in enumerate(outer_design)
        contest_id = Int(_processing_field(entry, :id, "outer_design[$index]"))
        contest_id > 0 || throw(ArgumentError(
            "outer_design[$index] has a nonpositive contest identifier"))
        haskey(design_by_id, contest_id) && throw(ArgumentError(
            "outer_design contains duplicate contest identifier $contest_id"))
        design_by_id[contest_id] = entry
    end
    expected_ids = Set(keys(design_by_id))
    summaries = ContestSummary[]
    for (index, value) in enumerate(stored_summaries)
        summary = _stored_summary(value, index)
        location = "contest_summaries[$index]"
        summary.method_ids == methods || throw(ArgumentError(
            "$location method identifiers do not match processed metadata"))
        summary.n_paths == n_paths || throw(ArgumentError(
            "$location path count does not match processed metadata"))
        summary.confidence_level == confidence_level || throw(ArgumentError(
            "$location confidence level does not match processed metadata"))
        haskey(design_by_id, summary.contest_id) || throw(ArgumentError(
            "$location has no matching outer-design entry"))
        design_entry = design_by_id[summary.contest_id]
        for field in (:beta, :gamma, :initial_cash)
            Float64(_processing_field(value, field, location)) ==
                Float64(_processing_field(design_entry, field,
                    "outer_design contest $(summary.contest_id)")) ||
                throw(ArgumentError(
                    "$location $field does not match its outer-design entry"))
        end
        push!(summaries, summary)
    end
    aggregates = NamedTuple[]
    for panel in panels
        selected = filter(summary -> summary.panel == panel, summaries)
        Set(getfield.(selected, :contest_id)) == expected_ids ||
            throw(ArgumentError(
                "stored summaries do not contain every outer contest for panel $panel"))
        push!(aggregates, aggregate_outcomes(selected))
    end
    length(summaries) == length(panels) * length(outer_design) ||
        throw(ArgumentError("processed artifact has an unexpected summary count"))
    return aggregates
end

function _update_identity!(context, value)
    fingerprint = stable_fingerprint(value)
    SHA.update!(context, collect(codeunits(string(length(fingerprint), ':',
        fingerprint))))
    return context
end

function _source_catalog_fingerprint(source_catalog, timings)
    context = SHA.SHA2_256_CTX()
    SHA.update!(context, collect(codeunits(SOURCE_CATALOG_SCHEMA_VERSION)))
    _update_identity!(context, _processing_field(source_catalog, :manifest,
        "source_catalog"))
    groups = _processing_field(source_catalog, :chunk_groups, "source_catalog")
    for (group_index, group) in enumerate(groups)
        chunks = _processing_field(group, :chunks, "chunk_groups[$group_index]")
        provenance = _processing_field(group, :provenance,
            "chunk_groups[$group_index]")
        header = (;
            contest_id=_processing_field(group, :contest_id),
            panel=_processing_field(group, :panel),
            method=_processing_field(group, :method),
            status=_processing_field(group, :status),
            completed_utc=_processing_field(group, :completed_utc),
            block_fingerprint=_processing_field(group, :block_fingerprint),
            provenance_fingerprint=stable_fingerprint(provenance),
        )
        _update_identity!(context, header)
        for chunk in chunks
            _update_identity!(context, chunk)
        end
    end
    for timing in timings
        _update_identity!(context, timing)
    end
    return bytes2hex(SHA.digest!(context))
end

function _processed_authentication_fields(metadata)
    return (;
        schema=PROCESSED_AUTHENTICATION_SCHEMA_VERSION,
        artifact_schema=PROCESSED_CONTEST_SCHEMA_VERSION,
        status=_processing_field(metadata, :status, "metadata"),
        completed_utc=String(_processing_field(metadata, :completed_utc, "metadata")),
        scientific_fingerprint=String(
            _processing_field(metadata, :scientific_fingerprint, "metadata")),
        run_fingerprint=String(_processing_field(metadata, :run_fingerprint, "metadata")),
        source_commit=String(_processing_field(metadata, :source_commit, "metadata")),
        input_tree_fingerprint=String(
            _processing_field(metadata, :input_tree_fingerprint, "metadata")),
        environment_fingerprint=String(
            _processing_field(metadata, :environment_fingerprint, "metadata")),
        confidence_level=Float64(
            _processing_field(metadata, :confidence_level, "metadata")),
        n_contests=Int(_processing_field(metadata, :n_contests, "metadata")),
        n_paths=Int(_processing_field(metadata, :n_paths, "metadata")),
        n_methods=Int(_processing_field(metadata, :n_methods, "metadata")),
        n_panels=Int(_processing_field(metadata, :n_panels, "metadata")),
        n_chunks=Int(_processing_field(metadata, :n_chunks, "metadata")),
        n_timing_receipts=Int(
            _processing_field(metadata, :n_timing_receipts, "metadata")),
        utility_payload_shape=Tuple(Int.(collect(
            _processing_field(metadata, :utility_payload_shape, "metadata")))),
        outer_design_fingerprint=String(
            _processing_field(metadata, :outer_design_fingerprint, "metadata")),
        source_catalog_fingerprint=String(
            _processing_field(metadata, :source_catalog_fingerprint, "metadata")),
        utilities_fingerprint=String(
            _processing_field(metadata, :utilities_fingerprint, "metadata")),
        summaries_fingerprint=String(
            _processing_field(metadata, :summaries_fingerprint, "metadata")),
        aggregates_fingerprint=String(
            _processing_field(metadata, :aggregates_fingerprint, "metadata")),
        result_fingerprint=String(
            _processing_field(metadata, :result_fingerprint, "metadata")),
    )
end

function _processed_authentication(metadata)
    fields = _processed_authentication_fields(metadata)
    return stable_fingerprint(merge(fields, (simulation_metadata_fingerprint=
        String(_processing_field(
            metadata, :simulation_metadata_fingerprint, "metadata")),)))
end

function _legacy_processed_authentication(metadata)
    fields = _processed_authentication_fields(metadata)
    return stable_fingerprint(merge(fields, (replication_metadata_fingerprint=
        String(_processing_field(
            metadata, :replication_metadata_fingerprint, "metadata")),)))
end

function _validate_processed_artifact(artifact)
    artifact isa AbstractDict || throw(ArgumentError(
        "processed contest artifact must be a dictionary"))
    metadata = _processing_field(artifact, :metadata)
    String(_processing_field(metadata, :artifact_schema, "metadata")) ==
        PROCESSED_CONTEST_SCHEMA_VERSION || throw(ArgumentError(
            "processed contest schema mismatch"))
    _processing_field(metadata, :status, "metadata") == :complete || throw(ArgumentError(
        "processed contest artifact is not complete"))
    completed = String(_processing_field(metadata, :completed_utc, "metadata"))
    !isempty(completed) && endswith(completed, 'Z') || throw(ArgumentError(
        "processed completion timestamp is invalid"))

    scientific = _processing_field(artifact, :scientific_configuration)
    stable_fingerprint(scientific) == String(
        _processing_field(metadata, :scientific_fingerprint, "metadata")) ||
        throw(ArgumentError("processed scientific configuration fingerprint mismatch"))
    methods = Symbol.(collect(_processing_field(artifact, :methods)))
    panels = Symbol.(collect(_processing_field(artifact, :panels)))
    outer_design = _processing_field(artifact, :outer_design)
    Tuple(methods) == Tuple(_processing_field(scientific, :methods,
        "scientific_configuration")) || throw(ArgumentError(
            "processed method identifiers do not match the scientific configuration"))
    scientific_panels = Symbol[_processing_field(panel, :id,
        "scientific_configuration.model.panels") for panel in
        _processing_field(_processing_field(scientific, :model, "scientific_configuration"),
            :panels, "scientific_configuration.model")]
    panels == scientific_panels || throw(ArgumentError(
        "processed panel identifiers do not match the scientific configuration"))
    stable_fingerprint(outer_design) == String(_processing_field(
        metadata, :outer_design_fingerprint, "metadata")) ||
        throw(ArgumentError("processed outer-design fingerprint mismatch"))
    expected_dimensions = (
        Int(_processing_field(metadata, :n_paths, "metadata")),
        Int(_processing_field(metadata, :n_methods, "metadata")),
        Int(_processing_field(metadata, :n_contests, "metadata")),
        Int(_processing_field(metadata, :n_panels, "metadata")),
    )
    utility_payload_shape = Tuple(Int.(collect(
        _processing_field(metadata, :utility_payload_shape, "metadata"))))
    utility_payload_shape == expected_dimensions || throw(DimensionMismatch(
        "processed utility payload shape does not match metadata"))
    length(methods) == expected_dimensions[2] || throw(DimensionMismatch(
        "processed method count does not match metadata"))
    length(outer_design) == expected_dimensions[3] || throw(DimensionMismatch(
        "processed outer-design count does not match metadata"))
    length(panels) == expected_dimensions[4] || throw(DimensionMismatch(
        "processed panel count does not match metadata"))

    summaries = _processing_field(artifact, :contest_summaries)
    aggregates = _processing_field(artifact, :panel_aggregates)
    confidence_level = Float64(
        _processing_field(metadata, :confidence_level, "metadata"))
    regenerated_aggregates = _aggregates_from_stored_summaries(
        summaries, methods, panels, outer_design, expected_dimensions[1],
        confidence_level)
    stable_fingerprint(aggregates) == stable_fingerprint(regenerated_aggregates) ||
        throw(ArgumentError("stored panel aggregates do not match contest summaries"))
    stable_fingerprint(summaries) == String(
        _processing_field(metadata, :summaries_fingerprint, "metadata")) ||
        throw(ArgumentError("processed summary fingerprint mismatch"))
    stable_fingerprint(aggregates) == String(
        _processing_field(metadata, :aggregates_fingerprint, "metadata")) ||
        throw(ArgumentError("processed aggregate fingerprint mismatch"))

    source_catalog = _processing_field(artifact, :source_catalog)
    String(_processing_field(source_catalog, :schema, "source_catalog")) ==
        SOURCE_CATALOG_SCHEMA_VERSION || throw(ArgumentError(
            "processed source-catalog schema mismatch"))
    timings = _processing_field(artifact, :timings)
    source_fingerprint = _source_catalog_fingerprint(source_catalog, timings)
    source_fingerprint == String(
        _processing_field(metadata, :source_catalog_fingerprint, "metadata")) ||
        throw(ArgumentError("processed source-catalog fingerprint mismatch"))
    groups = _processing_field(source_catalog, :chunk_groups, "source_catalog")
    chunk_count = sum(length(_processing_field(group, :chunks)) for group in groups)
    chunk_count == Int(_processing_field(metadata, :n_chunks, "metadata")) ||
        throw(ArgumentError("processed chunk count does not match metadata"))
    length(timings) == Int(
        _processing_field(metadata, :n_timing_receipts, "metadata")) ||
        throw(ArgumentError("processed timing-receipt count does not match metadata"))
    all(group -> _processing_field(group, :status) == :authenticated, groups) ||
        throw(ArgumentError("processed source catalog contains an unauthenticated chunk group"))
    all(timing -> _processing_field(timing, :status) == :authenticated, timings) ||
        throw(ArgumentError("processed source catalog contains an unauthenticated timing receipt"))
    simulation = _processing_field(artifact, :simulation_metadata)
    stable_fingerprint(simulation) == String(_processing_field(
        metadata, :simulation_metadata_fingerprint, "metadata")) ||
        throw(ArgumentError("processed simulation-metadata fingerprint mismatch"))
    manifest_source = _processing_field(source_catalog, :manifest, "source_catalog")
    stable_fingerprint(simulation) == String(_processing_field(manifest_source,
        :simulation_metadata_fingerprint, "source_catalog.manifest")) ||
        throw(ArgumentError(
            "processed simulation metadata differs from the raw manifest"))

    utilities_fingerprint = String(
        _processing_field(metadata, :utilities_fingerprint, "metadata"))
    length(utilities_fingerprint) == 64 && all(character ->
        character in '0':'9' || character in 'a':'f',
        utilities_fingerprint) ||
        throw(ArgumentError("processed utility fingerprint is not SHA-256 hex"))
    result_fingerprint = stable_fingerprint((;
        utilities_fingerprint,
        summaries_fingerprint=stable_fingerprint(summaries),
        aggregates_fingerprint=stable_fingerprint(aggregates),
    ))
    result_fingerprint == String(
        _processing_field(metadata, :result_fingerprint, "metadata")) ||
        throw(ArgumentError("processed result fingerprint mismatch"))
    legacy_authentication =
        !_processing_has_direct_field(metadata, :simulation_metadata_fingerprint) &&
        _processing_has_direct_field(metadata, :replication_metadata_fingerprint)
    expected_authentication = legacy_authentication ?
        _legacy_processed_authentication(metadata) :
        _processed_authentication(metadata)
    expected_authentication == String(
        _processing_field(metadata, :authentication_fingerprint, "metadata")) ||
        throw(ArgumentError("processed artifact authentication mismatch"))
    return artifact
end

"""Validate a historical compact IID artifact before adapting method names."""
function _load_legacy_iid_contest(artifact)
    _validate_processed_artifact(artifact)
    methods = Symbol.(collect(_processing_field(artifact, :methods)))
    if any(method -> method === :sceq_ptd || method === :oa_ptd, methods)
        artifact = _iid_paper_named_artifact(artifact)
        _validate_processed_artifact(artifact)
    end
    return artifact
end
