"""NeuroBridge-S4 Graph Learning package."""

__version__ = "0.1.0"

from neurobridge_graph.data_loader import (  # noqa: F401
    load_pseudo_crew,
    load_deviation_scores,
    load_domain_scores,
    load_baci_scores,
    load_baci_sensitivity,
    load_crew_level_summary,
    load_all,
)
from neurobridge_graph.interactive import (  # noqa: F401
    export_interactive_graph_html,
    export_all_graphs_html,
    create_interactive_index,
    export_index_html,
    validate_html_graph_output,
)
from neurobridge_graph.hazard_mapping import (  # noqa: F401
    get_default_hazard_domain_mapping,
    normalize_domain_name,
    compute_hazard_relevance_scores,
    compute_hazard_coverage,
    export_hazard_domain_mapping,
    interpret_hazard_score,
    HAZARD_CANONICAL,
    HAZARD_DISPLAY_NAMES,
    CORE_POSITIONING_SENTENCE,
)
from neurobridge_graph.embeddings import (  # noqa: F401
    load_phase4_feature_tables,
    build_hazard_aware_feature_matrix,
    select_numeric_features,
    scale_feature_matrix,
    compute_pca_embedding,
)
from neurobridge_graph.similarity import (  # noqa: F401
    compute_cosine_similarity_matrix,
    compute_euclidean_distance_matrix,
    summarize_pairwise_similarity,
    identify_most_similar_pair,
    identify_most_distinct_subject,
)
from neurobridge_graph.longitudinal import (  # noqa: F401
    detect_longitudinal_columns,
    validate_longitudinal_table,
    create_example_longitudinal_table,
    build_timepoint_graphs,
    identify_baseline_timepoint,
    compute_node_activation_delta,
    compute_graph_metric_delta,
    compute_subject_trajectory_table,
    compute_longitudinal_delta_tables,
    compute_hazard_scores_per_timepoint,
    sanitize_graph_for_graphml,
    SCHEMA_DEMO_DATA_TYPE,
    PRIMARY_SIGNAL_STATEMENT,
    SELF_BASELINE_STATEMENT,
)
from neurobridge_graph.trajectory_features import (  # noqa: F401
    compute_recovery_slope,
    compute_time_to_baseline_like_state,
    compute_recovery_fraction,
    compute_hazard_context_delta,
    compute_recovery_metrics_table,
    identify_dominant_trajectory_shift,
    derive_longitudinal_hazard_deltas,
    ensure_longitudinal_hazard_deltas,
    LONGITUDINAL_HAZARD_DELTA_COLUMNS,
)
from neurobridge_graph.trajectory_attribution import (  # noqa: F401
    load_phase6_delta_tables,
    compute_node_attribution,
    compute_graph_metric_attribution,
    compute_subgraph_attribution_from_node_deltas,
    compute_hazard_context_attribution,
    compute_recovery_attribution,
    build_phase7_attribution_summary,
    DEFAULT_SUBGRAPH_TEMPLATES,
)
from neurobridge_graph.explanation_generator import (  # noqa: F401
    generate_subject_timepoint_explanation,
    generate_phase7_report,
)
from neurobridge_graph.attribution_visualization import (  # noqa: F401
    plot_node_attribution_bar,
    plot_subgraph_attribution_heatmap,
    plot_hazard_context_attribution_heatmap,
    plot_recovery_attribution_summary,
    plot_subject_explanation_panel,
)
from neurobridge_graph.reference_envelope import (  # noqa: F401
    load_reference_envelope_inputs,
    create_example_reference_envelope,
    build_envelope_from_reference_deltas,
    build_envelope_from_summary_table,
    compute_robust_z_score,
    classify_envelope_position,
    score_node_deltas_against_envelope,
    score_graph_deltas_against_envelope,
    score_hazard_deltas_against_envelope,
    build_phase8_envelope_summary,
    CORE_ENVELOPE_STATEMENT,
    SCHEMA_DEMO_DATA_TYPE as ENVELOPE_SCHEMA_DEMO_DATA_TYPE,
)
from neurobridge_graph.envelope_visualization import (  # noqa: F401
    plot_domain_delta_envelope,
    plot_graph_metric_envelope,
    plot_hazard_delta_envelope,
    plot_envelope_exceedance_heatmap,
    plot_reference_envelope_overview,
)
from neurobridge_graph.envelope_reporting import (  # noqa: F401
    generate_envelope_interpretation,
    generate_phase8_report,
)
from neurobridge_graph.dashboard_data import (  # noqa: F401
    load_dashboard_tables,
    has_required_tables,
    build_dashboard_readiness_report,
    get_available_subjects,
    get_available_timepoints,
    get_subject_timepoint_context,
    filter_subject_timepoint,
    get_domain_delta_panel_data,
    get_graph_metric_panel_data,
    get_hazard_context_panel_data,
    get_attribution_panel_data,
    get_envelope_panel_data,
    get_recovery_panel_data,
    load_resilience_tables,
    get_resilience_panel_data,
    REQUIRED_TABLES as DASHBOARD_REQUIRED_TABLES,
    OPTIONAL_TABLES as DASHBOARD_OPTIONAL_TABLES,
)
from neurobridge_graph.dashboard_text import (  # noqa: F401
    DASHBOARD_GUARDRAIL,
    get_dashboard_intro_text,
    get_guardrail_text,
)
from neurobridge_graph.data_validation import (  # noqa: F401
    normalize_column_name,
    detect_table_format,
    detect_standard_columns,
    validate_required_columns,
    validate_longitudinal_structure,
    summarize_missingness,
    build_input_readiness_report,
)
from neurobridge_graph.domain_mapping import (  # noqa: F401
    CANONICAL_DOMAINS,
    get_default_variable_domain_mapping,
    normalize_variable_name,
    map_variable_to_domain,
    map_variables_dataframe,
    build_domain_coverage_report,
    INTENTIONALLY_UNMAPPED_NOTE,
)
from neurobridge_graph.data_adapters import (  # noqa: F401
    create_data_templates,
    standardize_wide_longitudinal_table,
    standardize_long_longitudinal_table,
    combine_standardized_streams,
    standardize_units_if_known,
    compute_variable_baseline_deltas,
    build_domain_scores_from_variables,
    pivot_domain_scores_wide,
    run_adapter_pipeline,
    SCHEMA_TEMPLATE_DATA_TYPE,
    STANDARDIZED_COLUMNS,
    DOMAIN_SCORE_LONG_COLUMNS,
    UNIT_CONVERSION_STATUSES,
)
from neurobridge_graph.adapter_reporting import (  # noqa: F401
    generate_adapter_report,
)
from neurobridge_graph.resilience_rules import (  # noqa: F401
    RESILIENCE_STATES,
    DOMINANT_ADAPTATION_MODES,
    DEFAULT_THRESHOLDS as RESILIENCE_DEFAULT_THRESHOLDS,
    classify_resilience_state,
    classify_dominant_adaptation_mode,
    evaluate_coverage_limitations,
    derive_resilience_evidence,
    build_evidence_chain,
)
from neurobridge_graph.resilience_interpretation import (  # noqa: F401
    load_phase11_inputs,
    build_phase11_input_readiness_report,
    core_inputs_available,
    get_subject_timepoint_pairs,
    interpret_subject_timepoint_resilience,
    build_resilience_state_table,
    build_mission_relevance_translation,
)
from neurobridge_graph.resilience_reporting import (  # noqa: F401
    generate_resilience_card,
    generate_phase11_report,
)
from neurobridge_graph.resilience_visualization import (  # noqa: F401
    plot_resilience_state_summary,
    plot_resilience_state_timeline,
    plot_adaptation_mode_heatmap,
    plot_evidence_chain_summary,
)

# Phase 12 (PyTorch) is optional: guard the import so the package still loads if
# torch is not installed in a given environment.
try:  # noqa: SIM105
    from neurobridge_graph.torch_dataset import (  # noqa: F401
        load_phase12_input_tables,
        build_phase12_input_readiness_report,
        build_trajectory_feature_matrix,
        encode_resilience_metadata,
        select_numeric_model_features,
        scale_model_features,
        create_masked_training_data,
        build_mask_summary,
        TrajectoryFeatureDataset,
    )
    from neurobridge_graph.torch_autoencoder import (  # noqa: F401
        TrajectoryAutoencoder,
        choose_latent_dim,
        count_trainable_parameters,
    )
    from neurobridge_graph.torch_training import (  # noqa: F401
        set_torch_seed,
        resolve_device,
        sufficient_for_training,
        train_autoencoder,
        compute_embeddings_and_reconstructions,
        compute_reconstruction_error_table,
        compute_similarity_matrix,
        save_model_state,
    )
    from neurobridge_graph.torch_reporting import (  # noqa: F401
        generate_phase12_model_card,
        generate_phase12_showcase_report,
        build_resilience_consistency_view,
    )
    from neurobridge_graph.torch_showcase import (  # noqa: F401
        create_phase12_showcase_html,
        generate_phase12_figures,
    )
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch is an optional dependency
    _TORCH_AVAILABLE = False
