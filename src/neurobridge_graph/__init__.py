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
