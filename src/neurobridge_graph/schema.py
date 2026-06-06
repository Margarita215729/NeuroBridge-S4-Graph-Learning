"""Graph schema constants for NeuroBridge-S4 Graph Learning Phase 1."""

DOMAIN_NODES = [
    "Cardiovascular regulation",
    "Metabolic regulation",
    "Inflammation / immune-adjacent status",
    "Body composition / physical status",
    "Hematologic / oxygen-carrying capacity",
    "Sleep / circadian regulation",
    "Autonomic regulation",
    "Cognitive load",
    "Emotional regulation",
    "Recovery capacity",
    "Monitoring priority",
    "Countermeasure consideration",
]

CONCEPTUAL_EDGES = [
    ("Sleep / circadian regulation", "Autonomic regulation"),
    ("Sleep / circadian regulation", "Cognitive load"),
    ("Sleep / circadian regulation", "Recovery capacity"),
    ("Autonomic regulation", "Recovery capacity"),
    ("Inflammation / immune-adjacent status", "Recovery capacity"),
    ("Metabolic regulation", "Recovery capacity"),
    ("Cognitive load", "Emotional regulation"),
    ("Recovery capacity", "Monitoring priority"),
    ("Monitoring priority", "Countermeasure consideration"),
]

NODE_ATTRIBUTES = [
    "domain_score",
    "signed_deviation",
    "percentile_rank",
    "missingness",
    "baci_contribution",
    "data_source",
    "confidence_level",
    "phase",
]

EDGE_ATTRIBUTES = [
    "edge_type",
    "weight",
    "source",
    "directional",
    "confidence",
]
