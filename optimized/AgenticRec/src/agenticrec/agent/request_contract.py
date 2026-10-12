"""Shared explicit request defaults for extraction and planning prompts."""

FIXED_REQUEST_DEFAULTS = (
    " schema_version is integer 1. Defaults for absent facts are user_id null, "
    "history_authorized false, all ID/genre/required_fields arrays empty, "
    "year_min/year_max null, exclude_seen true, k 5. Item ID references in text "
    "become liked_item_ids for similar recommendations. Unknown requested fields "
    "such as duration belong in required_fields. Preserve explicit conflicting "
    "include/exclude conditions for deterministic clarification."
)
