"""
Component 54: Production System - Rule Factories (Compatibility Wrapper)

This module provides backwards compatibility by re-exporting all rule factory functions
from the refactored modular structure.

DEPRECATED: This wrapper exists for backward compatibility only.
For new code, prefer importing directly from the specialized modules:
- component_54_production_rule_factories: Helper functions (calculate_specificity, create_production_rule)
- component_54_production_rules_content: Content selection rules
- component_54_production_rules_lexical: Lexicalization rules
- component_54_production_rules_discourse: Discourse management rules
- component_54_production_rules_syntax: Syntactic realization rules
- component_54_production_rules_aggregator: Aggregator functions

Author: KAI Development Team
Date: 2025-11-21
"""

# Re-export factory utilities
from component_54_production_rule_factories import (
    calculate_specificity,
    create_production_rule,
)

# Re-export aggregator functions
from component_54_production_rules_aggregator import (
    create_all_phase3_rules,
    create_all_phase4_rules,
    create_complete_production_system,
)

# Re-export content selection rules
from component_54_production_rules_content import (
    create_aggregate_multi_source_rule,
    create_all_content_selection_rules,
    create_finish_content_selection_rule,
    create_prefer_direct_fact_rule,
    create_prioritize_high_confidence_rule,
    create_require_high_confidence_rule,
    create_select_capability_fact_rule,
    create_select_definition_rule,
    create_select_is_a_fact_rule,
    create_select_location_fact_rule,
    create_select_part_of_fact_rule,
    create_select_property_fact_rule,
    create_select_synonym_rule,
    create_skip_low_confidence_rule,
    create_skip_uncertain_facts_rule,
    create_warn_medium_confidence_rule,
)

# Re-export discourse management rules
from component_54_production_rules_discourse import (
    create_all_discourse_management_rules,
    create_introduce_simple_rule,
    create_introduce_with_context_rule,
)

# Re-export lexicalization rules
from component_54_production_rules_lexical import (
    create_add_elaboration_rule,
    create_all_lexicalization_rules,
    create_avoid_repetition_rule,
    create_combine_facts_conjunction_rule,
    create_compress_similar_facts_rule,
    create_finish_lexicalization_rule,
    create_select_casual_style_rule,
    create_select_formal_style_rule,
    create_vary_copula_verb_rule,
    create_verbalize_capable_of_rule,
    create_verbalize_has_property_rule,
    create_verbalize_is_a_simple_rule,
    create_verbalize_is_a_variant_1_rule,
    create_verbalize_is_a_variant_2_rule,
    create_verbalize_located_in_rule,
    create_verbalize_part_of_rule,
)

# Re-export syntactic realization rules
from component_54_production_rules_syntax import (
    create_add_article_accusative_rule,
    create_add_article_nominative_rule,
    create_all_syntactic_realization_rules,
    create_capitalize_nouns_rule,
    create_capitalize_sentence_start_rule,
    create_finish_sentence_rule,
    create_order_sentence_elements_rule,
)

__all__ = [
    # Factory utilities
    "calculate_specificity",
    "create_production_rule",
    # Aggregators
    "create_all_content_selection_rules",
    "create_all_lexicalization_rules",
    "create_all_discourse_management_rules",
    "create_all_syntactic_realization_rules",
    "create_all_phase3_rules",
    "create_all_phase4_rules",
    "create_complete_production_system",
    # Content selection rules
    "create_select_is_a_fact_rule",
    "create_select_property_fact_rule",
    "create_select_capability_fact_rule",
    "create_select_location_fact_rule",
    "create_select_part_of_fact_rule",
    "create_prioritize_high_confidence_rule",
    "create_skip_low_confidence_rule",
    "create_select_synonym_rule",
    "create_select_definition_rule",
    "create_finish_content_selection_rule",
    "create_require_high_confidence_rule",
    "create_warn_medium_confidence_rule",
    "create_skip_uncertain_facts_rule",
    "create_aggregate_multi_source_rule",
    "create_prefer_direct_fact_rule",
    # Lexicalization rules
    "create_verbalize_is_a_simple_rule",
    "create_verbalize_is_a_variant_1_rule",
    "create_verbalize_is_a_variant_2_rule",
    "create_verbalize_has_property_rule",
    "create_verbalize_capable_of_rule",
    "create_verbalize_located_in_rule",
    "create_verbalize_part_of_rule",
    "create_vary_copula_verb_rule",
    "create_combine_facts_conjunction_rule",
    "create_avoid_repetition_rule",
    "create_select_formal_style_rule",
    "create_select_casual_style_rule",
    "create_add_elaboration_rule",
    "create_compress_similar_facts_rule",
    "create_finish_lexicalization_rule",
    # Discourse management rules
    "create_introduce_with_context_rule",
    "create_introduce_simple_rule",
    # Syntactic realization rules
    "create_add_article_nominative_rule",
    "create_add_article_accusative_rule",
    "create_capitalize_sentence_start_rule",
    "create_capitalize_nouns_rule",
    "create_order_sentence_elements_rule",
    "create_finish_sentence_rule",
]
