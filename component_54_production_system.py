"""
Component 54: Production System (Compatibility Wrapper)

This module provides backwards compatibility by re-exporting all classes and functions
from the refactored production system modules.

For new code, prefer importing from the specific modules:
- component_54_production_types: Enums and basic types
- component_54_production_state: ResponseGenerationState
- component_54_production_rule: ProductionRule
- component_54_production_engine: ProductionSystemEngine
- component_54_production_rules: Rule factory functions

Author: KAI Development Team
Date: 2025-11-14
"""

from component_54_production_engine import ProductionSystemEngine
from component_54_production_rule import ProductionRule
from component_54_production_rules import (
    calculate_specificity,
    create_add_article_accusative_rule,
    create_add_article_dative_rule,
    create_add_article_nominative_rule,
    create_add_comma_conjunction_rule,
    create_add_confidence_qualifier_rule,
    create_add_elaboration_rule,
    create_add_period_rule,
    create_add_transition_rule,
    create_aggregate_multi_source_rule,
    create_all_content_selection_rules,
    create_all_discourse_management_rules,
    create_all_lexicalization_rules,
    create_all_phase3_rules,
    create_all_phase4_rules,
    create_all_syntactic_realization_rules,
    create_avoid_repetition_rule,
    create_capitalize_nouns_rule,
    create_capitalize_sentence_start_rule,
    create_combine_facts_conjunction_rule,
    create_complete_production_system,
    create_compress_similar_facts_rule,
    create_conclude_answer_rule,
    create_ensure_gender_agreement_rule,
    create_explain_reasoning_path_rule,
    create_finish_content_selection_rule,
    create_finish_lexicalization_rule,
    create_finish_sentence_rule,
    create_fix_verb_agreement_rule,
    create_insert_preposition_rule,
    create_introduce_simple_rule,
    create_introduce_with_context_rule,
    create_mark_hypothesis_rule,
    create_mention_evidence_source_rule,
    create_offer_elaboration_rule,
    create_order_sentence_elements_rule,
    create_prefer_direct_fact_rule,
    create_prioritize_high_confidence_rule,
    create_production_rule,
    create_require_high_confidence_rule,
    create_select_capability_fact_rule,
    create_select_casual_style_rule,
    create_select_definition_rule,
    create_select_formal_style_rule,
    create_select_is_a_fact_rule,
    create_select_location_fact_rule,
    create_select_part_of_fact_rule,
    create_select_property_fact_rule,
    create_select_synonym_rule,
    create_signal_high_confidence_rule,
    create_signal_uncertainty_rule,
    create_skip_low_confidence_rule,
    create_skip_uncertain_facts_rule,
    create_structure_multi_part_answer_rule,
    create_vary_copula_verb_rule,
    create_verbalize_capable_of_rule,
    create_verbalize_has_property_rule,
    create_verbalize_is_a_simple_rule,
    create_verbalize_is_a_variant_1_rule,
    create_verbalize_is_a_variant_2_rule,
    create_verbalize_located_in_rule,
    create_verbalize_part_of_rule,
    create_warn_medium_confidence_rule,
)
from component_54_production_state import ResponseGenerationState

# Re-export all public classes and functions for backwards compatibility
from component_54_production_types import (
    DiscourseState,
    GenerationGoal,
    GenerationGoalType,
    PartialTextStructure,
    RuleCategory,
)

__all__ = [
    # Types
    "RuleCategory",
    "GenerationGoalType",
    "GenerationGoal",
    "DiscourseState",
    "PartialTextStructure",
    # State
    "ResponseGenerationState",
    # Rule
    "ProductionRule",
    # Engine
    "ProductionSystemEngine",
    # Rule factories (helper)
    "calculate_specificity",
    "create_production_rule",
    # Rule factories (content selection)
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
    "create_all_content_selection_rules",
    # Rule factories (lexicalization)
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
    "create_all_lexicalization_rules",
    # Rule factories (discourse)
    "create_introduce_with_context_rule",
    "create_introduce_simple_rule",
    "create_signal_uncertainty_rule",
    "create_signal_high_confidence_rule",
    "create_explain_reasoning_path_rule",
    "create_mark_hypothesis_rule",
    "create_add_confidence_qualifier_rule",
    "create_mention_evidence_source_rule",
    "create_structure_multi_part_answer_rule",
    "create_add_transition_rule",
    "create_conclude_answer_rule",
    "create_offer_elaboration_rule",
    "create_all_discourse_management_rules",
    "create_all_phase3_rules",
    # Rule factories (syntax)
    "create_add_article_nominative_rule",
    "create_add_article_accusative_rule",
    "create_add_article_dative_rule",
    "create_capitalize_sentence_start_rule",
    "create_capitalize_nouns_rule",
    "create_add_period_rule",
    "create_add_comma_conjunction_rule",
    "create_fix_verb_agreement_rule",
    "create_ensure_gender_agreement_rule",
    "create_insert_preposition_rule",
    "create_order_sentence_elements_rule",
    "create_finish_sentence_rule",
    "create_all_syntactic_realization_rules",
    "create_all_phase4_rules",
    # Complete system
    "create_complete_production_system",
]
