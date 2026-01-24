
import spacy
from collections import namedtuple
from src.utils.semantic import compute_rule_priority, RulePriority

# Mock objects to simulate rule data
def test_priority_calculation():
    print("\n--- Test 1: Priority Calculation Independence ---")
    
    # A generic lexical rule
    p_lex = compute_rule_priority("LEMMA", 0.9, 10, rule_key=("LEMMA", "foo"))
    # A semantic combined rule
    p_sem = compute_rule_priority("PREP_STRUCT_LEXNAME", 0.7, 10, rule_key=("PSL", "bar"))
    
    print(f"Lexical (Precision 0.9) Priority: {p_lex}")
    print(f"Combined (Precision 0.7) Priority: {p_sem}")
    
    # Semantic should be higher because TIER 1 (100) > TIER 5 (20)
    # even with lower precision.
    if p_sem > p_lex:
        print("SUCCESS: Combined rule has higher priority than high-precision lexical rule.")
    else:
        print("FAILURE: Priority calculation is wrong.")

def test_sorting_modes():
    print("\n--- Test 2: Sorting Logic (Simulator) ---")
    
    # Create dummy rules objects
    Rule = namedtuple("Rule", ["name", "precision", "support", "priority"])
    
    # Rule A: High Precision, Low Tier
    rule_a = Rule("LexicalRule", 0.9, 10, 20_000_000) # Tier 5 (20) roughly
    # Rule B: Low Precision, High Tier
    rule_b = Rule("SemanticRule", 0.7, 10, 100_000_000) # Tier 1 (100) roughly
    
    rules = [rule_a, rule_b]
    
    # First Match Sorting: (-Precision, -Support)
    fm_sorted = sorted(rules, key=lambda r: (-r.precision, -r.support))
    print(f"First Match Order: {[r.name for r in fm_sorted]}")
    
    if fm_sorted[0].name == "LexicalRule":
        print("SUCCESS: First Match picked High Precision rule first.")
    else:
        print("FAILURE: First Match sorting incorrect.")

    # Priority Based Sorting: (-Priority)
    pb_sorted = sorted(rules, key=lambda r: -r.priority)
    print(f"Priority Based Order: {[r.name for r in pb_sorted]}")
    
    if pb_sorted[0].name == "SemanticRule":
        print("SUCCESS: Priority Based picked High Tier rule first.")
    else:
        print("FAILURE: Priority Based sorting incorrect.")

def test_tier_sensitivity():
    print("\n--- Test 3: Tier Sensitivity ---")
    
    # Original: PREP_STRUCT (Tier 2=80) vs BIGRAM (Tier 4=40)
    p_prep = compute_rule_priority("PREP_STRUCT", 0.6, 5)
    p_bigram = compute_rule_priority("BIGRAM", 0.6, 5)
    
    print(f"Original: Prep={p_prep} vs Bigram={p_bigram}")
    assert p_prep > p_bigram
    
    # Monkey Patch: Swap tiers
    # We simulate this by momentarily changing the constants if possible, 
    # but since they are class attributes we can just patch them.
    original_tier2 = RulePriority.TIER_2_PREP_STRUCTURE
    original_tier4 = RulePriority.TIER_4_BIGRAM_DEP
    
    try:
        RulePriority.TIER_2_PREP_STRUCTURE = 10
        RulePriority.TIER_4_BIGRAM_DEP = 90
        
        p_prep_new = compute_rule_priority("PREP_STRUCT", 0.6, 5)
        p_bigram_new = compute_rule_priority("BIGRAM", 0.6, 5)
        
        print(f"Modified: Prep={p_prep_new} vs Bigram={p_bigram_new}")
        
        if p_bigram_new > p_prep_new:
            print("SUCCESS: Priority flipped after changing tiers.")
        else:
            print("FAILURE: Priority did not respond to tier change.")
            
    finally:
        # Restore
        RulePriority.TIER_2_PREP_STRUCTURE = original_tier2
        RulePriority.TIER_4_BIGRAM_DEP = original_tier4

if __name__ == "__main__":
    test_priority_calculation()
    test_sorting_modes()
    test_tier_sensitivity()
