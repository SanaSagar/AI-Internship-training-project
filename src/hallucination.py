import re
from src.embeddings import calculate_semantic_similarity

def check_rules(llm_output, prompt):
    """
    Enhanced rule-based detection for hallucinations.
    Checks for contradictions, numeric inconsistencies, vague fillers, and absolute claims.
    Returns a list of triggered rule descriptions.
    """
    flags = []
    text_lower = llm_output.lower()
    
    # 1. Contradictory statements (basic heuristic)
    # Looking for opposite polarity phrases in close proximity or within the same text
    contradictions = [
        ("is true", "is false"),
        ("always", "never"),
        ("can be", "cannot be"),
        ("does exist", "does not exist")
    ]
    for p1, p2 in contradictions:
        if p1 in text_lower and p2 in text_lower:
            flags.append(f"Contradictory statements detected ('{p1}' and '{p2}').")
            break # Just flag once for contradictions

    # 2. Numeric inconsistencies
    # Extract numbers from prompt and output. If output has numbers that don't exist in prompt (heuristic).
    # We only apply this loosely since LLMs are meant to provide new info, but if prompt has specific years/numbers,
    # and the response contradicts them, we flag it. A simple proxy is: Prompt has numbers, Response has numbers, but they have no overlap.
    prompt_numbers = set(re.findall(r'\b\d+\b', prompt))
    output_numbers = set(re.findall(r'\b\d+\b', llm_output))
    if prompt_numbers and output_numbers:
        # If there's a total mismatch in digits (e.g., prompt says 2020, response says 2021)
        if not prompt_numbers.intersection(output_numbers):
            flags.append(f"Numeric inconsistency: Response contains numbers ({', '.join(list(output_numbers)[:3])}...) not found in prompt.")

    # 3. Vague filler phrases
    vague_fillers = ["i think", "i believe", "it might be", "probably", "i'm not sure but"]
    found_vague = [f"'{filler}'" for filler in vague_fillers if filler in text_lower]
    if found_vague:
        flags.append(f"Vague filler phrases indicating uncertainty: {', '.join(found_vague)}.")

    # 4. Overly confident absolute claims
    absolute_claims = ["always", "never", "100%", "guaranteed", "without a doubt"]
    found_absolute = [f"'{claim}'" for claim in absolute_claims if claim in text_lower]
    if found_absolute:
        flags.append(f"Overly confident absolute claims with no grounding: {', '.join(found_absolute)}.")

    return flags


def compute_hallucination_scores(llm_output, expected_output, prompt):
    """
    Upgrade embedding similarity check.
    Returns output_vs_expected_score and output_vs_prompt_score.
    """
    output_vs_expected_score = 1.0  # Safe default if no expected
    if expected_output:
        output_vs_expected_score = calculate_semantic_similarity(llm_output, expected_output)
        
    output_vs_prompt_score = 0.0
    if prompt:
        output_vs_prompt_score = calculate_semantic_similarity(llm_output, prompt)
        
    return output_vs_expected_score, output_vs_prompt_score


def classify_hallucination_risk(rule_flags, output_vs_expected_score, output_vs_prompt_score):
    """
    Final classifier function combining all signals.
    """
    num_flags = len(rule_flags)
    
    # Determine Risk Level
    if num_flags >= 2 or output_vs_expected_score < 0.35 or output_vs_prompt_score < 0.35:
        risk_level = "HIGH"
        summary = "High risk of hallucination based on multiple triggered rules or very low semantic scores."
    elif num_flags == 1 or (0.35 <= output_vs_expected_score <= 0.50) or (0.35 <= output_vs_prompt_score <= 0.50):
        risk_level = "MEDIUM"
        summary = "Medium risk of hallucination. Some rules were triggered or semantic scores were borderline."
    else:
        risk_level = "LOW"
        summary = "Low risk of hallucination. Semantic scores are solid and no rules were triggered."

    return {
        "risk_level": risk_level,
        "triggered_rules": rule_flags,
        "embedding_scores": {
            "vs_expected": output_vs_expected_score,
            "vs_prompt": output_vs_prompt_score
        },
        "summary": summary
    }
