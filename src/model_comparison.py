import time
from src.llm import generate_response
from src.evaluator import evaluate_response
from src.utils import get_logger

logger = get_logger(__name__)

def run_model(prompt, model_name, temperature=0.7):
    """
    Calls Ollama API via existing generate_response and returns output with latency.
    """
    start_time = time.time()
    try:
        output = generate_response(prompt, model=model_name, temperature=temperature)
        latency_ms = (time.time() - start_time) * 1000.0
        return {
            "model": model_name,
            "output": output,
            "latency_ms": latency_ms,
            "error": False
        }
    except Exception as e:
        logger.error(f"Error running model {model_name}: {e}")
        return {
            "model": model_name,
            "output": f"Error: {e}",
            "latency_ms": 0.0,
            "error": True
        }


def compare_models(prompt, expected_output, models=["llama3", "phi3:mini"], use_judge=False, temperature=0.7):
    """
    Iterates through models, computes scores via evaluate_response, and formats the output.
    """
    results = []
    for model_name in models:
        model_res = run_model(prompt, model_name, temperature=temperature)
        
        if model_res["error"]:
            results.append({
                "model": model_name,
                "output": model_res["output"],
                "composite_score": 0.0,
                "latency_ms": 0.0,
                "semantic_score": 0.0,
                "judge_score": 0.0,
                "length_penalty": 0.0,
                "error": True
            })
            continue

        # Use the existing evaluator pipeline
        eval_res = evaluate_response(prompt, model_res["output"], expected_output=expected_output, model=model_name, use_judge=use_judge)
        
        # Calculate length penalty explicitly for display, based on the evaluator rule: 
        # (It returns total score components weighted, but to isolate we can infer or leave generic.
        # Since eval_res doesn't strictly export "length_penalty", we use word count to estimate if length penalty hit)
        
        results.append({
            "model": model_name,
            "output": model_res["output"],
            "composite_score": eval_res.get("overall_score", 0.0),
            "latency_ms": model_res["latency_ms"],
            "semantic_score": (eval_res.get("semantic_similarity") or 0.0) * 100.0,
            "judge_score": (eval_res.get("judge_score") or 0.0) * 10.0,
            "length_penalty": eval_res.get("word_count", 0),  # User requested length penalty generic display, using word_count as proxy
            "error": False
        })
        
    return results


def get_winner(results):
    """
    Returns the absolute best performing model across composite score, tied intelligently to latency.
    """
    valid_results = [r for r in results if not r.get("error")]
    if not valid_results:
        return None
        
    # Sort by composite_score DESC, then latency_ms ASC
    valid_results.sort(key=lambda x: (-x['composite_score'], x['latency_ms']))
    
    return valid_results[0]
