"""
Script to compare positivity of comment pairs using OpenAI's token probabilities.
"""
import argparse
import json
import os
from pathlib import Path
import time
from typing import List, Tuple, Dict
from openai import OpenAI
from tqdm import tqdm
import numpy as np

# File paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
INPUT_FILE = DATA_DIR / "comment_pairs.jsonl"  # Your input file with comment pairs
OUTPUT_FILE = DATA_DIR / "positivity_scores.jsonl"

# Prompt template
PROMPT_TEMPLATE = """###
Human: The following are comments on {topic}. Which comment is more positive? 
Comment (A): "{first_comment}" 
Comment (B): "{second_comment}" 
Assistant: The comment which is more positive is Comment ("""

def load_comment_pairs(path: Path) -> List[dict]:
    """Load comment pairs from a JSONL file."""
    if not path.exists():
        print(f"Warning: {path} does not exist. Using sample data instead.")
        # Sample data if file doesn't exist
        return [
            {
                "pair_id": "sample1",
                "topic": "a solution to a math problem",
                "first_comment": "This approach works but it's inefficient.",
                "second_comment": "Great solution! Very elegant and clear."
            },
            {
                "pair_id": "sample2",
                "topic": "an argument",
                "first_comment": "You make some valid points but your conclusion is flawed.",
                "second_comment": "I see your perspective and appreciate how you structured your reasoning."
            }
        ]
    
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def get_token_probabilities(
    client: OpenAI, 
    model_slug: str, 
    comment_pair: dict
) -> Dict:
    """Get token probabilities for A vs B prediction."""
    # Format the prompt
    prompt = PROMPT_TEMPLATE.format(
        topic=comment_pair.get("topic", "a solution"),
        first_comment=comment_pair["first_comment"],
        second_comment=comment_pair["second_comment"]
    )
    
    # Request logprobs for the next token
    response = client.chat.completions.create(
        model=model_slug,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1,
        logprobs=True,
        top_logprobs=5  # Get probabilities for top 5 tokens
    )
    
    # Extract logprobs
    token_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    
    # Convert to dictionaries with token:probability
    probs = {}
    for item in token_logprobs:
        token = item.token
        # Convert logprob to actual probability
        prob = np.exp(item.logprob)
        probs[token] = prob
        # probs[token] = float(prob)
    
    # Look specifically for "A" and "B" tokens
    a_prob = probs.get("A", 0)
    b_prob = probs.get("B", 0)
    
    # If "A" or "B" aren't in top tokens, check similar tokens
    if "A" not in probs:
        for token, prob in probs.items():
            if "A" in token:
                a_prob += prob
    
    if "B" not in probs:
        for token, prob in probs.items():
            if "B" in token:
                b_prob += prob
    
    return {
        "A_probability": a_prob,
        "B_probability": b_prob,
        "all_top_probs": {t.token: np.exp(t.logprob) for t in token_logprobs},
        "predicted_more_positive": "A" if a_prob > b_prob else "B"
    }

def main(model_slug: str, delay_s: float = 0.5):
    """Main function to process all comment pairs."""
    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable missing")
    
    client = OpenAI(api_key=api_key)
    
    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load comment pairs
    comment_pairs = load_comment_pairs(INPUT_FILE)
    print(f"Loaded {len(comment_pairs)} comment pairs")
    
    # Process each pair
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        for pair in tqdm(comment_pairs, desc="Comparing positivity"):
            try:
                result = get_token_probabilities(client, model_slug, pair)
                
                # Write results
                out_f.write(json.dumps({
                    "pair_id": pair.get("pair_id", "unknown"),
                    "topic": pair.get("topic", ""),
                    "first_comment": pair["first_comment"],
                    "second_comment": pair["second_comment"],
                    "results": result,
                    "meta": {"model": model_slug}
                }) + "\n")
                
            except Exception as e:
                print(f"Error processing pair {pair.get('pair_id', 'unknown')}: {e}")
            
            # Add delay between API calls
            if delay_s:
                time.sleep(delay_s)
    
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare positivity of comment pairs using OpenAI API")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between API calls")
    parser.add_argument("--input", type=str, help="Input JSONL file path (optional)")
    parser.add_argument("--output", type=str, help="Output JSONL file path (optional)")
    
    args = parser.parse_args()
    
    # Override file paths if provided
    if args.input:
        INPUT_FILE = Path(args.input)
    if args.output:
        OUTPUT_FILE = Path(args.output)
    
    main(args.model, args.delay)