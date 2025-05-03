# ── src/builder/run_builder.py ─────────────────────────────────────────
import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from typing import List

import yaml

from Eval_Pipeline.src.builder.persona_agent import Persona, PersonaAgent

# ─────────────────────────── paths ────────────────────────────────────
ROOT        = Path(__file__).resolve().parents[2]
CONFIG_DIR  = ROOT / "configs"
DATA_DIR    = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTFILE     = DATA_DIR / "conversations.jsonl"

# ────────────────────────── helpers ───────────────────────────────────
def load_specs():
    """Load personas & triggers from YAML."""
    with open(CONFIG_DIR / "personas.yaml", encoding="utf-8") as f:
        personas = [Persona(**p) for p in yaml.safe_load(f)]
    with open(CONFIG_DIR / "harmful_triggers.yaml", encoding="utf-8") as f:
        triggers: List[dict] = yaml.safe_load(f)
    return personas, triggers


def choose_trigger(persona: Persona, triggers: List[dict], *, fallback=True) -> dict:
    """
    Select a trigger whose tags intersect persona.tags (empty list if missing).
    Falls back to any trigger if no match and `fallback` is True.
    """
    p_tags = set(getattr(persona, "tags", []))  # ← handles missing tags

    matched = [t for t in triggers if p_tags & set(t.get("tags", []))]
    pool    = matched if matched else (triggers if fallback else [])

    if not pool:
        raise ValueError(f"No triggers match tags {p_tags} and fallback disabled.")
    return random.choice(pool)

# ────────────────────────── main loop ─────────────────────────────────
def main(num: int, model_slug: str):
    personas, triggers = load_specs()

    # quick sanity check
    print("\n[DEBUG] Loaded", len(personas), "personas from", CONFIG_DIR)
    for p in personas[:3]:
        print("  →", p.id, "tags:", getattr(p, "tags", []))

    conversations = []

    for i in range(num):
        persona = personas[i % len(personas)]
        trig    = choose_trigger(persona, triggers)
        prompt  = trig["user_prompt"]

        print(f"\n##### ({i+1}/{num}) persona={persona.id} trigger={trig['id']} #####")
        print("Persona tags:", getattr(persona, "tags", []),
              "| Trigger tags:", trig.get("tags", []))

        try:
            agent = PersonaAgent(persona, prompt, model=model_slug)
            conv  = agent.generate()
            conversations.append(conv)
        except Exception:
            print("!! GENERATION FAILED – traceback below")
            traceback.print_exc(file=sys.stdout)

    if not conversations:
        print("\n[WARNING] No conversations generated — nothing written.")
        return

    # save
    with open(OUTFILE, "a", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n=== ADDED {len(conversations)} conversations → {OUTFILE.resolve()} ===")

# ───────────────────────── CLI entry ──────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate sycophancy dataset")
    ap.add_argument("--n",     type=int,  default=5, help="number of conversations")
    ap.add_argument("--model", default="google/gemini-pro", help="OpenRouter model slug")
    args = ap.parse_args()
    main(args.n, args.model)
