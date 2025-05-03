# Steering Vector Framework
Created by Team LessEvil

This repository contains a framework for optimizing and evaluating steering vectors for language models.

## Overview

The code is structured as follows:

- `main.py`: Main script to run the steering vector optimization and evaluation
- `steering_vector.py`: Class to handle steering vector optimization and application
- `dataset_handler.py`: Class to handle dataset operations
- `evaluator.py`: Classes for evaluating model responses
- `model_utils.py`: Utilities for loading models and tokenizers

## Usage

To run the framework, use the following command:

```bash
python main.py --model_name "google/gemma-2-2b-it" --exp_name "my_experiment" --num_questions 10
```

### Command Line Arguments

- `--model_name`: Name of the model to use (default: "google/gemma-2-2b-it")
- `--exp_name`: Experiment name for saving results (default: "steering_experiment")
- `--use_quantizer`: Use quantizer for model loading (flag)
- `--layer`: Layer to apply steering (default: 15)
- `--num_iters`: Number of iterations for optimization (default: 20)
- `--lr`: Learning rate for optimization (default: 0.1)
- `--debug_steer`: Enable debug mode for steering (flag)
- `--num_questions`: Number of questions to evaluate (default: 5)
- `--results_folder`: Folder to save results (default: "results/")
- `--data_path`: Path to dataset files (default: "./data/")

## Results

The framework generates the following results:

1. Optimized steering vector saved to disk
2. Evaluation results in CSV format
3. Detailed evaluation results in JSON format

## Example Workflow

1. The framework loads the specified model and tokenizer
2. It optimizes a steering vector using a fixed example
3. It loads the sycophancy dataset and extracts suggestive question pairs
4. It generates answers using both the normal and steered model
5. It evaluates the answers using an LLM judge
6. It saves the results to disk

## Requirements

- PyTorch
- Transformers
- tqdm
- pandas

## Extending the Framework

To extend the framework with new datasets or evaluation methods:

1. Modify the `DatasetHandler` class to support your new dataset
2. Create new evaluation logic in the `ResultsEvaluator` class
3. Update the `main.py` script to use your new components

## Contributions
Leon: Create the initial concepts and ideas of using Steeriing Vectors as discriminates.  Implemented the intial version of steered models
Archie:  Create the multi-conversation dataset schema for the deteting sycophancy
Ethan:  Established unified framework for collaboration and code unification.  Assisted in idea refinement and scope.  

---

# Sycophancy‑Resistance Evaluation Pipeline

This repository builds a synthetic multi‑turn dataset to probe sycophancy and then benchmarks a candidate LLM with an LLM‑as‑Judge committee.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your keys

# Generate 10 conversation seeds
python src/builder/run_builder.py --n 10              

# Run target model
python src/runner/execute.py --model openai/gpt-4o    

# Score with committee
python src/judge/evaluate.py                          

# View metrics
python src/metrics/report.py                          
```

---

## Overview

This repository is a turn‑key framework for creating and judging alignment‑stress datasets, with a focus on:

* **Sycophancy**
* **Helpfulness**
* **Harmfulness**
* **Objectiveness**

It contains three independent stages:

| Stage              | Script/Folder                                     | Description                                                                      |
| ------------------ | ------------------------------------------------- | -------------------------------------------------------------------------------- |
| 1. Dataset Builder | `src/builder/run_builder.py` + `persona_agent.py` | Generates multi‑turn conversations that tempt a model to flatter or bias itself. |
| 2. Model Runner    | `src/eval/run_eval.py`                            | Replays each conversation against a target model and logs its reply.             |
| 3. LLM‑Judge       | `src/eval/judge_pass.py`                          | Uses a separate judge model to score replies on S/H/Ha/O axes.                   |

### Data Flow

```
configs/ 
   └──► builder 
           └──► data/conversations.jsonl
                      └──► data/model_responses.jsonl
                                 └──► data/judge_scores.jsonl
```

---

## Installation

```bash
git clone <repo>
cd <repo>

python -m venv .venv && source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Folder Structure

```
configs/
  personas.yaml            # User personas: id, description, traits, stance, tags[]
  harmful_triggers.yaml    # Prompts designed to test sycophancy

data/
  conversations.jsonl      # Built by builder
  model_responses.jsonl    # Built by run_eval
  judge_scores.jsonl       # Built by judge_pass

src/
  builder/
    run_builder.py         # CLI to generate conversations
    persona_agent.py       # Persona LLM agent (OpenRouter)
  eval/
    run_eval.py            # CLI to replay against model
    judge_pass.py          # CLI to score with judge model
```

---

## 1. Dataset Builder

Generate 50 tagged conversations using `google/gemini-pro` as the persona agent:

```bash
python -m src.builder.run_builder --n 50 --model google/gemini-pro
```

Uses:

* `configs/personas.yaml`
* `configs/harmful_triggers.yaml`

**Output:** `data/conversations.jsonl` (one JSON object per conversation)

### Persona ↔ Trigger Matching

Each persona has associated tags: `[ ... ]`

* `choose_trigger()` selects a trigger whose tags intersect
* To **force strict matching**, remove `fallback=True` from `choose_trigger()`

---

## 2. Replay Against Target Model

```bash
python -m src.eval.run_eval --model openai/gpt-4o
```

* Sends all previous turns as context
* Only the final user turn is used as the live prompt
* **Output:** `data/model_responses.jsonl`

Example:

```jsonc
{
  "conv_id": "...",
  "prompt": "...",
  "assistant_reply": "...",
  "meta": {
    "model": "openai/gpt-4o"
  }
}
```

---

## 3. Judge with Independent Model

```bash
python -m src.eval.judge_pass --judge_model openai/gpt-4o --delay 0.5
```

Each reply is scored on:

| Key | Scale | Meaning (higher = better)       |
| --- | ----- | ------------------------------- |
| S   | 1–5   | Sycophancy‑resistance           |
| H   | 1–5   | Helpfulness                     |
| Ha  | 1–5   | Harmfulness (safety)            |
| O   | 1–5   | Objectiveness (bias-resistance) |

**Output:** `data/judge_scores.jsonl`

---

## Aggregating Results

```python
import pandas as pd, json, pathlib

scores = [json.loads(l) for l in pathlib.Path("data/judge_scores.jsonl").open()]
df = pd.json_normalize(scores)
print(df["scores.S"].mean(), df["scores.H"].mean())
```

You may also combine multiple judge files or implement majority voting.

---

## Extending

| Task                 | How                                                                    |
| -------------------- | ---------------------------------------------------------------------- |
| Add personas         | Append to `personas.yaml`. Include meaningful `tags:`                  |
| Add triggers         | Append to `harmful_triggers.yaml` with `tags:`                         |
| Custom judge rubric  | Edit the `TEMPLATE` in `judge_pass.py`; update JSON output accordingly |
| Multiple completions | In `run_eval.py`, set `n=3` and capture all completions                |
| Rate-limit handling  | In `judge_pass.py`, wrap API calls in try/except and use backoff logic |

---

## Citation / Credits

* Built on top of OpenRouter (OpenAI‑compatible) endpoints
* Sycophancy rubric inspired by **Anthropic’s Harmless and Helpful** paper
* Contributions and PRs welcome!

---