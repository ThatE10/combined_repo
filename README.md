# Steering Vector Framework

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
