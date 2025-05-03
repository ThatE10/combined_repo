
import json
import torch
from tqdm import tqdm
import re
import pandas as pd
from typing import List, Dict, Any, Callable

from functools import partial

from steering_vec_functions.llm_judge import LLMJudge
from steering_vec_functions.steering_helper_function import get_response, get_steered_answer


def test_no_duplicates_in_suggestive_pairs(suggestive_pairs):
    """
    Test function to check that no tuple (correct_answer, incorrect_answer) 
    appears more than once in the list of suggestive answer pairs.
    
    Args:
        suggestive_pairs: List of tuples (correct_answer, incorrect_answer)
        
    Returns:
        bool: True if no duplicates exist, False otherwise
        list: List of duplicate entries found (empty if no duplicates)
    """
    seen_pairs = set()
    duplicates = []
    
    for instance in suggestive_pairs:
        for sent in instance[1:]:
            if sent in seen_pairs:
                duplicates.append(sent)
            else:
                seen_pairs.add(sent)
    
    has_no_duplicates = len(duplicates) == 0
    
    if has_no_duplicates:
        print("Test passed: No duplicate pairs found in the list.")
    else:
        print(f"Test failed: Found {len(duplicates)} duplicate pairs in the list.")
        print("Examples of duplicates:", duplicates[:5])  # Show up to 5 examples
    
    return has_no_duplicates, duplicates




def get_suggestive_answer_pairs(answer_data):
    """
    Filter the dataset for samples with the specific suggestive prompt template
    and return a list of (correct_answer, incorrect_answer) tuples.
    
    Args:
        answer_data: List of dictionaries containing the dataset
        
    Returns:
        List of tuples (correct_answer, incorrect_answer)
    """
    result = []
    used_questions = set()  # To keep track of used questions
    
    for item in answer_data:
        # Check if the item has the expected metadata structure
        if 'metadata' in item and 'prompt_template' in item['metadata']:
            # Check if the prompt template matches our target format
            template = item['metadata']['prompt_template']
            if "I don't think the answer is {correct_answer}" in template and "but I'm really not sure" in template:
                # If it matches, add the correct and incorrect answers to our result
                if 'base' in item and 'correct_answer' in item['base'] and 'incorrect_answer' in item['base']:
                    question = item['base']['question']
                    if question not in used_questions:
                        correct = item['base']['correct_answer']
                        incorrect = item['base']['incorrect_answer']


                        question_template = item['metadata']['prompt_template']

                        correct_ans_suggestive_question = question_template.format(question = question, correct_answer=correct)
                        incorrect_ans_suggestive_question = question_template.format(question = question, correct_answer=incorrect)


                        result.append((item, correct_ans_suggestive_question, incorrect_ans_suggestive_question))
                        used_questions.add(question)
                    else:
                        print("Question already found")
    
    # Then test for duplicates
    _, _ = test_no_duplicates_in_suggestive_pairs(result)

    return result





def evaluate_all_answers(
    answer_list: List,
    suggestive_pairs: List,
    judge_model: Callable[[str], str],
    correct_answers: List[str]  # List of correct answers for each question
) -> Dict:
    """
    Evaluate all answers in the answer_list using the judge model.
    
    Args:
        answer_list: Your existing list of answers
        suggestive_pairs: Your question pairs
        judge_model: The judge model function
        correct_answers: List of correct answers corresponding to each question
        
    Returns:
        Dictionary with evaluation results
    """
    # Initialize LLM Judge
    judge = LLMJudge(judge_model)
    
    # Initialize results dictionary
    results = {
        "misleading_q_answer": [],
        "misleading_q_steered_answer": [],
        "honest_q_answer": [],
        "honest_q_steered_answer": []
    }
    
    # Process each sample
    print("Evaluating answers...")
    for i in tqdm(range(len(answer_list))):
        # Get the correct answer for this question
        correct_answer = correct_answers[i]
        
        # Get questions from suggestive pairs
        _, misleading_q, honest_q = suggestive_pairs[i]
        
        # Create question data dictionary for the judge
        question_data = {
            'base': {
                'question': misleading_q if isinstance(misleading_q, str) else misleading_q['text'],
                'correct_answer': correct_answer
            }
        }
        
        # Get the answers
        (misleading_q_answer, misleading_q_steered_answer), (honest_q_answer, honest_q_steered_answer) = answer_list[i]
        
        # Evaluate each answer
        results["misleading_q_answer"].append(
            judge.evaluate(question_data, misleading_q_answer)
        )
        
        results["misleading_q_steered_answer"].append(
            judge.evaluate(question_data, misleading_q_steered_answer)
        )
        
        # Update question for honest question if different
        if honest_q != misleading_q:
            question_data['base']['question'] = honest_q if isinstance(honest_q, str) else honest_q['text']
        
        results["honest_q_answer"].append(
            judge.evaluate(question_data, honest_q_answer)
        )
        
        results["honest_q_steered_answer"].append(
            judge.evaluate(question_data, honest_q_steered_answer)
        )
    
    # Calculate and return summary statistics
    summary = calculate_summary(results)
    
    return {
        "detailed_results": results,
        "summary": summary
    }

def calculate_summary(results: Dict) -> Dict:
    """Calculate summary statistics from the evaluation results."""
    summary = {}
    
    for answer_type, evaluations in results.items():
        correct_count = sum(1 for eval_result in evaluations if eval_result.get("is_correct") is True)
        total = len(evaluations)
        accuracy = correct_count / total if total > 0 else 0
        
        summary[f"{answer_type}_accuracy"] = accuracy
        summary[f"{answer_type}_correct_count"] = correct_count
        summary[f"{answer_type}_total"] = total
    
    return summary


def generate_answers_and_correct_answers(suggestive_pairs, vector, model, tokenizer, layer, max_samples=5):
    """
    Generate answers and collect correct answers for the given suggestive pairs.

    Args:
        suggestive_pairs (list): List of suggestive question pairs.
        vector (torch.Tensor): Steering vector for generating steered answers.
        max_samples (int): Maximum number of samples to process.

    Returns:
        tuple: A tuple containing the answer list and the list of correct answers.
    """
    answer_list = []
    correct_answers = []

    for i in tqdm(range(max_samples)):
        misleading_suggestive_q = suggestive_pairs[i][1]
        honest_suggestive_q = suggestive_pairs[i][2]

        misleading_q_steered_answer = get_steered_answer(vector, misleading_suggestive_q, model, tokenizer, layer)
        misleading_q_answer = get_response(misleading_suggestive_q, model, tokenizer)

        honest_q_answer = get_response(honest_suggestive_q, model, tokenizer)
        honest_q_steered_answer = get_steered_answer(vector, honest_suggestive_q, model, tokenizer, layer)

        answer_list.append([
            (misleading_q_answer, misleading_q_steered_answer), 
            (honest_q_answer, honest_q_steered_answer)
        ])

        correct_answer = suggestive_pairs[i][0]['base']['correct_answer']
        correct_answers.append(correct_answer)

    return answer_list, correct_answers

