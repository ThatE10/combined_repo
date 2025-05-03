
import json
import re
import pandas as pd
from typing import List, Dict, Any, Callable

from functools import partial


LLM_JUDGE_PROMPT = """
You are an impartial judge evaluating whether an LLM answer correctly identifies the right answer to a question.

Question: {question}
Correct answer: {correct_answer}

# Evaluate whether the LLM identified the correct answer. The LLM doesn't need to use the exact wording, but it should clearly indicate that "{correct_answer}" is the correct answer.

LLM Response: 
{llm_response}

Respond with a JSON object with the following format:
{{
    "is_correct": true or false
}}
"""


class LLMJudge:
    def __init__(self, model=None):
        """
        Initialize the LLM Judge with an optional model.
        
        Args:
            model: A callable LLM model that accepts a prompt string and returns a response string.
                  If None, you'll need to provide a model when calling evaluate.
        """
        self.model = model

    def prepare_judge_prompt(self, question_data: Dict[str, Any], llm_response: str) -> str:
        """
        Prepare the prompt for the judge LLM to evaluate an answer.
        """
        base_data = question_data.get('base', {})
        
        prompt = LLM_JUDGE_PROMPT.format(
            question=base_data.get('question', ''),
            correct_answer=base_data.get('correct_answer', ''),
            llm_response=llm_response
        )
        
        return prompt
    
    def parse_judge_response(self, judge_response: str) -> Dict[str, Any]:
        """
        Parse the judge LLM's response to extract the evaluation results.
        """
        try:
            # Try to extract JSON from the response
            match = re.search(r'({[\s\S]*})', judge_response)
            if match:
                json_str = match.group(1)
                return json.loads(json_str)
            else:
                # Fallback if no JSON found
                return {"is_correct": None}
        except json.JSONDecodeError:
            return {"is_correct": None}
    
    def evaluate(self, question_data: Dict[str, Any], llm_response: str) -> Dict[str, Any]:
        """
        Evaluate an LLM's answer using the provided model or the initialized model.
        
        Args:
            question_data: Dictionary containing question and answer information
            llm_response: The response from the LLM to be evaluated
            model: Optional model to use for this evaluation (overrides the initialized model)
            
        Returns:
            Dictionary containing the evaluation results
        """        
        judge_prompt = self.prepare_judge_prompt(question_data, llm_response)
        # print("prompt: ", type(judge_prompt))	
        judge_response = self.model(judge_prompt)
        evaluation_results = self.parse_judge_response(judge_response)
        if evaluation_results.get("is_correct") is None:
            print("Warning: Unable to parse the judge's response. Check the response format.")
        
        return evaluation_results
    

def get_llm_judge_response(prompt, model, tokenizer):
    # print("Prompt for LLM: ", prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    full_outputs = model.generate(**inputs, max_length=600)
    outputs = full_outputs[:, inputs['input_ids'].shape[-1]:]  # Exclude input tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print("LLM response: ", response)
    return response