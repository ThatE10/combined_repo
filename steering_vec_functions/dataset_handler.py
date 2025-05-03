import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from steering_vec_functions.steering_datasets import  get_sycophancy_dataset

class DatasetHandler:
    """Class to handle dataset operations for steering experiments."""
    
    def __init__(self, data_path="./data/"):
        self.data_path = data_path
        self.answer_data = None
        self.suggestive_pairs = None
    
    def load_sycophancy_dataset(self, data_type="answer"):
        """Load the sycophancy dataset."""
        try:
            self.answer_data =  get_sycophancy_dataset(data_path=self.data_path, data_type=data_type)
            return self.answer_data
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def get_suggestive_answer_pairs(self):
        """Get pairs of suggestive questions from the dataset."""
        if not self.answer_data:
            print("No dataset loaded. Call load_sycophancy_dataset() first.")
            return []
        
        result = []
        used_questions = set()
        
        for item in self.answer_data:
            # Check for expected metadata structure
            if 'metadata' in item and 'prompt_template' in item['metadata']:
                template = item['metadata']['prompt_template']
                
                # Check for target template format
                if "I don't think the answer is {correct_answer}" in template and "but I'm really not sure" in template:
                    if 'base' in item and 'correct_answer' in item['base'] and 'incorrect_answer' in item['base']:
                        question = item['base']['question']
                        
                        if question not in used_questions:
                            correct = item['base']['correct_answer']
                            incorrect = item['base']['incorrect_answer']
                            
                            question_template = item['metadata']['prompt_template']
                            
                            # Format questions with correct and incorrect answers
                            correct_ans_suggestive_q = question_template.format(question=question, correct_answer=correct)
                            incorrect_ans_suggestive_q = question_template.format(question=question, correct_answer=incorrect)
                            
                            # Store the original item, and both question variants
                            result.append((item, correct_ans_suggestive_q, incorrect_ans_suggestive_q))
                            used_questions.add(question)
        
        self.suggestive_pairs = result
        return result
    
    def test_no_duplicates(self):
        """Test for duplicate questions in the suggestive pairs."""
        if not self.suggestive_pairs:
            print("No suggestive pairs loaded. Call get_suggestive_answer_pairs() first.")
            return True, []
        
        seen_pairs = set()
        duplicates = []
        
        for instance in self.suggestive_pairs:
            for sent in instance[1:]:
                if sent in seen_pairs:
                    duplicates.append(sent)
                else:
                    seen_pairs.add(sent)
        
        has_no_duplicates = len(duplicates) == 0
        
        if has_no_duplicates:
            print("No duplicate pairs found in the dataset.")
        else:
            print(f"Found {len(duplicates)} duplicate pairs in the dataset.")
            print("Examples:", duplicates[:3])
        
        return has_no_duplicates, duplicates
    
    def generate_answers(self, steering_vector, max_samples=5):
        """Generate both normal and steered answers for the suggestive pairs."""
        if not self.suggestive_pairs:
            print("No suggestive pairs loaded. Call get_suggestive_answer_pairs() first.")
            return [], []
            
        answer_list = []
        correct_answers = []
        
        for i in tqdm(range(min(max_samples, len(self.suggestive_pairs)))):
            pair = self.suggestive_pairs[i]
            misleading_q = pair[1]
            honest_q = pair[2]
            
            # Get normal and steered answers for both question types
            misleading_q_normal = steering_vector.get_response(misleading_q)
            misleading_q_steered = steering_vector.get_steered_response(misleading_q)
            
            honest_q_normal = steering_vector.get_response(honest_q)
            honest_q_steered = steering_vector.get_steered_response(honest_q)
            
            # Store the answers
            answer_list.append([
                (misleading_q_normal, misleading_q_steered),
                (honest_q_normal, honest_q_steered)
            ])
            
            # Store the correct answer
            correct_answer = pair[0]['base']['correct_answer']
            correct_answers.append(correct_answer)
            
        return answer_list, correct_answers
