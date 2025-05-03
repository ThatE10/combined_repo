from steering_vec_functions.steering_datasets import format_question
import steering_opt # for optimizing steering vectors


def get_response(question, model, tokenizer, max_tokens=50):

    formatted_question = format_question(question, tokenizer)

    input_ids = tokenizer(formatted_question, return_tensors='pt').input_ids
    # .input_ids.to(device)
    generated_tokens = model.generate(input_ids, max_new_tokens=max_tokens)

    # Exclude the input tokens from the generated tokens
    generated_tokens_only = generated_tokens[:, input_ids.shape[-1]:]
    generated_str = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)[0]
    return generated_str

def get_steered_answer(vector, question, model, tokenizer, layer, max_tokens=50):
    formatted_question = format_question(question, tokenizer)

    steering_hook = (layer, steering_opt.make_steering_hook_hf(vector))

    with steering_opt.hf_hooks_contextmanager(model, [steering_hook]): 
        input_ids = tokenizer(formatted_question, return_tensors='pt').input_ids
        # .input_ids.to(device)
        generated_tokens = model.generate(input_ids, max_new_tokens=max_tokens)

    # Exclude the input tokens from the generated tokens
    generated_tokens_only = generated_tokens[:, input_ids.shape[-1]:]
    generated_str = tokenizer.batch_decode(generated_tokens_only, skip_special_tokens=True)[0]
    return generated_str
