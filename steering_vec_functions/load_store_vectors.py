import os
import torch

def save_steering_vector(steering_vector, model_name, layer_name, exp_name="", folder="./steering_vectors"):
    """
    Saves the steering vector to a file.

    Parameters:
        steering_vector (SteeringVector): The steering vector to save.
        model_name (str): The name of the model.
        layer_name (str): The name or index of the layer.
        folder (str): The folder where the steering vector will be saved. Defaults to './steering_vectors'.
    """
    os.makedirs(folder, exist_ok=True)

    if len(exp_name) > 0:
        exp_name = f"_{exp_name}"

    filename = f"{model_name.replace('/', '_')}_layer-{layer_name}{exp_name}.pt"
    filepath = os.path.join(folder, filename)
    torch.save(steering_vector, filepath)
    print(f"Steering vector saved to {filepath}")


def load_steering_vector(model_name, layer_name, exp_name="", folder="./steering_vectors"):
    """
    Loads a steering vector from a file.

    Parameters:
        model_name (str): The name of the model.
        layer_name (str): The name or index of the layer.
        folder (str): The folder where the steering vector is stored. Defaults to './steering_vectors'.

    Returns:
        SteeringVector: The loaded steering vector.
    """
    
    if len(exp_name) > 0:
        exp_name = f"_{exp_name}"
        
    filename = f"{model_name.replace('/', '_')}_layer-{layer_name}{exp_name}.pt"
    filepath = os.path.join(folder, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Steering vector file not found: {filepath}")
    steering_vector = torch.load(filepath, weights_only=False)
    print(f"Steering vector loaded from {filepath}")
    return steering_vector

