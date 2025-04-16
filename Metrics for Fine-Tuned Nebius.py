!pip install openai requests scikit-learn datasets --quiet

import os
from datasets import load_dataset, DatasetDict, Dataset
from openai import OpenAI
from sklearn.metrics import accuracy_score, f1_score
import json

# --- Nebius API Configuration ---
NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"

# Specify the Nebius model ID you want to use for predictions
# USER ACTION: Replace 'YOUR_MODEL' with the actual model ID provided by Nebius
NEBIUS_MODEL_ID = "YOUR_MODEL"

# API key for authenticating with the Nebius API
# USER ACTION: Replace 'YOUR_API_KEY' with your actual Nebius API key
NEBIUS_API_KEY = "YOUR_API_KEY"
if not NEBIUS_API_KEY:
    raise ValueError("NEBIUS_API_KEY environment variable not set. Please set it before running.")

# Initialize the Nebius client using the OpenAI library, which provides a compatible interface
try:
    client = OpenAI(
        base_url=NEBIUS_BASE_URL,
        api_key=NEBIUS_API_KEY
    )
except Exception as e:
    print(f"Error initializing Nebius client: {e}")
    exit()

# --- Load Your Validation Dataset ---
# This section loads a JSON file containing validation data for evaluating the model
# USER ACTION: Uncomment/Edit the appropriate dataset file path for your use case
# The dataset should contain 'instruction' and 'response' fields for each sample

# validation_data_file = "/content/Instruction_SemEval14_Laptops_Validation.json"
validation_data_file = "/content/Instruction_SemEval14_Restaurants_Validation.json"
# validation_data_file = "/content/Instruction_SemEval15_Restaurants_Validation.json"
# validation_data_file = "/content/Instruction_SemEval16_Restaurants_Validation.json"

# Load the dataset using the 'datasets' library
try:
    validation_data = load_dataset('json', data_files=validation_data_file)
    validation_data = list(validation_data['train']) # Assuming data is in 'train' split
except FileNotFoundError:
    print(f"Error: Validation data file not found at {validation_data_file}")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

def get_model_prediction(instruction):
    """Send an instruction to the Nebius model and retrieve its sentiment prediction.

    Args:
        instruction (str): The input text (instruction) to send to the model.

    Returns:
        str: Predicted sentiment ('positive', 'negative', or 'neutral').
    """
    try:
        # Make an API call to the Nebius model to get a prediction
        response = client.chat.completions.create(
            model=NEBIUS_MODEL_ID,
            messages=[
                {"role": "user", "content": instruction}
            ],
            max_tokens=10,  # Limit the response length since we expect short sentiment labels
            temperature=0.0 # Set to 0 for consistent, deterministic outputs
        )
        # Access the content correctly from the response object
        prediction = response.choices[0].message.content.strip().lower()

        # Ensure prediction is one of the expected values
        valid_sentiments = ['positive', 'negative', 'neutral']
        return prediction if prediction in valid_sentiments else 'neutral'

    except Exception as e:
        print(f"Error getting prediction from Nebius API: {e}")
        # Check if the response object has more details in case of API errors
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Response Error: {e.response.text}")
        return 'neutral' # Default fallback

def evaluate_model(validation_data):
    """Evaluate the Nebius model's performance on the validation dataset.

    Args:
        validation_data (list): List of samples, each with 'instruction' and 'response' fields.

    Returns:
        dict: Dictionary containing accuracy, weighted F1 score, and macro F1 score.
    """
    true_labels = [] # Store the true sentiment labels
    predicted_labels = [] # Store the model's predicted sentiment labels

    print(f"Starting evaluation using Nebius model: {NEBIUS_MODEL_ID}")
    print("-" * 30)

    # Get predictions for all validation samples
    for i, sample in enumerate(validation_data):
        instruction = sample.get('instruction')
        true_label = sample.get('response')

        # Skip samples with missing instruction or response
        if not instruction or not true_label:
            print(f"Skipping sample {i+1} due to missing 'instruction' or 'response'.")
            continue

        # Get model prediction
        # USER ACTION: Use the 'instruction + " Answer..."' for the standard model (not fine-tuned) and use only 'instruction for fine-tuned models
        predicted_label = get_model_prediction(instruction + " Answer only with 'positive', 'negative' or 'neutral'")
        # predicted_label = get_model_prediction(instruction)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        # Optional: Print comparison
        print(f"Sample {i+1}/{len(validation_data)}")
        # print(f"Instruction: {instruction}") # Uncomment if needed for debugging
        print(f"True: {true_label}, Predicted: {predicted_label}")
        # Add a small delay or check API rate limits if needed for large datasets
        # import time
        # time.sleep(0.1) # Example: sleep 100ms between calls

    print("-" * 30)
    print("Evaluation complete.")

    # Check if any valid samples were processed
    if not true_labels or not predicted_labels:
        print("No valid samples were processed. Cannot calculate metrics.")
        return {'accuracy': 0, 'f1_score': 0, 'macro_f1': 0}

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'macro_f1': macro_f1
    }

# --- Run Evaluation ---
results = evaluate_model(validation_data)

# --- Print and Save Results ---
print("\nEvaluation Results:")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score (weighted): {results['f1_score']:.4f}")
print(f"Macro F1 Score: {results['macro_f1']:.4f}")

# Optional: Save results to file
results_filename = 'nebius_evaluation_results.json'
try:
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_filename}")
except Exception as e:
    print(f"Error saving results to file: {e}")
