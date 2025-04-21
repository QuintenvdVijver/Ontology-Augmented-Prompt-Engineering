!pip install openai requests scikit-learn datasets --quiet

from datasets import load_dataset, DatasetDict, Dataset
import requests
from sklearn.metrics import accuracy_score, f1_score
import json

# Your validation dataset
# validation_data = load_dataset('json', data_files="/content/Instruction_SemEval14_Laptops_Validation.json")
# validation_data = load_dataset('json', data_files="/content/Instruction_SemEval14_Restaurants_Validation.json")
validation_data = load_dataset('json', data_files="/content/Instruction_SemEval15_Restaurants_Validation.json")
# validation_data = load_dataset('json', data_files="/content/Instruction_SemEval16_Restaurants_Validation.json")
validation_data = list(validation_data['train'])  # Assuming your data is in the 'train' split

# OpenAI API configuration
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"
API_KEY = "YOUR_API_KEY"  # Replace with your OpenAI API key
# MODEL_ID = "gpt-3.5-turbo-0125"
MODEL_ID = "YOUR_MODEL_ID"

def get_model_prediction(instruction):
    """Get prediction from your fine-tuned model via API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_ID,  # Replace with your model ID
        "messages": [
            {"role": "user", "content": instruction}
        ],
        "max_tokens": 10,
        "temperature": 0.0
    }

    try:
        response = requests.post(API_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        prediction = result['choices'][0]['message']['content'].strip().lower()
        # Ensure prediction is one of the expected values
        valid_sentiments = ['positive', 'negative', 'neutral']
        return prediction if prediction in valid_sentiments else 'neutral'
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return 'neutral'  # Default fallback

def evaluate_model(validation_data):
    """Evaluate model performance"""
    true_labels = []
    predicted_labels = []

    # Get predictions for all validation    validation samples
    for sample in validation_data:
        instruction = sample['instruction']
        true_label = sample['response']

        # Get model prediction
        # predicted_label = get_model_prediction(instruction + "(answer with 'positive', 'negative' or 'neutral')")
        predicted_label = get_model_prediction(instruction)

        true_labels.append(true_label)
        predicted_labels.append(predicted_label)

        # Optional: Print comparison for debugging
        print(f"Instruction: {instruction}")
        print(f"True: {true_label}, Predicted: {predicted_label}\n")

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    macro_f1 = f1_score(true_labels, predicted_labels, average='macro')

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'macro_f1': macro_f1
    }

# Run evaluation
results = evaluate_model(validation_data)

# Print results
print("\nEvaluation Results:")
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1 Score (weighted): {results['f1_score']:.4f}")
print(f"Macro F1 Score: {results['macro_f1']:.4f}")

# Optional: Save results to file
with open('evaluation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
