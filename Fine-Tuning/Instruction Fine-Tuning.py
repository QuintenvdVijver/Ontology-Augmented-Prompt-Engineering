!pip install datasets transformers peft evaluate accelerate scikit-learn --quiet

from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    TrainerCallback)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import gc
import os



# --- Define the MODIFIED Callback ---
class MemoryCleanupCallback(TrainerCallback):
    def on_evaluate_begin(self, args, state, control, model=None, **kwargs):
        """Called at the beginning of the evaluation phase."""
        print("\n--- Evaluation Start ---")
        # Explicitly disable cache before evaluation starts to save memory
        if model is not None and hasattr(model.config, "use_cache"):
             # Check if cache was already disabled, maybe unnecessary check but safe
            if model.config.use_cache:
                model.config.use_cache = False
                print("Explicitly disabled model.config.use_cache for evaluation.")
            else:
                 print("model.config.use_cache was already False.")
        else:
             print("Model or model.config not available, or does not have use_cache attribute.")

        # Clear CUDA cache and run Python garbage collector
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleared GPU cache and ran garbage collector before evaluation start.")
        # Optionally print memory stats
        if torch.cuda.is_available():
             print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
             print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    def on_evaluate_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of the evaluation phase."""
        # Clear CUDA cache and run Python garbage collector *after* evaluation too
        torch.cuda.empty_cache()
        gc.collect()
        print("Cleared GPU cache and ran garbage collector after evaluation end.")
         # Optionally re-enable cache if you have subsequent steps that need it
         # Be cautious enabling this if memory is tight.
        # if model is not None and hasattr(model.config, "use_cache"):
        #     model.config.use_cache = True
        #     print("Re-enabled model.config.use_cache after evaluation.")
        print("--- Evaluation End ---\n")

# Clear any existing cached memory
torch.cuda.empty_cache()
accelerator = Accelerator(mixed_precision="bf16")

# Load the training file
train_dataset = load_dataset('json', data_files="/content/Instruction_SemEval14_Restaurants_Train.json")
validation_dataset = load_dataset('json', data_files="/content/Instruction_SemEval14_Restaurants_Validation.json")
# train_dataset # To check train_dataset structure
# validation_dataset # To check validation_dataset structure

import huggingface_hub
key_hf = "YOUR_API_KEY"
huggingface_hub.login(token=key_hf)

# Load the model and tokenizer
model_name = 'meta-llama/Llama-3.2-3B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Use bfloat16 instead of float16
    device_map="auto",          # Let the framework decide on optimal mapping
    use_cache=False,            # Explicitly disable KV cache to work with gradient checkpointing
    )

# Define the compute_metrics function with debugging
def compute_metrics(eval_pred):
    """
    Computes accuracy, F1, and macro F1 scores from model predictions,
    with added debugging print statements.

    Args:
        eval_pred (EvalPrediction): An object containing predictions and label_ids.
                                    Predictions are logits, label_ids are the true token IDs.

    Returns:
        dict: A dictionary containing the computed metrics.
    """

    predictions, labels = eval_pred

    # predictions are logits, so we need to take the argmax
    preds = np.argmax(predictions, axis=-1)


    # --- Crucial Step: Filter out ignored labels (-100) ---
    mask = labels != -100

    labels_flat = labels[mask]
    preds_flat = preds[mask]

    # Handle cases where filtering might result in empty arrays
    if len(labels_flat) == 0:
        # Returning metrics even in debug mode as expected by Trainer
        return {
            'accuracy': 0.0,
            'f1': 0.0,
            'macro_f1': 0.0,
        }

    # Calculate metrics
    try:
        accuracy = accuracy_score(y_true=labels_flat, y_pred=preds_flat)
        f1 = f1_score(y_true=labels_flat, y_pred=preds_flat, average='weighted', zero_division=0)
        macro_f1 = f1_score(y_true=labels_flat, y_pred=preds_flat, average='macro', zero_division=0)

        metrics_result = {
            'accuracy': accuracy,
            'f1': f1,
            'macro_f1': macro_f1,
        }
    except Exception as e:
        print(f"[DEBUG] ERROR calculating metrics: {e}")
        # Return default values in case of error during calculation
        metrics_result = {
            'accuracy': 0.0,
            'f1': 0.0,
            'macro_f1': 0.0,
        }

    return metrics_result

# Keep the increased max_length
MAX_LEN = 126

# Tokenize function for causal LM - Robust Masking Attempt
def tokenize_function(examples):
    # Prepare full chat history for templating
    messages_list = [
        [
            {"role": "user", "content": inst},
            {"role": "assistant", "content": resp}
        ]
        for inst, resp in zip(examples['instruction'], examples['response'])
    ]

    # Apply chat template to get the full formatted text including the response
    full_formatted_texts = [
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in messages_list
    ]

    # Tokenize the full formatted text (Input + Response)
    tokenized_inputs = tokenizer(
        full_formatted_texts,
        truncation=True,
        padding="max_length", # Pad to max_length
        max_length=MAX_LEN,
        return_tensors=None, # Return lists
    )

    # Initialize labels as a copy of input_ids
    labels = [list(ids) for ids in tokenized_inputs['input_ids']]

    # --- ROBUST MASKING ---
    # Iterate through each example in the batch
    for i in range(len(labels)):
        instruction = examples['instruction'][i]
        response = examples['response'][i]
        current_input_ids = tokenized_inputs['input_ids'][i]
        current_labels = labels[i]

        # Tokenize the response *separately* to get its token IDs
        # Crucially, do NOT add special tokens here
        response_token_ids = tokenizer.encode(response, add_special_tokens=False)

        if not response_token_ids:
            print(f"Warning: Response '{response}' tokenized to empty list for example {i}. Masking all.")
            for j in range(len(current_labels)):
                 current_labels[j] = -100
            continue # Skip to next example

        response_start_index = -1
        # Find the *last* occurrence of the response tokens in the input_ids sequence
        # We search from the end backwards because the response should be at the end.
        for k in range(len(current_input_ids) - len(response_token_ids), -1, -1):
             if current_input_ids[k : k + len(response_token_ids)] == response_token_ids:
                 # Basic check: Ensure it's not immediately preceded by BOS if response starts sequence
                 # A more robust check might involve looking for assistant prompt tokens before it,
                 # but that adds complexity. Let's assume last occurrence is the target.
                 response_start_index = k
                 break # Found the last occurrence

        if response_start_index == -1:
             # If response tokens are not found (e.g., due to truncation or tokenization quirks)
             print(f"Warning: Response tokens {response_token_ids} for '{response}' not found in input_ids for example {i}. Masking all.")
             # Mask all labels for this example as we can't identify the target
             for j in range(len(current_labels)):
                 current_labels[j] = -100
        else:
             # Mask everything *before* the response starts
             for j in range(response_start_index):
                 current_labels[j] = -100
             # Mask everything *after* the response ends (including padding)
             response_end_index = response_start_index + len(response_token_ids)
             for j in range(response_end_index, len(current_labels)):
                 current_labels[j] = -100

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

# Tokenize train and validation datasets
train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=16,
    remove_columns=['instruction', 'response'],
    num_proc=1,
    load_from_cache_file=False,
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=16,
    remove_columns=['instruction', 'response'],
    num_proc=1,
    load_from_cache_file=False,
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding="longest",
    return_tensors="pt",
    )

print(train_dataset)
print(validation_dataset)

# LoRA configuration
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    lora_dropout=0.01,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    bias="none",
    modules_to_save=None,
)

# Apply LoRA to Llama model
model = get_peft_model(model, peft_config)
model = accelerator.prepare(model)
# model.print_trainable_parameters()
model.config.use_cache = False

# peft_config
model.print_trainable_parameters()

# Hyperparameters
lr = 5e-5 # Step size of weight updates during optimization
batch_size = 8 # Number of examples processed per optimization step
num_epochs = 5 # Number of time model runs through training data

eval_subset_size = 100 # Or 100, etc.
small_eval_dataset = validation_dataset['train'].select(range(eval_subset_size))

# Ensure model parameters require gradients
model.train()  # Set model to training mode
for param in model.parameters():
    param.requires_grad_(True)  # Ensure all parameters require gradients

# Training arguments with memory optimizations
training_args = TrainingArguments(
    output_dir="./lora-sentiment-output",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=1,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    gradient_checkpointing=True,    # Enable gradient checkpointing to save memory
    bf16=True,                      # Use mixed precision
    fp16=False,                     # Use mixed precision
    gradient_accumulation_steps=32,  # Increased from 4 to 16 to reduce memory usage
    report_to="none",               # Disable reporting to save memory
    logging_steps=50,               # Reduce logging frequency
    save_total_limit=1,             # Keep only the best model
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    seed=42,
    max_grad_norm=1.0,
)

# Trainer
trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=train_dataset['train'],    # Training dataset
      eval_dataset=small_eval_dataset,       # Validation dataset
      compute_metrics=compute_metrics,
      data_collator=data_collator,
      processing_class=tokenizer, # No effect on training, but ensures tokenizer is saved with the model
      callbacks=[MemoryCleanupCallback()],
  )

print("Evaluating initial model state on small validation set...")
initial_eval_results = trainer.evaluate(eval_dataset=small_eval_dataset)
print(f"Initial Evaluation Results: {initial_eval_results}")

# Decide whether to proceed to training based on results
if initial_eval_results.get('eval_loss') is not None and np.isnan(initial_eval_results['eval_loss']):
     print("ERROR: Initial evaluation loss is NaN. Halting before training.")
else:
     print("Initial evaluation seems okay (or loss not computed). Proceeding to train...")

trainer.train()

trainer.save_model("./final-lora-sentiment-model")
