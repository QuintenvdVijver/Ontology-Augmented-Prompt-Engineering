# Knowledge and Instruction Fine-Tuning for ABSC
This code can be used to fine-tune and validate various models. We fine-tuned Llama-3.2-3B-Instruct through huggingface, fine-tuned Llama-3.1-8B-Instruct through Nebius AI Studio and fine-tuned GPT-3.5 Turbo using the OpenAI API Platform.

## Before running the code
- Set up the environment
  - Create a Google Collab file (or something similar)
    - We were lucky to be able to use the Nvidia A100 GPU with 40GB RAM (you need close to 40GB RAM to be able to fine-tune the Llama-3.2-3B-Instruct in Google Collab)
  - Run `pip install -r requirements.txt` in your terminal
- Set up data
  - The data and ontologies can be found at `Data/Domain ontologies` and `Data/Raw SemEval data`
  - For fine-tuning in Google Collab, we pre-processed the SemEval datasets.
    - First we transformed the datasets from xml to json
  - For fine-tuning in Nebius AI Studio and OpenAI API Platform, we used the `jsonToJsonl.py`, `clean jsonl.py` (to remove irregularities), and `jsonlToFinetuneReady.py` (to get the `{"message": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "neutral"}]}` format)
  - The processed json and jsonl files can be found in the `Fine-Tuning/JSON` and `Fine-Tuning/JSON/JSONL` directory

## Training and validating process Google Collab
Note these steps are for the `Instruction Fine-Tuning.py` to fine-tune the Llama-3.2-3B-Instruct model in Google Collab

1. Change the `train_dataset` and `validation_dataset` directory to the respective training and validation/test json datasets
2. Get and fill in your Huggingface API
3. Change the LoRA hyperparameters to your liking

After following these steps and running the code, you should be able to see the different evaluation metrics after running `trainer.save_model("./final-lora-sentiment-model")` and then rerunning `def compute_metrics(eval_pred)`

## Training and validating process Nebius AI Studio
1. Make an account at Nebius AI Studio and deposit some money (you'll get a long way with just $5)
2. Go to Fine-Tuning and pick the right model (Llama-3.1-8B-Instruct)
3. Throw in the training and test datasets and set hyperparameters
4. Run fine-tuning job
5. Choose a checkpoint and deploy model
6. Use the Nebius API and your fine-tuned model id and compute metrics in `Metrics for Fine-Tuned Nebius.py`

## Training and validating process OpenAI API Platform
1. Make an account at OpenAI API Platform and deposit some money
2. Go to Fine-Tuning and pick the right model (GPT-3.5 Turbo)
3. Throw in the training and test datasets and set hyperparameters
4. Run fine-tuning job
5. Use the OpenAI API and your fine-tuned model id and compute metrics in `Metrics for Fine-Tuned GPT.py`
