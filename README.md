# Ontology-Augmented Prompt Engineering for Aspect–Based Sentiment Classification
This code can be used to conduct ontology prompt injection into various LLM models. Models we used were GPT-3.5 Turbo, GPT-4o mini, Llama-3-70B Instruct, and Llama-4-Scout-17B-16E Instruct.

## Before running the code
- Set up environment:
  - Set up your Google Collab (or something similar) with Python 3.10
  - Run `pip install -r requirements.txt` in your terminal
- Set up data:
  - The data and ontologies can be found at `Data/Raw SemEval data` & `Data/Domain ontologies`
  - Using the `data pre-processing.py` file, preprocess the data as follows:
    1. For 2014 data, convert the XML structure to the 2015/2016 XML structure
    2. Remove the implicit aspects from the data
    3. Remove the intersections between the training and test data from the training data

## Running the code
1. Open the `Ontology prompt injection.py` file
2. Get and fill in your OpenAI API and DeepInfra API KEY
3. Fill in the respective file paths of the domain ontology, train and validation (test) datasets
4. Since this code includes all options (no injection, full injection, and aspect-based injection), different demonstration strategies (no demonstrations, BM25 demonstrations, or SimCSE) and all models, some code needs to be commented ('#' put in front of it) for the desired result, such as:
  - **Prompt** - Depending on whether you want to run the code with or without ontology injection, with or without demonstrations, you have to comment out `Domain Ontology: \n {domain_ontology}` and/or `Demonstrations: \n {demonstrations}` in `base_prompt`
  - **Demonstration Selection** - In the main, you have to comment out either `top_indices = SimCSE_demonstration_selection(...)` or `top_indices = BM25_demonstration_selection(...)` to get either demonstration selection (if you do not want any demonstration selection, you can remove the demonstrations part in the base prompt and leave one or both uncommented).
  - **Ontology Injeciton** - Just below that you have to comment out `ontology_injection = full_ontology_injection(...)` or `ontology_injection = aspect_based_ontology_injection(...)` (the one you do not want to use).
  - **Model** - To get the output for the model you which to get output from, you can comment out the other models of which the code looks like this: `output = get_response_...`

After following these steps and running the code, you should be able to see the three different evaluation metrics (accuracy, f1, and macro f1)
