import json
import xml.etree.ElementTree as ET
import huggingface_hub
import rank_bm25
import numpy as np
import torch
import tiktoken

from owlready2 import *
from openai import OpenAI
from rank_bm25 import BM25Okapi
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline, AutoModel, AutoTokenizer
from collections import defaultdict

# OpenAI response functions
key_openai = ''
key_deepinfra = ''

# Load the training file and ontology
domain_ontology_path = ""
train_file_path = ""
validation_file_path = ""

# Load the ontology
ontology = get_ontology(f"file://{domain_ontology_path}").load()

# Load training dataset for demonstration selection
train_tree = ET.parse(train_file_path)
train_root = train_tree.getroot()
train_sentences = train_root.findall(".//sentence")
train_corpus = [sentence.find("text").text for sentence in train_sentences]

# Load validation dataset for result generation
validation_tree = ET.parse(validation_file_path)
validation_root = validation_tree.getroot()
validation_sentences = validation_root.findall(".//sentence")

openai_client_openai = OpenAI(
    api_key= key_openai
)

openai_client_deepinfra = OpenAI(
    api_key= key_deepinfra,
    base_url = "https://api.deepinfra.com/v1/openai"
)

def get_response_gpt35turbo(openai_client, prompt, temperature=0):
    messages = [{"role":"user", "content":prompt}]
    output = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature = temperature
    )
    return output.choices[0].message.content

def get_response_gpt4omini(openai_client, prompt, temperature=0):
    messages = [{"role":"user", "content":prompt}]
    output = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature = temperature
    )
    return output.choices[0].message.content

def get_response_Llama370Binstruct(openai_client, prompt, temperature=0):
    messages = [{"role":"user", "content":prompt}]
    output = openai_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=messages,
        temperature = temperature
    )
    return output.choices[0].message.content

def get_response_Llama417Binstruct(openai_client, prompt, temperature=0):
    messages = [{"role":"user", "content":prompt}]
    output = openai_client.chat.completions.create(
        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        messages=messages,
        temperature = temperature
    )
    return output.choices[0].message.content

# BM25
def BM25_demonstration_selection(query_sentence, train_corpus, top_k):

    # Preparing BM25 - keyword-based demonstration selection
    tokenized_train_corpus = [sentence.split(" ") for sentence in train_corpus]
    bm25 = BM25Okapi(tokenized_train_corpus)

    # Tokenize the query sentence
    tokenized_query = query_sentence.lower().split()

    # BM25 demonstration selection
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]  # few shots demonstration

    return top_indices

# SimCSE
def precompute_train_embeddings(train_corpus, tokenizer, model):
    model.eval()  # Set to evaluation mode for faster inference

    train_inputs = tokenizer(train_corpus, padding=True, truncation=True, return_tensors="pt").to(device) #Tokenize training texts

    with torch.no_grad(): # Get the embeddings
        train_embeddings = SimCSE_model(**train_inputs, output_hidden_states=True, return_dict=True).pooler_output

    # Concatenate all batches into a single tensor
    return train_embeddings

def SimCSE_demonstration_selection(query_sentence, train_corpus, train_embeddings, tokenizer, model, top_k):
    # Tokenize and embed the query sentence
    query_input = tokenizer(query_sentence, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = model(**query_input, output_hidden_states=True, return_dict=True).pooler_output

    # Calculate cosine similarities
    similarities = cosine_similarity(query_embedding.cpu().detach().numpy(), train_embeddings.numpy())

    # Get indices of top k similar sentences
    top_indices = np.argsort(similarities[0])[::-1][:top_k]

    return top_indices

# Full ontology injection
def full_ontology_injection(ontology):
    branch_info = []

    # Helper function to format the class and its relations
    def format_class_relations(cls_list, indent=0):
        relations = []

        # Classes
        for cls in cls_list:

            # # Extract superclasses
            # superclasses = cls.ancestors()
            # if superclasses != cls:
            #     relations.append("  " * (indent + 1) + f"Superclasses: {', '.join([s.name for s in superclasses if hasattr(s, 'name')])}")

            # Class name
            relations.append("  " * (indent + 2) + f"Class: {cls.name}")

            # Help function to iteratively collect all subclasses
            subclasses = []
            def collect_all_subclasses(cls, subclasses_list, current_indent=0):
                direct_subclasses = list(cls.subclasses())
                for subclass in direct_subclasses:
                    subclasses_list.append((subclass, current_indent))
                    collect_all_subclasses(subclass, subclasses_list, current_indent + 1)

            # Subclasses
            collect_all_subclasses(cls, subclasses, indent + 3)
            for subclass, sub_indent in subclasses:
                relations.append("  " * sub_indent + f"Subclass: {subclass.name}")

                # Lexical forms
                lex_annotations = getattr(subclass, "lex", [])
                if lex_annotations:
                    relations.append("  " * (sub_indent + 1)+ f"Lexical Forms: {', '.join(lex_annotations)}")

                # Sentiment relations
                if "negative" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Negative")
                elif "positive" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Positive")
                elif "neutral" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Neutral")

            # Space
            relations.append("")

        return relations

    # Function to check if a class is a root class
    def is_root_class(cls):
        # A class is a root if it has no parents (is_a is empty) or only owl.Thing as parent
        parents = cls.is_a
        return not parents or (len(parents) == 1 and parents[0] == owl.Thing)

    root_classes = []
    # Find root classes
    for cls in ontology.classes():
        if is_root_class(cls) and 'sentiment' not in cls.name.lower():
            root_classes.append(cls)
    branch_info.extend(format_class_relations(root_classes))

    # Return extraced branches as a formatted string
    return "\n".join(branch_info)

# Aspect-based ontology injection
def aspect_based_ontology_injection(ontology, aspects_category):
    branch_info = []

    # Helper function to format the class and its relations
    def format_class_relations(cls_list, indent=0):
        relations = []

        # Classes
        for cls in cls_list:

            # Extract superclasses
            superclasses = cls.ancestors()
            if superclasses != cls:
                relations.append("  " * (indent + 1) + f"Superclasses: {', '.join([s.name for s in superclasses if hasattr(s, 'name')])}")

            # Class name
            relations.append("  " * (indent + 2) + f"Class: {cls.name}")

            # Help function to iteratively collect all subclasses
            subclasses = []
            def collect_all_subclasses(cls, subclasses_list, current_indent=0):
                direct_subclasses = list(cls.subclasses())
                for subclass in direct_subclasses:
                    subclasses_list.append((subclass, current_indent))
                    collect_all_subclasses(subclass, subclasses_list, current_indent + 1)

            # Subclasses
            collect_all_subclasses(cls, subclasses, indent + 3)
            for subclass, sub_indent in subclasses:
                relations.append("  " * sub_indent + f"Subclass: {subclass.name}")

                # Lexical forms
                lex_annotations = getattr(subclass, "lex", [])
                if lex_annotations:
                    relations.append("  " * (sub_indent + 1)+ f"Lexical Forms: {', '.join(lex_annotations)}")

                # Sentiment relations
                if "negative" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Negative")
                elif "positive" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Positive")
                elif "neutral" in subclass.name.lower():
                    relations.append("  " * (sub_indent + 1) + f"Sentiment Relation: Neutral")

            # Space
            relations.append("")

        return relations

    for aspect_category in aspects_category:
        search_aspect = aspect_category.split("#")[0].lower()

        branch_info.append(f"\nOntology Branch(es) for Aspect '{search_aspect}':")

        # Search for lex that match the aspect (assuming a class name is always also in lex property)
        root_classes = []
        for cls in ontology.classes():

            lex_annotations = getattr(cls, "lex", [])
            aspect_annotations = getattr(cls, "aspect", [])

            # Appending cls to root_classes if aspect matches with a lex property or aspect_annotation
            if (any(search_aspect == lex.lower() for lex in lex_annotations) or
                any(search_aspect == aspect.lower().split("#")[0] for aspect in aspect_annotations)):
                root_classes.append(cls)

        # Extract branches for all root classes
        if root_classes:
            # branch_info.append(f"Root Class for Aspect '{search_aspect}':")
            branch_info.extend(format_class_relations(root_classes))
        else:
            branch_info.append(f"No matching root class found for aspect '{search_aspect}'.")

    # Return extraced branches as a formatted string
    return "\n".join(branch_info)

# Evaluation metrics
def evaluation(validation_sentences, results):

    ground_truth = []
    for sentence in validation_sentences:
        sentence_id = sentence.get("id")
        opinions = sentence.findall(".//Opinion")
        sentence_truth = {opinion.get("target"): opinion.get("polarity").capitalize() for opinion in opinions}
        ground_truth.append(sentence_truth)

    filtered_ground_truth = []
    filtered_results = []

    # Filter out invalid JSON, aspect mismatches
    for truth, predicted in zip(ground_truth, results):

        # Skip if JSON is invalid
        try:
            predicted = json.loads(predicted)
        except json.JSONDecodeError as e:
            continue

        # Extract aspects
        truth_aspects = list(truth.keys())
        predicted_aspects = list(predicted.keys())

        # Keep observation only if aspects are exactly equal
        if truth_aspects == predicted_aspects:
            filtered_ground_truth.append(truth)
            filtered_results.append(predicted)

    polarities_true = []
    polarities_predicted = []

    for truth, predicted in zip(filtered_ground_truth, filtered_results):

        for aspect, true_polarity in truth.items():
            predicted_polarity = predicted.get(aspect).capitalize()
            polarities_true.append(true_polarity.capitalize())
            polarities_predicted.append(predicted_polarity)

    labels = ["Positive", "Negative", "Neutral"]

    # Accuracy
    accuracy = accuracy_score(polarities_true, polarities_predicted)

    # F1 score
    f1 = f1_score(polarities_true, polarities_predicted, labels=labels, average="weighted", zero_division=0)

    # Macro F1 score
    macro_f1 = f1_score(polarities_true, polarities_predicted, labels=labels, average="macro", zero_division=0)

    return {
        "accuracy": accuracy * 100,
        "f1": f1*100,
        "macro_f1": macro_f1 * 100
    }

# Base prompt
base_prompt = """
Instruction:

Your task is aspect-based sentiment classification.
Assign a polarity (positive, neutral, negative) to the given aspect(s) in the following sentence based on its context and
the provided demonstrations.

Domain Ontology:
{domain_ontology}

Demonstrations:
{demonstrations}

Tested sample:
- Sentence: {sentence}
- Aspects: {aspects}

Output:
Generate the answer in a compact JSON format with no newlines or indentation, containing the following fields:
- {aspects} - string that is one of the polarities ("Positive", "Negative", "Neutral")

Always respond with a valid JSON. Do not invlude any extra characters, symbols, or text in or outside the JSON itself (including backticks, ", /)

"""

# Main
for sentence in validation_sentences:
    sentence_id = sentence.get("id")
    sentence_text = sentence.find("text").text

    # Get the list of aspects and corresponding categories for the sentence
    opinions = sentence.findall(".//Opinion")
    aspects = [opinion.get("target") for opinion in opinions] # Currently I don't do anything with this
    aspects_category = [opinion.get("category") for opinion in opinions]

    # Demonstration selection (SELECT CORRECT ONE)
    shots = 3
    top_indices = SimCSE_demonstration_selection(sentence_text, train_corpus, train_embeddings, tokenizer, SimCSE_model, shots)
    top_indices = BM25_demonstration_selection(sentence_text, train_corpus, shots)

    # Ontology injection (SELECT CORRECT ONE)
    ontology_injection = full_ontology_injection(ontology)
    ontology_injection = aspect_based_ontology_injection(ontology, aspects_category)

    # Format demonstrations
    demonstrations = []
    for i in top_indices:
        demo = train_sentences[i]
        demo_id = demo.get("id")
        demo_text = demo.find("text").text
        demo_opinions = demo.findall(".//Opinion")
        demo_aspect_polarity_pairs = [
            {"aspect": opinion.get("target"), "polarity": opinion.get("polarity")}
            for opinion in demo_opinions
        ]

        demo_pairs_str = ", ".join(
            [f"{pair['aspect']} ({pair['polarity']})" for pair in demo_aspect_polarity_pairs]
        ) if demo_aspect_polarity_pairs else "None"
        demonstrations.append(f"Sentence: {demo_text}\nAspects: {demo_pairs_str}")

    demonstrations = "\n\n".join(demonstrations)

    # Preparing prompt
    prompt = base_prompt.format(
        domain_ontology=ontology_injection,
        demonstrations=demonstrations,
        sentence=sentence_text,
        aspects=", ".join(aspects) if aspects else "None",
        sentence_id=sentence_id
    )
    print(prompt)

    # Output generation (SELECT CORRECT ONE)
    output = get_response_gpt35turbo(openai_client_openai, prompt)
    output = get_response_gpt4omini(openai_client_openai, prompt)
    output = get_response_Llama370Binstruct(openai_client_deepinfra, prompt)
    output = get_response_Llama417Binstruct(openai_client_deepinfra, prompt)

    results.append(output)

final_results = evaluation(validation_sentences, results)
print(final_results)
