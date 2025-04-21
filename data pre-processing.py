import xml.etree.ElementTree as ET
import json

# Definition to convert 2014 XML structure to 2015/2016 XML structure
def convert_semeval14_to_15_16_format(semeval14_path, output_path):
    # Converts SemEval14 dataset to SemEval15/16 format 
    
    # Load SemEval14 dataset
    tree = ET.parse(semeval14_path)
    root = tree.getroot()
    
    # Create a new root and subroot elements for the new structure
    new_root = ET.Element("Reviews")
    review = ET.SubElement(new_root, "Review")
    sentences = ET.SubElement(review, "sentences")
    
    # Iterate over all sentences in the dataset
    for sentence in root.findall("sentence"):
        sentence_id = sentence.get("id")
        text = sentence.find("text").text

        # Create new sentence subroot element
        new_sentence = ET.SubElement(sentences, "sentence", id=sentence_id)
        ET.SubElement(new_sentence, "text").text = text
        
        # Create new opinion subroot element
        opinions = ET.SubElement(new_sentence, "Opinions")

        # Convert <aspectTerms> into <Opinions>
        aspect_terms = sentence.find("aspectTerms")
        if aspect_terms is not None:
            # Case 1: aspectTerm is present
            for aspect in aspect_terms.findall("aspectTerm"):
                target = aspect.get("term")
                polarity = aspect.get("polarity")
                from_idx = aspect.get("from")
                to_idx = aspect.get("to")
                ET.SubElement(opinions, "Opinion", target=target, category="", polarity=polarity, from_=from_idx, to=to_idx)
        else:
            # Case 2: no aspectTerm is present, set target = NULL
            ET.SubElement(opinions, "Opinion", target="NULL")

    # Write the converted XML to the output file
    tree = ET.ElementTree(new_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Converted SemEval14 dataset saved to: {output_path}")

# Definition to delete the implict aspect from the dataset (only works for 2015/2016 XML structure!)
def delete_implicit_aspects(input_path, output_path):
    #deleting the implicit aspects (target = null) from the dataset

    # Load input file
    tree = ET.parse(input_path)
    root = tree.getroot()

    #Iterature over all reviews and sentences
    for review in root.findall('.//Review'):
        sentences = review.find('sentences')
        if sentences is None:
            continue
            
        # Create a list to store sentences to remove
        sentences_to_remove = []
        
        for sentence in sentences.findall('sentence'):
            opinions_elem = sentence.find('Opinions')
            if opinions_elem is None:
                sentences_to_remove.append(sentence)
                continue
                
            opinions = opinions_elem.findall('Opinion')
 
            # Count NULL targets and total opinions
            null_targets = sum(1 for opinion in opinions if opinion.get('target') == 'NULL')
            conflict_polarities = sum(1 for opinion in opinions if opinion.get('polarity') == 'conflict')
            total_opinions = len(opinions)
            
            if null_targets == total_opinions or conflict_polarities == total_opinions:
                # All targets are NULL, mark sentence for removal
                sentences_to_remove.append(sentence)
            else:
                # Some NULL targets or conflict polarities exist, remove only those opinions
                for opinion in opinions[:]:  # Create a copy to modify during iteration
                    if opinion.get('target') == 'NULL'or opinion.get('polarity') == 'conflict':
                        opinions_elem.remove(opinion)
        
        # Remove marked sentences
        for sentence in sentences_to_remove:
            sentences.remove(sentence)

    # Write the modified XML to the output file
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"Dataset with implicit aspects removed saved to: {output_path}")

# Definition to remove intersections between train and test data from the train data (only works for 2015/2016 XML structure!)
def remove_intersections(training_input_path, test_input_path, training_output_path):

    def extract_sentences_from_xml(input_file):
            # Parse the XML string
            root = ET.parse(input_file)
            
            # Create a set to store sentence texts
            sentences = set()
            
            # Find all sentence elements and extract their text
            for sentence in root.findall('.//sentence'):
                text_elem = sentence.find('text')
                if text_elem is not None and text_elem.text:
                    sentences.add(text_elem.text.strip())
            
            return sentences
    
    def intersection(training_file, test_file):

        # Extract sentences from both datasets
        train_sentences = extract_sentences_from_xml(training_file)
        valid_sentences = extract_sentences_from_xml(test_file)

        # Find intersection of sentences (common ones)
        common_sentences = train_sentences.intersection(valid_sentences)
    
        # Return count of common sentences
        return common_sentences
        
    # Load input files
    training_tree = ET.parse(training_input_path)
    training_root = training_tree.getroot()
    intersection = intersection(training_input_path, test_input_path)

    #Iterature over all reviews and sentences
    for review in training_root.findall('.//Review'):
        sentences = review.find('sentences')
        if sentences is None:
            continue
            
        # Create a list to store sentences to remove
        sentences_to_remove = []
        
        for sentence in sentences.findall('sentence'):
            
            # If sentence text in intersection, remove sentence
            if sentence.find('text').text.strip() in intersection:
                sentences_to_remove.append(sentence)

        # Remove marked sentences
        for sentence in sentences_to_remove:
            sentences.remove(sentence)

    # Write the modified XML to the output file
    training_tree.write(training_output_path, encoding="utf-8", xml_declaration=True)
    print(f"Training Dataset with intersections removed saved to: {training_output_path}")


# Definition to convert XML to JSON for instruction fine-tuning
def instruction_finetuning_preprocessing(input_path, output_path):

    # Load the XML file
    tree = ET.parse(input_path)

    json_data = []
    for review in tree.findall('.//Review'):

        for sentence in review.findall('.//sentence'):
            sentence_text = sentence.find('text').text

            for opinion in sentence.findall('.//Opinion'):
                aspect = opinion.get('target')
                polarity = opinion.get('polarity')

                instruction = f"Given the sentence '{sentence_text}' and the aspect '{aspect}', what is the sentiment?"

                json_entry = {
                    "instruction": instruction,
                    "response": polarity
                }
                json_data.append(json_entry)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"JSON file has been created at: {output_path}")
