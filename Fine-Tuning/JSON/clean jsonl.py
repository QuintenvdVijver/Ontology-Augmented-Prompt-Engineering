import json
import os

def clean_jsonl_file(input_file, output_file):
    try:
        # Check if input file exists using os
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        # Open input and output files
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            # Process each line
            for line in infile:
                try:
                    # Skip empty lines
                    if not line.strip():
                        continue
                    
                    # Load JSON from the line
                    json_obj = json.loads(line)
                    
                    # Recursively clean the JSON object
                    cleaned_obj = clean_json_object(json_obj)
                    
                    # Write cleaned JSON to output file
                    json.dump(cleaned_obj, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON line: {e}")
                    continue
                
        print(f"Successfully cleaned {input_file} and saved to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def clean_json_object(obj):
    """Recursively clean \u00a0 from JSON object"""
    if isinstance(obj, str):
        # Replace \u00a0 with a regular space or remove it
        return obj.replace('\u2013', '').strip()
    elif isinstance(obj, dict):
        return {key: clean_json_object(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_object(item) for item in obj]
    else:
        return obj

# Specify your file paths here
INPUT_FILE_PATH = "C:\Seminar BA\SemEval data\JSON\Instruction_SemEval16_Restaurants_Validation.jsonl"  # Replace with your actual input file path
OUTPUT_FILE_PATH = "C:\Seminar BA\SemEval data\JSON\Instruction_SemEval16_Restaurants_Validation_clean.jsonl"  # Replace with your actual output file path

if __name__ == "__main__":
    # Use the hardcoded file paths
    clean_jsonl_file(INPUT_FILE_PATH, OUTPUT_FILE_PATH)