import json
import os

def transform_jsonl(input_path, output_path):
    """
    Transform a JSONL file from the original format to the new format.
    
    Args:
        input_path (str): Path to the input JSONL file.
        output_path (str): Path to the output JSONL file.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    # Parse the input JSON line
                    data = json.loads(line.strip())
                    
                    # Extract instruction and response
                    instruction = data.get('instruction', '')
                    response = data.get('response', '')
                    
                    # Create the new structure
                    new_data = {
                        "messages": [
                            {"role": "user", "content": instruction},
                            {"role": "assistant", "content": response}
                        ]
                    }
                    
                    # Write the transformed data as a JSON line
                    json.dump(new_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in line: {line.strip()}. Error: {e}")
                except Exception as e:
                    print(f"Error processing line: {line.strip()}. Error: {e}")
                    
        print(f"Transformation complete. Output written to {output_path}")
        
    except FileNotFoundError:
        print(f"Input file not found: {input_path}")
    except PermissionError:
        print(f"Permission denied when accessing files: {input_path} or {output_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    # Define input and output paths (modify these as needed)
    input_path = "C:\Seminar BA\SemEval data\JSON\Instruction_SemEval16_Restaurants_Validation_clean.jsonl"          # Full path required
    output_path = "C:\Seminar BA\SemEval data\JSON\Instruction_SemEval16_Restaurants_Validation.jsonl" 
    
    # Transform the JSONL file
    transform_jsonl(input_path, output_path)

if __name__ == "__main__":
    main()