import json
import os
from pathlib import Path

def convert_to_jsonl(input_path, output_path):
    """
    Convert a JSON file containing an array of objects to .jsonl format
    
    Args:
        input_path (str): Full path to the input JSON file
        output_path (str): Full path including desired filename for the output file
    """
    # Convert to Path objects for better path handling
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Check if input file exists
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' does not exist")
        return
    if not input_path.is_file():
        print(f"Error: '{input_path}' is not a file")
        return
    
    # Create output directory if it doesn't exist
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read the JSON file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Verify that the input is a list
        if not isinstance(data, list):
            raise ValueError("Input JSON must contain an array of objects")
            
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                # Verify each item is a dictionary
                if not isinstance(item, dict):
                    raise ValueError("Each item in the array must be an object")
                # Convert dictionary to JSON string and write with newline
                json_line = json.dumps(item)
                f.write(json_line + '\n')
                
        print(f"Successfully converted '{input_path}' to '{output_path}'")
        print(f"Total records processed: {len(data)}")
        
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{input_path}'")
    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except PermissionError:
        print(f"Error: Permission denied when accessing '{input_path}' or '{output_path}'")
    except Exception as e:
        print(f"Unexpected error occurred: {str(e)}")

def verify_output(output_path):
    """
    Optional function to verify the contents of the output file
    """
    output_path = Path(output_path)
    try:
        if not output_path.exists():
            print(f"Output file '{output_path}' not found for verification")
            return
            
        print(f"\nFirst few lines of '{output_path}':")
        with open(output_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 3:  # Show first 3 lines as preview
                    print("... (preview truncated)")
                    break
                print(line.strip())
    except Exception as e:
        print(f"Error verifying output: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Example paths - replace with your actual paths
    input_path = "C:\Seminar BA\SemEval data\JSON\Instruction_SemEval16_Restaurants_Validation.json"          # Full path required
    output_path = "C:\Seminar BA\SemEval data\JSON\Instruction_SemEval16_Restaurants_Validation.jsonl"       # Full path with desired filename
    
    # Convert the file
    convert_to_jsonl(input_path, output_path)
    
    # Optional: Verify the output
    verify_output(output_path)