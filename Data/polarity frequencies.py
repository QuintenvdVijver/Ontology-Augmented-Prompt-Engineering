import xml.etree.ElementTree as ET
import json
from collections import Counter

def polarity_frequencies(input_path):
    polarity_counts = Counter()
    
    tree = ET.parse(input_path)
    root = tree.getroot()

    # Iterate through all Opinion elements
    for opinion in root.findall('.//Opinion'):
        polarity = opinion.get('polarity').lower()
        polarity_counts[polarity] += 1

    # Calculate total for frequency percentages
    total = sum(polarity_counts.values())

    polarity_frequencies = {
        polarity: count / total * 100 
        for polarity, count in polarity_counts.items()
    }

    return {
        'frequencies': {k: f"{v:.2f}%" for k, v in polarity_frequencies.items()},
        'total': total
    }