#!/usr/bin/env python3
"""
Script to generate a tabular view of effect ID to name mapping.
"""
import json
import pandas as pd
from pathlib import Path

def main():
    # Path to the effect mapping file
    mapping_file = Path('data/effect_mapping.json')
    output_file = Path('data/effect_id_to_name.tsv')
    
    # Load the mapping
    with open(mapping_file, 'r') as f:
        effect_mapping = json.load(f)
    
    # Convert to a list of dictionaries with additional details
    effects = []
    for effect_id, data in effect_mapping.items():
        effects.append({
            'Effect ID': int(effect_id),
            'MEDDRA ID': data['meddra_id'],
            'Effect Name': data['name'],
            'Description': data['description']
        })
    
    # Convert to DataFrame and sort by Effect ID
    df = pd.DataFrame(effects).sort_values('Effect ID')
    
    # Format the output
    pd.set_option('display.max_colwidth', 50)
    pd.set_option('display.width', 1000)
    
    # Save as TSV
    df.to_csv(output_file, sep='\t', index=False)
    
    # Also save as markdown table
    markdown_file = output_file.with_suffix('.md')
    with open(markdown_file, 'w') as f:
        f.write("# Side Effect Mapping\n\n")
        f.write("This table shows the mapping between effect IDs used in the model and their corresponding names.\n\n")
        f.write("| Effect ID | MEDDRA ID | Effect Name | Description |\n")
        f.write("|-----------|-----------|-------------|-------------|\n")
        for _, row in df.iterrows():
            f.write(f"| {row['Effect ID']} | {row['MEDDRA ID']} | {row['Effect Name']} | {row['Description']} |\n")
    
    print(f"Effect mapping table saved to {output_file}")
    print(f"Markdown table saved to {markdown_file}")
    
    # Print a sample
    print("\n=== Sample of the Effect Mapping Table ===")
    print("\nFirst 10 effects:")
    print(df.head(10).to_string(index=False, justify='left'))
    
    print("\nLast 5 effects:")
    print(df.tail(5).to_string(index=False, justify='left'))
    
    print(f"\nTotal effects: {len(df)}")
    print("\nNote: This includes only the top 100 effects used in the model.")

if __name__ == "__main__":
    main()
