#!/usr/bin/env python3
"""
Script to generate a JSON mapping of effect numbers to their names.
"""
import os
import json
import joblib
import pandas as pd
import argparse
from pathlib import Path

def load_effect_mapping(model_path, data_dir='data'):
    """Load effect names from the trained model and original data."""
    try:
        # Load the model data
        model_data = joblib.load(model_path)
        
        # Get MEDDRA IDs from the model
        mlb = model_data.get('mlb')
        if mlb is None:
            print("Warning: No MultiLabelBinarizer found in model")
            return {}
            
        # Get the top indices used in the model
        top_indices = model_data.get('top_indices')
        if top_indices is None:
            print("Warning: No top_indices found in model, using first 100 effects")
            top_indices = range(100)
        
        # Get the MEDDRA IDs in the order used by the model
        all_meddra_ids = mlb.classes_
        meddra_ids = [all_meddra_ids[i] for i in top_indices[:100]]  # Only take top 100
        
        print(f"Using top {len(meddra_ids)} effects from the model")
        
        # Load the original data to get effect names
        # Use the symlink to avoid issues with spaces in filename
        data_file = Path(data_dir) / 'meddra_all_se.tsv.gz'
        if data_file.exists():
            print(f"Loading effect names from {data_file}...")
            # The file format is: STITCH_ID, SIDER_ID, MEDDRA_ID, SEPTYPE, FREQ, HL_FREQ, EFFECT_NAME
            # We only care about MEDDRA_ID and EFFECT_NAME
            df = pd.read_csv(data_file, sep='\t', compression='gzip', header=None,
                          names=['STITCH_ID', 'SIDER_ID', 'MEDDRA_ID', 'SEPTYPE', 'FREQ', 'EFFECT_NAME'])
            
            print(f"Sample data from file:\n{df[['MEDDRA_ID', 'SEPTYPE', 'EFFECT_NAME']].head()}")
            
            # Create a mapping from MEDDRA_ID to EFFECT_NAME
            # Prefer 'PT' (Preferred Term) type when available
            pt_terms = df[df['SEPTYPE'] == 'PT'][['MEDDRA_ID', 'EFFECT_NAME']].drop_duplicates()
            pt_terms = pt_terms[~pt_terms['MEDDRA_ID'].duplicated()]  # Take first occurrence if multiple PTs per ID
            effect_name_map = dict(zip(pt_terms['MEDDRA_ID'], pt_terms['EFFECT_NAME']))
            
            # For any IDs without a PT, use the first available name
            missing_ids = set(df['MEDDRA_ID']) - set(effect_name_map.keys())
            if missing_ids:
                other_terms = df[df['MEDDRA_ID'].isin(missing_ids)]
                other_terms = other_terms[~other_terms['MEDDRA_ID'].duplicated()]
                other_terms_map = dict(zip(other_terms['MEDDRA_ID'], other_terms['EFFECT_NAME']))
                effect_name_map.update(other_terms_map)
            print(f"Loaded {len(effect_name_map)} effect names from data")
        else:
            print(f"Warning: Data file {data_file} not found. Using generic names.")
            effect_name_map = {}
        
        # Create the mapping with actual names
        mapping = {}
        for i, meddra_id in enumerate(meddra_ids):
            # Get the effect name from the map, or use a default
            name = effect_name_map.get(meddra_id, f"Effect_{i}")
            # Clean up the name
            if isinstance(name, str):
                name = name.replace('PT', '').strip()
                if not name:
                    name = f"Effect_{i}"
                # Capitalize first letter of each word
                name = ' '.join(word.capitalize() for word in name.split() if word)
            else:
                name = f"Effect_{i}"
            
            mapping[i] = {
                'effect_id': i,
                'meddra_id': str(meddra_id) if meddra_id is not None else f"EFFECT_{i}",
                'name': name,
                'description': f"{name} (MEDDRA: {meddra_id})" if meddra_id else name
            }
            
        return mapping
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return {}

def save_mapping(mapping, output_file):
    """Save the mapping to a JSON file."""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Save with pretty printing
        with open(output_file, 'w') as f:
            json.dump(mapping, f, indent=2)
            
        print(f"Mapping saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving mapping: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate effect name mapping from trained model')
    parser.add_argument('--model', type=str, default='models/model.joblib',
                       help='Path to the trained model file')
    parser.add_argument('--data', type=str, default='data',
                       help='Directory containing meddra_all_se.tsv.gz')
    parser.add_argument('--output', type=str, default='data/effect_mapping.json',
                       help='Output JSON file path')
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    model_path = Path(args.model).resolve()
    data_dir = Path(args.data).resolve()
    output_path = Path(args.output).resolve()
    
    # Load the mapping
    print(f"Loading effect mapping from {model_path}...")
    mapping = load_effect_mapping(model_path, data_dir)
    
    if not mapping:
        print("No mapping could be generated. Please check the model file.")
        return
    
    # Save the mapping
    print(f"Found {len(mapping)} effects")
    save_mapping(mapping, output_path)
    
    # Print sample
    print("\nSample of the mapping:")
    for i in range(min(5, len(mapping))):
        print(f"  {mapping[i]}")

if __name__ == "__main__":
    main()
