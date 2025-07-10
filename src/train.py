import joblib
import argparse
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, hamming_loss, classification_report, precision_recall_fscore_support
from lightgbm import LGBMClassifier
from src.utils import load_sider, build_smiles_map, smiles_to_fp

def main(data_dir, top_k, out_dir):
    # Load and inspect the data
    df = load_sider(data_dir)
    print("\n=== Data Overview ===")
    print(f"Total rows: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\n=== Missing Values ===")
    print(df.isnull().sum())
    
    # Check unique drugs and side effects
    print("\n=== Unique Counts ===")
    print(f"Unique drugs: {df['DRUG_NAME'].nunique()}")
    print(f"Unique side effects: {df['MEDDRA_ID'].nunique()}")
    
    # Group by drug and collect side effects
    drug_se = df.groupby('DRUG_NAME')['MEDDRA_ID'].apply(list).reset_index()
    
    # Debug: Show first few drugs and their side effects
    print("\n=== First few drugs and their side effects ===")
    for i, (_, row) in enumerate(drug_se.head().iterrows(), 1):
        print(f"{i}. {row['DRUG_NAME']} - {len(row['MEDDRA_ID'])} side effects")
    
    # Prepare labels
    print("\nPreparing labels...")
    mlb = MultiLabelBinarizer()
    Y_full = mlb.fit_transform(drug_se['MEDDRA_ID'])
    print(f"Shape of Y_full: {Y_full.shape}")
    
    # Filter out rare side effects and select top-K
    side_effect_counts = Y_full.sum(axis=0)
    
    # Only keep side effects that appear at least 5 times (adjust this number as needed)
    min_samples = 5
    common_effects = side_effect_counts >= min_samples
    Y_common = Y_full[:, common_effects]
    
    print(f"Original number of side effects: {Y_full.shape[1]}")
    print(f"Number of side effects with at least {min_samples} samples: {Y_common.shape[1]}")
    
    # Now select the top-K most frequent from the common ones
    top_k = min(top_k, Y_common.shape[1])  # In case we have fewer than top_k
    top_idx = np.argsort(Y_common.sum(axis=0))[::-1][:top_k]
    Y = Y_common[:, top_idx]
    
    print(f"Selected {Y.shape[1]} most frequent side effects (all with at least {min_samples} samples)")
    print(f"Shape of Y: {Y.shape}")

    # SMILES → fingerprints
    print("\nBuilding SMILES to fingerprint mapping...")
    smiles_map = build_smiles_map(df)
    
    # Debug: Print some SMILES mapping info
    print("\n=== SMILES Mapping Info ===")
    print(f"Total drugs in mapping: {len(smiles_map)}")
    print("\nFirst few SMILES mappings:")
    for i, (stitch_id, smi) in enumerate(list(smiles_map.items())[:5], 1):
        print(f"{i}. {stitch_id}: {smi}")
    
    # Create a mapping from drug name to STITCH_ID
    drug_to_stitch = dict(zip(df['DRUG_NAME'], df['STITCH_ID']))
    
    fps, keep_idx = [], []
    total_drugs = len(drug_se['DRUG_NAME'])
    found_smiles = 0
    
    print("\n=== Processing Drugs ===")
    for i, row in drug_se.iterrows():
        drug_name = row['DRUG_NAME']
        stitch_id = drug_to_stitch.get(drug_name)
        
        if not stitch_id:
            print(f"Warning: No STITCH_ID found for drug: {drug_name}")
            continue
            
        smi = smiles_map.get(stitch_id)
        if not smi:
            print(f"Warning: No SMILES found for STITCH_ID: {stitch_id} (Drug: {drug_name})")
            continue
            
        try:
            fp = smiles_to_fp(smi)
            if fp is not None:
                fps.append(fp)
                keep_idx.append(i)
                found_smiles += 1
                if found_smiles <= 5:  # Print first 5 successful conversions
                    print(f"Successfully processed: {drug_name} (STITCH_ID: {stitch_id}) -> {smi[:50]}...")
                elif found_smiles == 6:
                    print("... (additional drugs processed, omitting details)")
        except Exception as e:
            print(f"Error processing SMILES for {drug_name} (STITCH_ID: {stitch_id}): {str(e)}")
    
    print(f"Found SMILES strings for {found_smiles} out of {total_drugs} drugs")
    
    if not fps:
        raise ValueError("No valid SMILES strings found. Check your data and SMILES mapping.")
    
    print(f"Successfully processed {len(fps)} drugs with valid fingerprints")
    X_fp = np.stack(fps)
    Y = Y[keep_idx]

    # For multi-label data, we need to use a different approach for train/test split
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import f1_score, hamming_loss, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Convert to dense if sparse
    if hasattr(Y, 'toarray'):
        Y = Y.toarray()
    
    # Better LightGBM parameters for multi-label classification
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_child_samples': 20,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'n_jobs': -1,
        'verbose': -1,
        'seed': 42
    }
    
    # Cross-validation setup
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics
    micro_f1_scores = []
    macro_f1_scores = []
    hamming_losses = []
    
    print(f"\n=== Starting {n_splits}-fold cross-validation ===")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_fp), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        
        # Split data
        X_train, X_val = X_fp[train_idx], X_fp[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        
        # Initialize the model with proper parameters
        base_estimator = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            random_state=42,
            verbose=-1  # Suppress LightGBM output
        )
        
        # Use MultiOutputClassifier for multi-label classification
        from sklearn.multioutput import MultiOutputClassifier
        model = MultiOutputClassifier(base_estimator, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        micro_f1 = f1_score(y_val, y_pred, average='micro')
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        h_loss = hamming_loss(y_val, y_pred)
        
        micro_f1_scores.append(micro_f1)
        macro_f1_scores.append(macro_f1)
        hamming_losses.append(h_loss)
        
        print(f"Micro F1: {micro_f1:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Hamming Loss: {h_loss:.4f}")
    
    # Print average metrics
    print("\n=== Cross-validation Results ===")
    print(f"Average Micro F1: {np.mean(micro_f1_scores):.4f} (+/- {np.std(micro_f1_scores):.4f})")
    print(f"Average Macro F1: {np.mean(macro_f1_scores):.4f} (+/- {np.std(macro_f1_scores):.4f})")
    print(f"Average Hamming Loss: {np.mean(hamming_losses):.4f} (+/- {np.std(hamming_losses):.4f})")
    
    # Train final model on full training set
    print("\nTraining final model on full training set...")
    
    # Initialize the final model with the same parameters as in CV
    final_base_estimator = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    from sklearn.multioutput import MultiOutputClassifier
    final_model = MultiOutputClassifier(final_base_estimator, n_jobs=-1)
    final_model.fit(X_fp, Y)
    
    # Save model and metadata
    os.makedirs(out_dir, exist_ok=True)
    
    # Get unique MEDDRA ID to name mapping
    meddra_map = df[['MEDDRA_ID', 'SEPTYPE']].drop_duplicates()
    # Create a proper mapping from MEDDRA_ID to SEPTYPE
    meddra_dict = dict(zip(meddra_map['MEDDRA_ID'], meddra_map['SEPTYPE']))
    
    # Get the top k most frequent side effects
    effect_counts = Y.sum(axis=0)
    top_k = min(100, len(mlb.classes_))  # Ensure we don't exceed the number of classes
    top_indices = np.argsort(effect_counts)[-top_k:][::-1]  # Get indices of top k effects
    
    # Get the effect names for the top k effects and clean them up
    effect_names = []
    for i in top_indices:
        meddra_id = mlb.classes_[i]
        name = meddra_dict.get(meddra_id, f'effect_{i}')
        
        # Clean up the name
        if isinstance(name, str):
            # Remove 'PT' prefix and any extra whitespace
            name = name.replace('PT', '').strip()
            # If name is empty after cleaning, use a default
            if not name:
                name = f'Effect_{i}'
            # Capitalize first letter of each word
            name = ' '.join(word.capitalize() for word in name.split() if word)
        else:
            name = f'Effect_{i}'
            
        effect_names.append(name)
    
    # Debug: Print some example mappings
    print("\nExample MEDDRA ID to name mappings:")
    for i, idx in enumerate(top_indices[:5]):
        print(f"  {mlb.classes_[idx]} -> {effect_names[i]}")
    
    # Define the model path
    model_path = os.path.join(out_dir, 'model.joblib')
    
    # Filter the model to only predict the top k effects
    from sklearn.multioutput import MultiOutputClassifier
    
    if isinstance(final_model, MultiOutputClassifier):
        # For MultiOutputClassifier, we'll create a new instance with filtered estimators
        filtered_estimators = []
        filtered_classes = []
        
        # Ensure top_indices is a sorted list of integers
        top_indices = sorted(top_indices)
        
        for i in top_indices:
            if i < len(final_model.estimators_):
                filtered_estimators.append(final_model.estimators_[i])
                # Handle both cases where classes_ is a list of arrays or a single array
                if isinstance(final_model.classes_, list) and i < len(final_model.classes_):
                    filtered_classes.append(final_model.classes_[i])
        
        # Create a new MultiOutputClassifier with filtered estimators
        filtered_model = MultiOutputClassifier(final_model.estimator, n_jobs=final_model.n_jobs)
        filtered_model.estimators_ = filtered_estimators
        # Set the filtered classes
        filtered_model.classes_ = filtered_classes if filtered_classes else final_model.classes_
        
        # Copy other necessary attributes
        for attr in ['n_features_in_', 'estimators_']:
            if hasattr(final_model, attr):
                setattr(filtered_model, attr, getattr(final_model, attr))
                
        final_model = filtered_model
    elif hasattr(final_model, 'classes_'):
        # For single-output models that support classes_
        try:
            filtered_model = type(final_model)()
            # Handle both numpy array and list cases
            if isinstance(final_model.classes_, np.ndarray):
                filtered_model.classes_ = final_model.classes_[top_indices]
            elif isinstance(final_model.classes_, list):
                filtered_model.classes_ = [final_model.classes_[i] for i in top_indices if i < len(final_model.classes_)]
            else:
                filtered_model.classes_ = final_model.classes_
                
            # Copy other necessary attributes
            for attr in ['coef_', 'intercept_', 'n_features_in_', 'estimators_', 'feature_importances_']:
                if hasattr(final_model, attr):
                    setattr(filtered_model, attr, getattr(final_model, attr))
            final_model = filtered_model
        except Exception as e:
            print(f"Warning: Could not filter model classes: {str(e)}")
    
    # Save the model, MultiLabelBinarizer, and effect names
    joblib.dump({
        'model': final_model,
        'mlb': mlb,
        'effect_names': effect_names,  # Save the actual effect names
        'top_indices': top_indices,    # Save the indices of the top effects
        'params': params,
        'metrics': {
            'micro_f1': micro_f1_scores,
            'macro_f1': macro_f1_scores,
            'hamming_loss': hamming_losses
        }
    }, model_path)
    
    print(f"\nModel saved to {model_path}")
    
    # Plot feature importance (for the first target as an example)
    try:
        plt.figure(figsize=(12, 8))
        feature_importances = final_model.estimators_[0].feature_importances_
        top_n = min(20, len(feature_importances))
        sorted_idx = np.argsort(feature_importances)[-top_n:]
        
        plt.barh(range(top_n), feature_importances[sorted_idx], align='center')
        plt.yticks(range(top_n), [f'Bit {i}' for i in sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Important Features (First Target)')
        
        importance_path = f'{out_dir}/feature_importance.png'
        plt.tight_layout()
        plt.savefig(importance_path)
        print(f"Feature importance plot saved to {importance_path}")
    except Exception as e:
        print(f"Could not generate feature importance plot: {str(e)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--top_k", type=int, default=100)
    p.add_argument("--out_dir", default=".")
    args = p.parse_args()
    main(**vars(args))
