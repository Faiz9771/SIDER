import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now import other packages
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from src.utils import smiles_to_fp

# Use absolute path for model file
model_path = project_root / "models" / "model.joblib"

try:
    # Load the model
    clf = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
    
    # Create dummy labels if not available
    # Replace 100 with the actual number of classes your model predicts
    labels = [f"effect_{i}" for i in range(100)]  # Adjust the number as needed
    print("Using auto-generated effect labels")
    
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

app = FastAPI(title="SIDER Side‑Effect API")

class Query(BaseModel):
    smiles: str

@app.post("/predict")
def predict(q: Query):
    try:
        # Convert SMILES to fingerprint and ensure it's in the correct format
        fp = smiles_to_fp(q.smiles).reshape(1, -1)
        
        # Get model and mlb
        model = clf['model']
        mlb = clf['mlb']
        
        # Suppress feature name warnings
        import warnings
        from sklearn.exceptions import DataConversionWarning
        warnings.filterwarnings('ignore', category=DataConversionWarning)
        warnings.filterwarnings('ignore', message='X does not have valid feature names, but XgbDMatrix was created and passed to DMatrix.')
        
        # Make predictions
        try:
            # First try to get probabilities if available
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(fp)
                # Handle different output formats
                if isinstance(probs, list):
                    # For multi-output models
                    if len(probs) > 0 and hasattr(probs[0], 'shape') and len(probs[0].shape) > 1:
                        probs = np.array([p[0][1] if p.shape[1] > 1 else p[0][0] for p in probs])
                    else:
                        probs = np.array(probs).flatten()
                elif hasattr(probs, 'shape') and len(probs.shape) > 1:
                    # For single output models with probability for each class
                    probs = probs[0] if probs.shape[0] == 1 else probs[:, 1]
                else:
                    probs = np.array(probs).flatten()
            else:
                # If no predict_proba, use predict
                probs = model.predict(fp)
                if isinstance(probs, list):
                    probs = np.array(probs).flatten()
                elif hasattr(probs, 'shape'):
                    probs = probs.flatten()
                else:
                    probs = np.array([probs])
            
            # Ensure probs is a 1D array
            probs = np.asarray(probs).flatten()
            print(f"Raw probabilities shape: {probs.shape}")
            print(f"First few probabilities: {probs[:5]}")
            
            # Get class names if available
            try:
                print("Available keys in clf:", clf.keys())
                
                # Try to get effect names in order of preference
                if 'effect_names' in clf and clf['effect_names'] is not None:
                    print(f"Found {len(clf['effect_names'])} effect names in model")
                    if len(clf['effect_names']) == len(probs):
                        class_names = clf['effect_names']
                    else:
                        print(f"Warning: Mismatch in effect names ({len(clf['effect_names'])}) and probabilities ({len(probs)})")
                        class_names = clf['effect_names'][:len(probs)]
                        if len(class_names) < len(probs):
                            class_names.extend([f"effect_{i}" for i in range(len(class_names), len(probs))])
                elif hasattr(mlb, 'classes_') and mlb.classes_ is not None:
                    print(f"Found {len(mlb.classes_)} classes in mlb")
                    if len(mlb.classes_) == len(probs):
                        class_names = mlb.classes_
                    else:
                        print(f"Warning: Mismatch in mlb classes ({len(mlb.classes_)}) and probabilities ({len(probs)})")
                        class_names = mlb.classes_[:len(probs)]
                        if len(class_names) < len(probs):
                            class_names = list(class_names) + [f"effect_{i}" for i in range(len(class_names), len(probs))]
                else:
                    print("No effect names found, using numeric indices")
                    class_names = [f"effect_{i}" for i in range(len(probs))]
                
                print(f"Using {len(class_names)} class names")
                print(f"First few class names: {class_names[:5]}")
                    
            except Exception as e:
                print(f"Error processing class names: {str(e)}")
                class_names = [f"effect_{i}" for i in range(len(probs))]
                
            # Convert all probabilities to Python native types
            predictions = {}
            for i, (name, prob) in enumerate(zip(class_names, probs)):
                try:
                    # Skip empty or invalid names
                    if not name or name == 'PT':
                        name = f'Effect_{i}'
                        
                    # Try to convert numpy types to Python native types
                    if hasattr(prob, 'item') and hasattr(prob, 'shape') and prob.shape == ():
                        prob_val = prob.item()
                    elif hasattr(prob, 'item'):
                        prob_val = float(prob.item())
                    else:
                        prob_val = float(prob)
                        
                    # Only include predictions with probability > 0.1 (10%)
                    if prob_val > 0.1:
                        predictions[name] = prob_val
                        
                except (TypeError, ValueError) as e:
                    print(f"Error processing prediction {i}: {str(e)}")
                    continue
            
            return {"predictions": predictions}
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}\n{str(type(probs))} - {probs}"
            print(error_msg)
            return {"error": error_msg, "status": "error"}
        
    except Exception as e:
        import traceback
        error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "error": "An error occurred during prediction",
            "details": str(e),
            "status": "error"
        }