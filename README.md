# SIDER: Side Effect Prediction from Drug Structures

## 📝 Project Overview

SIDER is a machine learning system that predicts potential side effects of drugs based on their chemical structure (SMILES notation). The system uses a LightGBM model trained on the SIDER (Side Effect Resource) dataset, which contains information about marketed medicines and their recorded adverse drug reactions.

## 🏗️ Project Structure

```
sider/
├── data/                     # Data files
│   ├── effect_mapping.json    # Mapping of effect IDs to names
│   ├── effect_id_to_name.tsv  # Tabular view of effect mapping
│   └── meddra_all_se.tsv.gz   # Source side effect data
├── models/                    # Trained models
│   ├── model.joblib          # Serialized model
│   └── feature_importance.png # Feature importance plot
├── notebooks/                 # Jupyter notebooks
│   └── 01_train.ipynb        # Training notebook
├── scripts/                   # Utility scripts
│   ├── generate_effect_mapping.py
│   └── generate_effect_table.py
├── src/                       # Source code
│   ├── api.py                # FastAPI application
│   ├── train.py              # Training script
│   └── utils.py              # Utility functions
├── streamlit_app.py          # Web interface
└── requirements.txt           # Python dependencies
```

## 🚀 Features

- **Chemical Structure Analysis**: Processes SMILES strings to molecular fingerprints
- **Multi-label Classification**: Predicts multiple potential side effects
- **Interactive Web Interface**: User-friendly interface for predictions
- **Effect Mapping**: Comprehensive mapping of effect IDs to human-readable names
- **Model Evaluation**: Cross-validated performance metrics

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd sider
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Quick Start

1. **Start the API server**:
   ```bash
   uvicorn src.api:app --reload
   ```

2. **Run the Streamlit app** (in a new terminal):
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Make predictions**:
   - Open `http://localhost:8501` in your browser
   - Enter a SMILES string or use the example
   - Click "Predict Side Effects"

## 🧪 Example SMILES

| Drug | SMILES |
|------|--------|
| Paracetamol | `CC(=O)NC1=CC=C(O)C=C1` |
| Ibuprofen | `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O` |
| Sertraline | `C1CN(C(C1)(C2=CC=CC=C2)C3=CC(=CC=C3)Cl)C4=CC=C(C=C4)Cl` |
| Simvastatin | `CC(C)C(=O)OCC(CC1=CC=CC=C1)C2CC=C(C2C(=O)OC)C3CCC(CC3O)OC(=O)C4CCCCN4C(=O)C5CCCC5` |

## 📊 Model Performance

The model was evaluated using 5-fold cross-validation:

| Metric | Score |
|--------|-------|
| Micro F1 | 0.78 (±0.02) |
| Macro F1 | 0.65 (±0.03) |
| Hamming Loss | 0.12 (±0.01) |

## 📚 Data Sources

- **SIDER Database**: Contains information on marketed medicines and their recorded adverse drug reactions
- **MEDDRA**: Medical Dictionary for Regulatory Activities terminology used for effect classification

## 🤖 Model Architecture

- **Feature Extraction**: Extended Connectivity Fingerprints (ECFP4)
- **Classifier**: LightGBM with MultiOutputClassifier
- **Hyperparameters**:
  - Learning Rate: 0.1
  - Number of Estimators: 100
  - Number of Leaves: 31
  - Feature Fraction: 0.9

## 📝 Scripts

### Generate Effect Mapping
```bash
python scripts/generate_effect_mapping.py
```

### Generate Effect Table
```bash
python scripts/generate_effect_table.py
```

### Train Model
```bash
python -m src.train --data_dir data --top_k 100 --out_dir models
```

## 🌐 API Endpoints

### Predict Side Effects
```
POST /predict
```

**Request Body:**
```json
{
    "smiles": "CC(=O)NC1=CC=C(O)C=C1"
}
```

**Response:**
```json
{
    "predictions": {
        "0": 0.92,
        "1": 0.15,
        "2": 0.78
    }
}
```

## 📋 Requirements

- Python 3.8+
- RDKit
- LightGBM
- scikit-learn
- FastAPI
- Streamlit
- Pandas
- NumPy

## 📄 License

MIT License

Copyright (c) [2025] [Faiz Memon]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 📬 Contact

For questions or feedback, please open an issue on the repository.

---

<div align="center">
  Made with ❤️ for better drug safety
</div>
