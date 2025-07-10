import streamlit as st
import requests
import pandas as pd
import json
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO

# Configuration
API_URL = "http://localhost:8000/predict"
EFFECT_MAPPING_FILE = Path("data/effect_mapping.json")

# Load effect mapping
@st.cache_data
def load_effect_mapping():
    with open(EFFECT_MAPPING_FILE, 'r') as f:
        return json.load(f)

def draw_molecule(smiles):
    """Draw molecule from SMILES string"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol, size=(300, 200))
        return img
    return None

# Page config
st.set_page_config(
    page_title="Drug Side‑Effect Explorer",
    page_icon="💊",
    layout="wide"
)

# Sidebar for input
with st.sidebar:
    st.title("💊 SIDER Explorer")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["Predict", "Effect Mapping"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if page == "Predict":
        st.header("Input Molecule")
        smiles = st.text_input(
            "Enter SMILES string:",
            "CCOc1ccc2nc(S(N)(=O)=O)sc2c1",
            help="Enter a valid SMILES string or use the example"
        )
        
        # Display molecule
        img = draw_molecule(smiles)
        if img:
            st.image(img, caption="Molecule Structure", use_container_width=True)
        
        if st.button("Predict Side Effects", type="primary"):
            st.session_state.predict = True
            st.session_state.show_mapping = False

# Main content
if page == "Predict":
    st.title("Drug Side‑Effect Predictor")
    
    if st.session_state.get('predict', False):
        with st.spinner("Predicting side effects..."):
            try:
                # Make the API request
                resp = requests.post(API_URL, json={"smiles": smiles}, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                
                # Check if we got an error response
                if "error" in data:
                    st.error(f"Error from API: {data.get('details', 'Unknown error')}")
                else:
                    preds = data.get("predictions", {})
                    
                    if not preds:
                        st.warning("No predictions returned from the model")
                    else:
                        # Load effect mapping
                        effect_mapping = load_effect_mapping()
                        
                        # Convert to DataFrame and sort
                        effects_data = []
                        for effect_id, prob in preds.items():
                            if prob > 0.1:  # Only show effects with >10% probability
                                # Extract numeric ID from 'Effect_X' format or use as-is
                                try:
                                    if isinstance(effect_id, str) and effect_id.startswith('Effect_'):
                                        effect_num = int(effect_id.split('_')[1])
                                    else:
                                        effect_num = int(effect_id)
                                    effect_key = str(effect_num)
                                except (ValueError, IndexError):
                                    effect_num = effect_id
                                    effect_key = str(effect_id)
                                
                                effects_data.append({
                                    'Effect ID': effect_num,
                                    'Effect Name': effect_mapping.get(effect_key, {}).get('name', f'Effect {effect_id}'),
                                    'Probability': prob,
                                    'Description': effect_mapping.get(effect_key, {}).get('description', '')
                                })
                        
                        df = pd.DataFrame(effects_data).sort_values('Probability', ascending=False)
                        
                        # Display results
                        st.subheader("Predicted Side Effects")
                        
                        # Show top 3 high-risk effects
                        high_risk = df[df['Probability'] > 0.7]
                        if not high_risk.empty:
                            st.warning("⚠️ **High Risk Effects Detected**")
                            for _, row in high_risk.head(3).iterrows():
                                st.markdown(f"- **{row['Effect Name']}** ({row['Probability']:.1%} risk)")
                        
                        # Show all effects in a nice table
                        st.subheader("All Predicted Effects")
                        
                        # Format the table
                        df_display = df[['Effect Name', 'Probability', 'Description']].copy()
                        df_display['Probability'] = df_display['Probability'].map('{:.1%}'.format)
                        
                        # Display with expandable details
                        for _, row in df.iterrows():
                            with st.expander(f"{row['Effect Name']} ({row['Probability']:.1%})"):
                                st.markdown(f"**Description:** {row['Description']}")
                                st.markdown(f"**Probability:** {row['Probability']:.1%}")
                                st.markdown(f"**Effect ID:** {row['Effect ID']}")
                        
                        # Add metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Effects Predicted", len(df))
                        with col2:
                            st.metric("High Risk (≥70%)", len(high_risk))
                        with col3:
                            med_risk = len(df[(df['Probability'] >= 0.3) & (df['Probability'] < 0.7)])
                            st.metric("Medium Risk (30-70%)", med_risk)
            
            except requests.exceptions.RequestException as e:
                st.error(f"Error making prediction: {str(e)}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)  # Show full traceback for debugging

# Effect Mapping Page
elif page == "Effect Mapping":
    st.title("Effect ID to Name Mapping")
    st.markdown("This table shows the mapping between effect IDs used in the model and their corresponding names and descriptions.")
    
    # Load and display the mapping
    effect_mapping = load_effect_mapping()
    
    # Convert to DataFrame for display
    effects = []
    for effect_id, data in effect_mapping.items():
        effects.append({
            'Effect ID': int(effect_id),
            'MEDDRA ID': data['meddra_id'],
            'Effect Name': data['name'],
            'Description': data['description']
        })
    
    df = pd.DataFrame(effects).sort_values('Effect ID')
    
    # Add search and filter
    st.markdown("### Search Effects")
    search_term = st.text_input("Search by name or description:", "")
    
    if search_term:
        mask = (df['Effect Name'].str.contains(search_term, case=False)) | \
               (df['Description'].str.contains(search_term, case=False))
        df = df[mask]
    
    # Show the table
    st.dataframe(
        df,
        column_config={
            "Effect ID": st.column_config.NumberColumn("Effect ID"),
            "MEDDRA ID": "MEDDRA ID",
            "Effect Name": "Effect Name",
            "Description": "Description"
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Add download button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Effect Mapping (CSV)",
        data=csv,
        file_name='effect_mapping.csv',
        mime='text/csv',
    )
else:
    st.info("Enter a SMILES string and click 'Predict Side Effects' to see predictions.")
    st.markdown("### Example SMILES strings:")
    examples = {
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)CC(=O)O",
        "Paracetamol": "CC(=O)NC1=CC=C(C=C1)O"
    }
    for name, example_smiles in examples.items():
        if st.button(f"{name}: {example_smiles}"):
            st.session_state.predict = True
            st.rerun()
