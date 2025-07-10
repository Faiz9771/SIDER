import numpy as np, requests, json, time, gzip, pandas as pd, random
from rdkit import Chem
from rdkit.Chem import AllChem

# ---------- Data loading ----------
def load_sider(files_dir):
    # Read meddra_all_se.tsv.gz (gzipped)
    se = pd.read_csv(f"{files_dir}/meddra_all_se.tsv.gz", sep='\t',
                     compression='gzip', header=None,
                     names=['STITCH_ID','SIDER_ID','MEDDRA_ID',
                            'SEPTYPE','FREQ','HL_FREQ'])
    
    # Read drug_names.tsv (uncompressed)
    drug_names = pd.read_csv(f"{files_dir}/Drug Names.tsv", sep='\t',
                             header=None,
                             names=['STITCH_ID','DRUG_NAME'])
    return se.merge(drug_names, on='STITCH_ID', how='left')

# ---------- PubChem lookup ----------
def stitch_to_cid(stitch_id):
    """Convert STITCH ID to PubChem CID.
    
    Args:
        stitch_id: Can be in format 'CIDxxxx', 'SIDxxxx', or numeric ID
        
    Returns:
        str: Numeric CID as string, or None if conversion not possible
    """
    if not stitch_id or pd.isna(stitch_id):
        return None
        
    sid = str(stitch_id).strip().upper()
    
    # If already a CID (starts with CID)
    if sid.startswith('CID'):
        return sid[3:].lstrip('0') or '0'  # Return numeric part, handle CID0000 case
    
    # If it's a SID (starts with SID), we can't directly convert to CID
    if sid.startswith('SID'):
        print(f"Warning: SID {sid} cannot be directly converted to CID")
        return None
    
    # If it's just a number, assume it's already a CID
    if sid.isdigit():
        return sid
    
    # If we get here, it's an unexpected format
    print(f"Warning: Could not convert STITCH ID to CID: {stitch_id}")
    return None

def cid_to_smiles(cid):
    """Convert PubChem CID to SMILES string using PubChem REST API."""
    if not cid or str(cid).lower() == 'nan':
        return None
    
    # Clean the CID - remove 'CID' prefix if present and any whitespace
    cid = str(cid).strip()
    if cid.upper().startswith('CID'):
        cid = cid[3:].strip()
    
    # Make sure we have a valid numeric CID
    if not cid.isdigit():
        print(f"Invalid CID format: {cid}")
        return None
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/JSON"
    headers = {
        'User-Agent': 'SIDER Project (your-email@example.com)',
        'Accept': 'application/json'
    }
    
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and data['PropertyTable']['Properties']:
                    return data['PropertyTable']['Properties'][0].get('CanonicalSMILES')
                return None
            
            # Handle rate limiting
            if response.status_code == 429:  # Too Many Requests
                retry_after = int(response.headers.get('Retry-After', 10))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                continue
                
            if response.status_code == 404:
                print(f"CID {cid} not found in PubChem")
                return None
                
            print(f"PubChem API error (attempt {attempt + 1}/{max_retries}): {response.status_code} - {response.text}")
            
        except requests.exceptions.RequestException as e:
            print(f"Request error for CID {cid} (attempt {attempt + 1}/{max_retries}): {str(e)}")
        
        # Exponential backoff with jitter
        sleep_time = (2 ** attempt) + (random.random() * 0.5)
        time.sleep(sleep_time)
    
    return None

def build_smiles_map(df, cache_json="smiles_cache.json"):
    """Build a mapping from STITCH IDs to SMILES strings.
    
    Args:
        df: DataFrame containing STITCH_ID column
        cache_json: Path to cache file
        
    Returns:
        dict: Mapping from STITCH_ID to SMILES string or None if not found
    """
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(os.path.abspath(cache_json)) or '.', exist_ok=True)
    
    # Try to load from cache
    if os.path.exists(cache_json):
        try:
            with open(cache_json, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Cache file {cache_json} is corrupted. Rebuilding... {str(e)}")
    
    print("Building SMILES mapping (this may take a while)...")
    smiles_map = {}
    unique_sids = df['STITCH_ID'].unique()
    total = len(unique_sids)
    
    for i, sid in enumerate(unique_sids, 1):
        if i % 10 == 0 or i == 1 or i == total:
            print(f"Processing compound {i}/{total}...")
        
        cid = stitch_to_cid(sid)
        if not cid:
            smiles_map[sid] = None
            continue
            
        smi = cid_to_smiles(cid)
        smiles_map[sid] = smi
        
        # Save progress every 100 compounds
        if i % 100 == 0 or i == total:
            with open(cache_json, 'w') as f:
                json.dump(smiles_map, f)
    
    return smiles_map

# ---------- Fingerprints ----------
def smiles_to_fp(smiles, n_bits=1024, radius=2):
    """Convert a SMILES string to a molecular fingerprint.
    
    Args:
        smiles (str): Input SMILES string
        n_bits (int): Number of bits in the fingerprint
        radius (int): Morgan fingerprint radius
        
    Returns:
        numpy.ndarray: Fingerprint as a binary vector, or None if conversion fails
    """
    if not smiles or not isinstance(smiles, str):
        print("Invalid SMILES: Input is empty or not a string")
        return None
        
    try:
        # Sanitize SMILES and convert to molecule
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"Could not parse SMILES: {smiles}")
            return None
            
        # Generate fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, 
            radius=radius, 
            nBits=n_bits,
            useChirality=True,  # Include chirality information
            useBondTypes=True,  # Include bond type information
            useFeatures=False    # Don't use feature-based invariants
        )
        
        # Convert to numpy array
        arr = np.zeros((n_bits,), dtype=np.int8)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
        
    except Exception as e:
        print(f"Error generating fingerprint for SMILES {smiles}: {str(e)}")
        return None
