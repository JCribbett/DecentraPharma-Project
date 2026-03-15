import os
import requests
import json

IPFS_API_URL = "http://127.0.0.1:5001/api/v0/add"

def upload_file_to_ipfs(file_path):
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None

    print(f"Uploading {os.path.basename(file_path)}...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(IPFS_API_URL, files={'file': f})
        
        if response.status_code == 200:
            result = response.json()
            ipfs_hash = result['Hash']
            print(f"✅ Success! CID: {ipfs_hash}")
            return ipfs_hash
        else:
            print(f"❌ Failed to upload. Status code: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the local IPFS daemon.")
        return None

def main():
    print("--- DecentraPharma IPFS Uploader ---\n")
    
    # List of key artifacts to share with the decentralized network
    files_to_upload = [
        "best_gnn_model.pt",
        "smiles_generator_rl.pth",
        "virtual_screening_hits.csv",
        "generated_hits.png"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uploaded_records = {}
    
    # Check connection first
    try:
        requests.post("http://127.0.0.1:5001/api/v0/version", timeout=2)
    except requests.exceptions.ConnectionError:
        print("🚨 CRITICAL: Cannot connect to IPFS daemon!")
        print("Please start IPFS Desktop or run 'ipfs daemon' in your terminal.")
        return

    for filename in files_to_upload:
        file_path = os.path.join(script_dir, filename)
        cid = upload_file_to_ipfs(file_path)
        if cid:
            uploaded_records[filename] = cid
            
    if uploaded_records:
        record_file = os.path.join(script_dir, "ipfs_manifest.json")
        with open(record_file, "w") as f:
            json.dump(uploaded_records, f, indent=4)
        print(f"\n📦 Manifest saved to {os.path.basename(record_file)}")
        print("Share these CIDs with your nodes so they can download your models!")

if __name__ == "__main__":
    main()