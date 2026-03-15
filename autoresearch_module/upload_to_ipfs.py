import os
import requests
import json

PINATA_API_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"

def upload_file_to_ipfs(file_path, headers):
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return None

    print(f"Uploading {os.path.basename(file_path)} to Pinata...")
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(PINATA_API_URL, files={'file': f}, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            ipfs_hash = result['IpfsHash']
            print(f"✅ Success! CID: {ipfs_hash}")
            return ipfs_hash
        else:
            print(f"❌ Failed to upload. Status code: {response.status_code}, {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to Pinata.")
        return None

def main():
    print("--- DecentraPharma IPFS Uploader (via Pinata) ---\n")
    
    headers = {}
    pinata_jwt = os.environ.get("DECENTRAPHARMA_PINATA_JWT") or os.environ.get("PINATA_JWT")
    pinata_api_key = os.environ.get("DECENTRAPHARMA_PINATA_API_KEY") or os.environ.get("PINATA_API_KEY")
    pinata_secret_api_key = os.environ.get("DECENTRAPHARMA_PINATA_API_SECRET") or os.environ.get("PINATA_SECRET_API_KEY")
    
    if pinata_jwt and pinata_jwt.startswith("ey"):
        headers["Authorization"] = f"Bearer {pinata_jwt}"
    elif pinata_api_key and pinata_secret_api_key:
        headers["pinata_api_key"] = pinata_api_key
        headers["pinata_secret_api_key"] = pinata_secret_api_key
    else:
        print("🚨 CRITICAL: Pinata authentication missing or incomplete!")
        print("Please set DECENTRAPHARMA_PINATA_JWT, OR both DECENTRAPHARMA_PINATA_API_KEY and DECENTRAPHARMA_PINATA_API_SECRET.")
        return
    
    # List of key artifacts to share with the decentralized network
    files_to_upload = [
        "best_gnn_model.pt",
        "smiles_generator_rl.pth",
        "virtual_screening_hits.csv",
        "generated_hits.png"
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    uploaded_records = {}
    
    for filename in files_to_upload:
        file_path = os.path.join(script_dir, filename)
        cid = upload_file_to_ipfs(file_path, headers)
        if cid:
            uploaded_records[filename] = cid
            
    if uploaded_records:
        record_file = os.path.join(script_dir, "ipfs_manifest.json")
        with open(record_file, "w") as f:
            json.dump(uploaded_records, f, indent=4)
        print(f"\n📦 Manifest saved to {os.path.basename(record_file)}")
        print("Share these CIDs with your nodes so they can download your models!")

if __name__ == "__main__":
    # Load .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    os.environ[line.split("=", 1)[0].strip()] = line.split("=", 1)[1].strip().strip("'\"")
    main()