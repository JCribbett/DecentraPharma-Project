import os
import requests

def download_pdb(pdb_id, output_dir="data"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    
    print(f"Downloading {pdb_id.upper()} from RCSB Protein Data Bank...")
    response = requests.get(url)
    
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Successfully downloaded HIV target protein to {output_path}")
    else:
        print(f"❌ Failed to download {pdb_id}. Status code: {response.status_code}")

if __name__ == "__main__":
    # 3OXZ: Crystal structure of HIV-1 Reverse Transcriptase 
    download_pdb("3OXZ", output_dir=os.path.dirname(os.path.abspath(__file__)))