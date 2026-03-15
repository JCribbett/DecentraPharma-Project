import time, random, logging, os, sys
import requests
import torch

try: 
    from rdkit import Chem
except ImportError: 
    print("Install RDKit: pip install rdkit-pypi")
    sys.exit(1)

# Dynamically import your latest smart architectures
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'autoresearch_module'))
from best_model import AntiviralGNN
from data_loader import mol_to_graph
from torch_geometric.loader import DataLoader
from generate_3d import generate_3d_conformer
from run_docking import run_docking
from download_target import download_pdb
from upload_to_ipfs import upload_file_to_ipfs

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

IPFS_GATEWAY = "https://ipfs.io/ipfs/"

class Node:
    def __init__(self, node_id, model_cid): 
        self.node_id = node_id
        self.model_cid = model_cid
        self.model_path = os.path.join(os.path.dirname(__file__), "downloaded_model.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.server_url = os.environ.get("DECENTRAPHARMA_SERVER_URL", "http://127.0.0.1:8000")
        
        logging.info(f"Node {self.node_id} booting up...")
        self.download_model()
        self.load_model()

    def download_model(self):
        """Fetches the smartest GNN model directly from the IPFS blockchain."""
        if os.path.exists(self.model_path):
            logging.info("Model already cached locally. Skipping download.")
            return
            
        logging.info(f"Downloading model from IPFS (CID: {self.model_cid})...")
        response = requests.get(f"{IPFS_GATEWAY}{self.model_cid}")
        if response.status_code == 200:
            with open(self.model_path, 'wb') as f:
                f.write(response.content)
            logging.info("✅ Model downloaded successfully from decentralized storage!")
        else:
            logging.error(f"Failed to download model. HTTP {response.status_code}")
            
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = AntiviralGNN().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.eval()
            logging.info("🧠 Neural Network loaded into memory and ready.")

    def fetch_task(self): 
        try:
            response = requests.get(f"{self.server_url}/task")
            if response.status_code == 200:
                task = response.json()
                logging.info(f"⬇️ Fetched Task {task['task_id']} ({task['type']}): {task['smiles'][:25]}...")
                return task
        except requests.exceptions.ConnectionError:
            logging.warning("⚠️ Cannot connect to central dispatch server. Retrying...")
        return None
        
    def process_task(self, task): 
        if task["type"] == "2d_scoring":
            if not self.model: return {"status": "error", "message": "Model not loaded"}
            mol = Chem.MolFromSmiles(task['smiles'])
            if not mol: return {"status": "error"}
            
            data = mol_to_graph(mol)
            if data is None or data.x.numel() == 0: return {"status": "error"}
            
            loader = DataLoader([data], batch_size=1)
            batch = next(iter(loader)).to(self.device)
            
            with torch.no_grad():
                try:
                    edge_attr = getattr(batch, 'edge_attr', None)
                    out = self.model(batch.x, batch.edge_index, batch.batch, edge_attr=edge_attr)
                except TypeError:
                    out = self.model(batch.x, batch.edge_index, batch.batch)
                prob = torch.sigmoid(out).item()
                
            logging.info(f"⚙️ Task {task['task_id']} Evaluated: Predicted HIV Active Prob = {prob:.4f}")
            return {"status": "success", "hiv_probability": prob}
            
        elif task["type"] == "3d_docking":
            smiles = task['smiles']
            target_cid = task.get('target_cid')
            
            # Target setup
            script_dir = os.path.join(os.path.dirname(__file__), '..', 'autoresearch_module')
            
            if target_cid:
                pdb_path = os.path.join(script_dir, f"target_{target_cid}.pdb")
                if not os.path.exists(pdb_path):
                    logging.info(f"Downloading target protein from IPFS (CID: {target_cid})...")
                    response = requests.get(f"{IPFS_GATEWAY}{target_cid}")
                    if response.status_code == 200:
                        with open(pdb_path, 'wb') as f:
                            f.write(response.content)
                    else:
                        return {"status": "error", "message": f"Failed to download CID {target_cid}"}
            else:
                target_pdb_id = "3OXZ"
                pdb_path = os.path.join(script_dir, f"{target_pdb_id.lower()}.pdb")
                if not os.path.exists(pdb_path):
                    logging.info(f"Downloading target protein {target_pdb_id}...")
                    download_pdb(target_pdb_id, output_dir=script_dir)
            
            # File paths
            project_dir = os.path.dirname(script_dir)
            sdf_filename = f"task_{task['task_id']}.sdf"
            docked_filename = f"docked_{task['task_id']}.sdf"
            ligand_path = os.path.join(project_dir, sdf_filename)
            docked_path = os.path.join(project_dir, docked_filename)
            
            # Generate 3D conformer
            logging.info(f"⚙️ Task {task['task_id']}: Generating 3D conformer...")
            success = generate_3d_conformer(smiles, ligand_path)
            if not success:
                return {"status": "error", "message": "Failed 3D generation"}
            
            # Run docking
            logging.info(f"⚙️ Task {task['task_id']}: Running Smina docking...")
            affinity = run_docking(
                receptor_file=pdb_path,
                ligand_file=ligand_path,
                output_file=docked_path,
                script_dir=script_dir
            )
            
            # Upload highly successful results to IPFS
            ipfs_cid = None
            if affinity is not None and affinity <= -6.5:
                logging.info(f"🏆 High affinity detected ({affinity} kcal/mol)! Uploading result to IPFS...")
                
                headers = {}
                pinata_jwt = os.environ.get("DECENTRAPHARMA_PINATA_JWT") or os.environ.get("PINATA_JWT")
                pinata_api_key = os.environ.get("DECENTRAPHARMA_PINATA_API_KEY") or os.environ.get("PINATA_API_KEY")
                pinata_secret_api_key = os.environ.get("DECENTRAPHARMA_PINATA_API_SECRET") or os.environ.get("PINATA_SECRET_API_KEY")
                
                if pinata_jwt and pinata_jwt.startswith("ey"):
                    headers["Authorization"] = f"Bearer {pinata_jwt}"
                elif pinata_api_key and pinata_secret_api_key:
                    headers["pinata_api_key"] = pinata_api_key
                    headers["pinata_secret_api_key"] = pinata_secret_api_key
                
                if headers:
                    ipfs_cid = upload_file_to_ipfs(docked_path, headers)
                else:
                    logging.warning("⚠️ No Pinata credentials found in environment. Skipping IPFS upload.")
            
            # Cleanup
            if os.path.exists(ligand_path): os.remove(ligand_path)
            if os.path.exists(docked_path): os.remove(docked_path)
            
            if affinity is not None:
                logging.info(f"⚙️ Task {task['task_id']} Docking complete! Affinity = {affinity} kcal/mol")
                return {"status": "success", "docking_affinity": affinity, "ipfs_cid": ipfs_cid}
            else:
                return {"status": "error", "message": "Docking failed"}
                
        elif task["type"] == "3d_folding":
            sequence = task.get("sequence")
            if not sequence: return {"status": "error", "message": "No sequence provided"}
            
            logging.info(f"⚙️ Task {task['task_id']}: Folding protein sequence ({len(sequence)} amino acids) using ESMFold...")
            
            try:
                # Use Meta's ESMFold public API for decentralized folding without requiring local TBs of databases
                response = requests.post('https://api.esmatlas.com/foldSequence/v1/pdb/', data=sequence, timeout=120)
                
                if response.status_code == 200:
                    project_dir = os.path.dirname(os.path.dirname(__file__))
                    pdb_path = os.path.join(project_dir, f"folded_task_{task['task_id']}.pdb")
                    
                    with open(pdb_path, 'w') as f:
                        f.write(response.text)
                        
                    logging.info("✅ Protein folded successfully! Uploading 3D PDB structure to IPFS...")
                    
                    headers = {}
                    pinata_jwt = os.environ.get("DECENTRAPHARMA_PINATA_JWT") or os.environ.get("PINATA_JWT", "")
                    pinata_api_key = os.environ.get("DECENTRAPHARMA_PINATA_API_KEY") or os.environ.get("PINATA_API_KEY")
                    pinata_secret_api_key = os.environ.get("DECENTRAPHARMA_PINATA_API_SECRET") or os.environ.get("PINATA_SECRET_API_KEY")
                    
                    if pinata_jwt.startswith("ey"):
                        headers["Authorization"] = f"Bearer {pinata_jwt}"
                    elif pinata_api_key and pinata_secret_api_key:
                        headers["pinata_api_key"] = pinata_api_key
                        headers["pinata_secret_api_key"] = pinata_secret_api_key
                        
                    ipfs_cid = upload_file_to_ipfs(pdb_path, headers) if headers else None
                    
                    if os.path.exists(pdb_path): os.remove(pdb_path)
                    
                    return {"status": "success", "message": "Protein folded", "ipfs_cid": ipfs_cid}
                else:
                    return {"status": "error", "message": f"ESMFold API error: {response.status_code}"}
            except Exception as e:
                return {"status": "error", "message": f"Folding failed: {e}"}
        
    def submit_result(self, task_id, result): 
        payload = {
            "task_id": task_id,
            "node_id": self.node_id,
            **result
        }
        try:
            response = requests.post(f"{self.server_url}/result", json=payload)
            if response.status_code == 200:
                logging.info(f"⬆️ Submitted result for Task {task_id} back to network.\n")
            else:
                logging.error(f"❌ Failed to submit result: {response.text}")
        except requests.exceptions.ConnectionError:
            logging.error("❌ Cannot connect to server to submit result.")
        
    def run(self):
        try:
            while True:
                task = self.fetch_task()
                if task:
                    result = self.process_task(task)
                    result['smiles'] = task.get('smiles')
                    result['task_type'] = task.get('type')
                    self.submit_result(task['task_id'], result)
                time.sleep(5)
        except KeyboardInterrupt:
            logging.info("Node shutting down.")

if __name__ == "__main__": 
    # Load .env file for Pinata credentials
    env_path = os.path.join(os.path.dirname(__file__), '..', 'autoresearch_module', '.env')
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.strip() and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip("'\"")

    # NOTE: Replace this with the actual CID of 'best_gnn_model.pt' from your ipfs_manifest.json
    YOUR_MODEL_CID = "QmYourActualCIDGoesHere..."
    
    node = Node(node_id=f"DP-Node-{random.randint(100,999)}", model_cid=YOUR_MODEL_CID)
    node.run()
