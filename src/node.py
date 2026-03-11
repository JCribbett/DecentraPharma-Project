import time, random, logging
try: from rdkit import Chem; from rdkit.Chem import Descriptors
except ImportError: print("Install RDKit: pip install rdkit-pypi"); exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class Node:
    def __init__(self, node_id): self.node_id = node_id; logging.info(f"Node {self.node_id} ready.")
    def fetch_task(self): # Simulates fetching task (SMILES string)
        smiles_list = ["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C"]
        task = {"task_id": random.randint(1000, 9999), "smiles": random.choice(smiles_list)}
        logging.info(f"Fetched Task {task["task_id"]} for SMILES: {task["smiles"]}")
        return task
    def process_task(self, task): # Uses RDKit to calculate MW and LogP
        mol = Chem.MolFromSmiles(task['smiles'])
        if mol: mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol); logging.info(f"Task {task['task_id']} Done: MW={mw:.2f}, LogP={logp:.2f}"); return {"status": "success", "MW": mw, "LogP": logp}
        else: logging.error(f"Failed parsing SMILES: {task['smiles']}"); return {"status": "error"}
    def submit_result(self, task_id, result): logging.info(f"Submitted Task {task_id}.
")
    def run(self):
        try:
            while True:
                task = self.fetch_task()
                result = self.process_task(task)
                self.submit_result(task['task_id'], result)
                time.sleep(5)
        except KeyboardInterrupt:
            logging.info("Node shutting down.")

if __name__ == "__main__": Node(node_id=f"node_{random.randint(100,999)}").run()
