from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import logging
import random
import csv
import os
import uvicorn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

app = FastAPI(title="DecentraPharma Central Dispatch")

# In-memory queue of AI-generated drug candidates
pending_tasks = [
    {"task_id": 1001, "type": "3d_docking", "smiles": "Cc1cn(C2CC(F)C(COC(=O)CCCCCCCCCCCN=[N+]=[N-])O2)c(=O)[nH]c1=O"}, # The Super AZT
    {"task_id": 1002, "type": "3d_docking", "smiles": "Cc1cn(C2CC(N=[N+]=[NH2+])C(COC(=O)CCCCCCCCCCCI)O2)c(=O)[nH]c1=O"}, # Iodine Tail
    {"task_id": 1003, "type": "2d_scoring", "smiles": "CSc1nc(N)c(NC2OC(CO)C(O)C2O)c(O)n1"}, # Nucleoside Analogue
    {"task_id": 1004, "type": "2d_scoring", "smiles": "Cc1cc(S(=O)(=O)Nc2nnc3cnc4ccccc4n23)c(S)cc1Cl"}
]
completed_tasks = []

CSV_FILE = "decentrapharma_results.csv"
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "node_id", "status", "hiv_probability", "docking_affinity", "ipfs_cid", "message", "smiles"])

class TaskResult(BaseModel):
    task_id: int
    node_id: str
    status: str
    hiv_probability: Optional[float] = None
    docking_affinity: Optional[float] = None
    ipfs_cid: Optional[str] = None
    message: Optional[str] = None
    smiles: Optional[str] = None

@app.get("/task")
def get_task():
    if not pending_tasks:
        # If the queue is empty, generate random placeholder tasks to keep the network busy
        return {
            "task_id": random.randint(5000, 9999),
            "type": random.choice(["2d_scoring", "3d_docking"]),
            "smiles": random.choice(["CC(=O)OC1=CC=CC=C1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"])
        }
    return pending_tasks.pop(0)

@app.post("/result")
def submit_result(result: TaskResult):
    completed_tasks.append(result.dict())
    logging.info(f"✅ Node {result.node_id} completed Task {result.task_id} ({result.status})")
    
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            result.task_id, 
            result.node_id, 
            result.status, 
            result.hiv_probability, 
            result.docking_affinity, 
            result.ipfs_cid, 
            result.message,
            result.smiles
        ])

    if result.docking_affinity:
        logging.info(f"   -> 🧲 Affinity: {result.docking_affinity} kcal/mol")
    if result.hiv_probability:
        logging.info(f"   -> 🎯 HIV Prob: {result.hiv_probability:.4f}")
    if result.ipfs_cid:
        logging.info(f"   -> 🏆 IPFS CID uploaded: {result.ipfs_cid}")
    return {"status": "success"}

if __name__ == "__main__":
    print("🚀 Starting DecentraPharma Central Dispatch Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")