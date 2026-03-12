import subprocess
import os

cwd = r"C:\Users\Jeffr\.nanobot\workspace\DecentraPharma\autoresearch_module"
with open(os.path.join(cwd, "run_gnn.log"), "w") as f:
    subprocess.Popen(["python", "train_gnn.py"], cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
print("Started train_gnn.py in background")
