import os
import sys
import time
import subprocess
import random
import re
import shutil

# Files
SCRIPT_NAME = "train_trading.py" if "trading" in os.getcwd() else "train_gnn.py"
RESULTS_FILE = "results.tsv" if "trading" in os.getcwd() else "results_gnn.tsv"
BACKUP_FILE = SCRIPT_NAME + ".bak"

def get_best_score():
    if not os.path.exists(RESULTS_FILE):
        return -float('inf')
    try:
        with open(RESULTS_FILE, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2: return -float('inf')
        # Trading uses R2 (col 1), Pharma uses AUC (col 1)
        scores = []
        for line in lines[1:]:
            parts = line.split('\t')
            if len(parts) > 1:
                try:
                    scores.append(float(parts[1]))
                except:
                    pass
        return max(scores) if scores else -float('inf')
    except:
        return -float('inf')

def mutate_script():
    with open(SCRIPT_NAME, 'r') as f:
        content = f.read()
    
    # Find modifiable section
    # Simple regex to find numbers and tweak them
    # We look for patterns like "LEARNING_RATE = 0.001" or "n_estimators = 100"
    
    # Mutate Learning Rate
    def float_replacer(match):
        val = float(match.group(1))
        if random.random() < 0.5:
            new_val = val * (1 + random.uniform(-0.2, 0.2))
        else:
            new_val = val
        return f"{match.group(0).split('=')[0]}= {new_val:.2e}"

    # Mutate Integers (like estimators, dimensions)
    def int_replacer(match):
        val = int(match.group(1))
        if random.random() < 0.5:
            new_val = int(val * (1 + random.uniform(-0.2, 0.2)))
            new_val = max(1, new_val)
        else:
            new_val = val
        return f"{match.group(0).split('=')[0]}= {new_val}"

    # Apply mutations to capitalized constants (common convention)
    content = re.sub(r'([A-Z_]+)\s*=\s*(\d+\.\d+e?-?\d*)', float_replacer, content)
    content = re.sub(r'([A-Z_]+)\s*=\s*(\d+)', int_replacer, content)

    with open(SCRIPT_NAME, 'w') as f:
        f.write(content)

def restore_backup():
    if os.path.exists(BACKUP_FILE):
        shutil.copy(BACKUP_FILE, SCRIPT_NAME)
        print("[-] Reverted to previous best.")

def create_backup():
    shutil.copy(SCRIPT_NAME, BACKUP_FILE)
    print("[+] New best! Backed up script.")

def main():
    print(f"Starting Auto-Research on {SCRIPT_NAME}...")
    best_score = get_best_score()
    print(f"Current Best Score: {best_score}")
    
    # Initial backup
    if not os.path.exists(BACKUP_FILE):
        create_backup()

    while True:
        print("\n" + "="*40)
        print(f"Starting Experiment. Best: {best_score:.4f}")
        
        # 1. Mutate
        shutil.copy(BACKUP_FILE, SCRIPT_NAME) # Start from best
        mutate_script()
        
        # 2. Run
        start_time = time.time()
        try:
            # Use the current python executable
            cmd = [sys.executable, SCRIPT_NAME]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        except Exception as e:
            print(f"Execution failed: {e}")
            restore_backup()
            continue
            
        # 3. Evaluate
        current_score = get_best_score() # The script appends to results.tsv
        
        # Check if the last run actually produced the new max
        # We read the last line of results
        try:
            with open(RESULTS_FILE, 'r') as f:
                last_line = f.readlines()[-1]
                last_run_score = float(last_line.split('\t')[1])
        except:
            last_run_score = -float('inf')

        if last_run_score > best_score:
            print(f"🚀 IMPROVEMENT: {last_run_score:.4f} > {best_score:.4f}")
            best_score = last_run_score
            create_backup()
        else:
            print(f"📉 No improvement: {last_run_score:.4f} <= {best_score:.4f}")
            # We don't need to restore immediately, because we overwrite from backup at start of loop
            
        time.sleep(2)

if __name__ == "__main__":
    main()
