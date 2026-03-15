import subprocess
import os
import re
import pandas as pd
import time
import sys
import random
import shutil

# --- Optimization Parameters ---
MAX_ITERATIONS = 100 # Set a limit to prevent infinite loops
MUTATION_RANGE = 0.2 # +/- 20% for numerical parameters

def get_best_score(results_file):
    if not os.path.exists(results_file):
        return -1.0
    try:
        df = pd.read_csv(results_file, sep='\t')
        if not df.empty and 'val_auc' in df.columns:
            return df['val_auc'].max()
        else:
            return -1.0
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        return -1.0

def mutate_script(script_path, mutation_range):
    try:
        with open(script_path, 'r') as f:
            content = f.read()

        pattern = re.compile(r'([A-Z_]+)\s*=\s*(\d+\.?\d*e?-?\d*)')

        def replace_number(match):
            var_name = match.group(1)
            value_str = match.group(2)
            
            try:
                value = float(value_str)
                if value == 0: return match.group(0) 
                    
                # Use random module for non-deterministic mutation
                mutation_factor = 1 + random.uniform(-mutation_range, mutation_range)
                mutated_value = value * mutation_factor
                
                if '.' in value_str or 'e' in value_str.lower():
                    if 'e' in value_str.lower():
                         return f"{var_name} = {mutated_value:.6e}"
                    else:
                         return f"{var_name} = {mutated_value:.6f}"
                else:
                    return f"{var_name} = {int(round(max(1, mutated_value)))}"
            except ValueError:
                return match.group(0)

        new_content = pattern.sub(replace_number, content)
        
        if new_content == content:
            print("Warning: No numerical parameters found or mutated.")
            return False

        with open(script_path, 'w') as f:
            f.write(new_content)
        return True

    except Exception as e:
        print(f"Error during script mutation: {e}")
        return False

def run_training(script_path):
    try:
        # Use the same python executable that is running this optimizer script
        command = [sys.executable, script_path]
        print(f"Executing command: {' '.join(command)}")
        
        process = subprocess.run(command, check=True, capture_output=True, text=True, cwd=os.path.dirname(script_path))
        print("Training script finished successfully.")
        print(process.stdout)
        if process.stderr:
            print("--- STDERR ---")
            print(process.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training script execution (return code {e.returncode}):")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred during training execution: {e}")
        return False

def main(target_script, results_file):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_full_path = os.path.join(script_dir, target_script)
    backup_script_path = script_full_path + ".bak"

    if not os.path.exists(script_full_path):
        print(f"Error: Target training script '{script_full_path}' not found.")
        return

    # Create a backup of the original script
    shutil.copy(script_full_path, backup_script_path)
    print(f"Created backup of '{target_script}' to '{os.path.basename(backup_script_path)}'.")

    current_best_score = get_best_score(results_file)
    print(f"Initial best score ({results_file}): {current_best_score:.4f}")

    try:
        for iteration in range(MAX_ITERATIONS):
            print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")

            print("Mutating script...")
            if not mutate_script(script_full_path, MUTATION_RANGE):
                print("Mutation failed. Retrying...")
                time.sleep(5)
                continue

            if not run_training(script_full_path):
                print("Training run failed. Reverting to previous best version.")
                shutil.copy(backup_script_path, script_full_path)
                continue

            print("Training finished. Evaluating results...")
            new_score = get_best_score(results_file)
            print(f"Previous best score: {current_best_score:.4f}, New score from this run: {new_score:.4f}")

            if new_score > current_best_score:
                print("🚀 Improvement found! Keeping mutation.")
                current_best_score = new_score
                shutil.copy(script_full_path, backup_script_path)
                print("Updated backup with new best script.")
            else:
                print("📉 No improvement. Reverting to previous best version.")
                shutil.copy(backup_script_path, script_full_path)

            time.sleep(2)
    finally:
        print("\nOptimization finished. Restoring original script.")
        shutil.copy(backup_script_path, script_full_path)
        os.remove(backup_script_path)
        print(f"Final best score ({results_file}): {current_best_score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python optimizer.py <target_script.py> <results_file.tsv>")
        sys.exit(1)
    
    target_script_arg = sys.argv[1]
    results_file_arg = sys.argv[2]
    
    main(target_script_arg, results_file_arg)