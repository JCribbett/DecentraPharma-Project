import subprocess
import os
import re
import pandas as pd
import time

# --- Configuration ---
TARGET_TRAINING_SCRIPT = "train_hiv.py"
RESULTS_FILE = "results_hiv.tsv"
SCRIPT_DIR = "autoresearch_module" # Directory where train_hiv.py and the optimizer script reside

# --- Optimization Parameters ---
MAX_ITERATIONS = 100 # Set a limit to prevent infinite loops
MUTATION_RANGE = 0.2 # +/- 20% for numerical parameters

def get_best_score(results_file):
    if not os.path.exists(results_file):
        return -1.0
    try:
        # Read the TSV file
        df = pd.read_csv(results_file, sep='\t')
        if not df.empty:
            # Assuming the score is the last column and is named 'val_auc'
            # Adjust column name if necessary based on actual logging format
            if 'val_auc' in df.columns:
                return df['val_auc'].iloc[-1]
            elif 'combined_score' in df.columns: # Fallback for combined score
                return df['combined_score'].iloc[-1]
            else: # Try to get the last column if specific name not found
                return df.iloc[-1, -1]
        else:
            return -1.0
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        return -1.0

def mutate_script(script_path, mutation_range):
    try:
        with open(script_path, 'r') as f:
            content = f.read()

        # Regex to find capitalized variables assigned to numbers (float or int)
        # This regex targets lines like:
        # CAPITALIZED_VAR = 1.23e-4
        # CAPITALIZED_VAR = 100
        # It captures the variable name (group 1) and the value (group 2)
        pattern = re.compile(r'([A-Z_]+)\s*=\s*(\d+\.?\d*e?-?\d*)')

        def replace_number(match):
            var_name = match.group(1)
            value_str = match.group(2)
            
            try:
                # Try converting to float first
                value = float(value_str)
                
                # Apply mutation
                if value == 0: # Avoid mutating zero if it's meaningful, or handle as needed
                    return match.group(0) 
                    
                mutation = value * mutation_range
                mutated_value = value + (value * mutation_range * (2 * (hash(var_name + value_str) % 100) / 100.0 - 1)) # Randomness based on hash
                
                # Preserve original format (int or float)
                if '.' in value_str:
                    # Check if original was scientific notation
                    if 'e' in value_str.lower():
                         return f"{var_name} = {mutated_value:.6e}" # Format as scientific notation
                    else:
                         return f"{var_name} = {mutated_value:.6f}" # Format as float
                else:
                    return f"{var_name} = {int(round(max(1, mutated_value)))}" # Ensure it stays at least 1 for integers like batch size
            except ValueError:
                # If conversion fails, return original match (e.g., it's a string or complex type)
                return match.group(0)

        # Apply mutation to the script content
        new_content = pattern.sub(replace_number, content)
        
        # Check if any changes were made
        if new_content == content:
            print("Warning: No numerical parameters found or mutated.")
            return False # Indicate no change was made

        # Save the mutated script temporarily
        mutated_script_path = script_path + ".mutated"
        with open(mutated_script_path, 'w') as f:
            f.write(new_content)
        return True # Indicate change was made

    except FileNotFoundError:
        print(f"Error: Script file not found at {script_path}")
        return False
    except Exception as e:
        print(f"Error during script mutation: {e}")
        return False

def run_training(script_path):
    try:
        # Execute the training script. Use working_dir to ensure it finds other modules.
        # Use shell=True cautiously. Ensure script_path is safe.
        command = f"python {script_path}"
        print(f"Executing command: {command}")
        
        # Use Popen for non-blocking execution if needed, but here we wait for it.
        # Capturing output can be complex with long-running scripts.
        # For simplicity, we'll let it print to stdout/stderr.
        process = subprocess.run(command, shell=True, check=True, cwd=SCRIPT_DIR)
        print("Training script finished successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during training script execution: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during training execution: {e}")
        return False

def main():
    script_full_path = os.path.join(SCRIPT_DIR, TARGET_TRAINING_SCRIPT)
    backup_script_path = script_full_path + ".bak"

    # --- Initial Setup ---
    if not os.path.exists(script_full_path):
        print(f"Error: Target training script '{script_full_path}' not found.")
        return

    # Create a backup of the original script if it doesn't exist
    if not os.path.exists(backup_script_path):
        try:
            with open(script_full_path, 'r') as f_orig, open(backup_script_path, 'w') as f_bak:
                f_bak.write(f_orig.read())
            print(f"Created backup of '{script_full_path}' to '{backup_script_path}'.")
        except Exception as e:
            print(f"Error creating backup: {e}")
            return

    current_best_score = get_best_score(RESULTS_FILE)
    print(f"Initial best score ({RESULTS_FILE}): {current_best_score:.4f}")

    # --- Optimization Loop ---
    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1}/{MAX_ITERATIONS} ---")

        # 1. Mutate the script
        print("Mutating script...")
        if not mutate_script(script_full_path, MUTATION_RANGE):
            print("Mutation failed or no changes made. Skipping this iteration.")
            time.sleep(5) # Wait a bit before retrying or stopping
            continue

        mutated_script_path = script_full_path + ".mutated"

        # 2. Run the mutated script
        print(f"Running training with mutated script: {TARGET_TRAINING_SCRIPT}...")
        if not run_training(mutated_script_path):
            print("Training run failed. Reverting to backup.")
            # Revert to backup if training fails
            try:
                with open(script_full_path, 'w') as f_orig, open(backup_script_path, 'r') as f_bak:
                    f_orig.write(f_bak.read())
                print("Reverted to backup script.")
            except Exception as e:
                print(f"Error reverting to backup: {e}")
            os.remove(mutated_script_path) # Clean up mutated file
            continue # Skip to next iteration

        # 3. Get the new score
        print("Training finished. Evaluating results...")
        new_score = get_best_score(RESULTS_FILE)
        print(f"Current best score: {current_best_score:.4f}, New score: {new_score:.4f}")

        # 4. Decide whether to keep the mutation
        if new_score > current_best_score:
            print("Improvement found! Keeping mutation.")
            current_best_score = new_score
            
            # Overwrite the original script with the mutated version
            try:
                os.replace(mutated_script_path, script_full_path)
                # Also update the backup to reflect the new "best" state
                with open(backup_script_path, 'w') as f_bak:
                    with open(script_full_path, 'r') as f_orig:
                        f_bak.write(f_orig.read())
                print("Updated main script and backup.")
            except Exception as e:
                print(f"Error updating script/backup: {e}")
        else:
            print("No improvement or score decreased. Reverting to backup.")
            # Revert to backup
            try:
                with open(script_full_path, 'w') as f_orig, open(backup_script_path, 'r') as f_bak:
                    f_orig.write(f_bak.read())
                print("Reverted to backup script.")
            except Exception as e:
                print(f"Error reverting to backup: {e}")
            
            # Clean up the mutated script file
            if os.path.exists(mutated_script_path):
                os.remove(mutated_script_path)

        # Optional: Add a small delay between iterations
        time.sleep(2)

    print("\nOptimization finished.")
    print(f"Final best score ({RESULTS_FILE}): {current_best_score:.4f}")

if __name__ == "__main__":
    main()
