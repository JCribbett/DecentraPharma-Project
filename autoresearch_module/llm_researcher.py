import os
import subprocess
import time
import re
import sys
#import google.generativeai as genai
import requests

# Note: You would install a library like `google-generativeai` or `openai` to power this.
# For this example, we'll use a placeholder structure for the LLM call.

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_TO_EDIT = os.path.join(SCRIPT_DIR, "train_gnn.py")
RESULTS_FILE = os.path.join(SCRIPT_DIR, "results_gnn.tsv")
PROGRAM_INSTRUCTIONS = os.path.join(SCRIPT_DIR, "program_gnn.md")

def load_env():
    env_path = os.path.join(SCRIPT_DIR, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip("'\"")

def get_file_content(filepath):
    with open(filepath, "r") as f:
        return f.read()

def run_git_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode

def extract_code_block(text):
    """Extracts python code from a markdown block."""
    match = re.search(r"```(?:python)?(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def call_llm(prompt, current_code):
    """
    Calls the LLM API using either Google Gemini directly or via OpenRouter.
    """
    print("\n[🧠] Asking LLM for the next experiment...")
    
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")

    if gemini_key:
        print("Using Gemini API...")
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel('gemini-1.5-pro')
        full_prompt = f"{prompt}\n\nHere is the current code:\n```python\n{current_code}\n```"
        try:
            response = model.generate_content(full_prompt)
            return extract_code_block(response.text)
        except Exception as e:
            print(f"⚠️ Error calling Gemini API: {e}")
            return current_code

    elif openrouter_key:
        print("Using OpenRouter API...")
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {openrouter_key}"},
                json={
                    "model": "google/gemini-3-pro-preview", # Using Claude 3.5 Sonnet (excellent for coding)
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"Here is the current code:\n```python\n{current_code}\n```"}
                    ]
                }
            )
            response.raise_for_status()
            data = response.json()
            return extract_code_block(data["choices"][0]["message"]["content"])
        except Exception as e:
            print(f"⚠️ Error calling OpenRouter API: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response details: {e.response.text}")
            return current_code

    print("⚠️ No API key found. Please set GEMINI_API_KEY or OPENROUTER_API_KEY.")
    return current_code # Placeholder

def extract_section(code, start_marker, end_marker):
    start_idx = code.find(start_marker)
    end_idx = code.find(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        return None
        
    return code[start_idx:end_idx + len(end_marker)]

def inject_section(full_code, new_section, start_marker, end_marker):
    start_idx = full_code.find(start_marker)
    end_idx = full_code.find(end_marker) + len(end_marker)
    
    if start_idx == -1 or end_idx == -1:
        return full_code
        
    return full_code[:start_idx] + new_section + full_code[end_idx:]

def get_best_score():
    if not os.path.exists(RESULTS_FILE):
        return -float('inf')
    try:
        with open(RESULTS_FILE, 'r') as f:
            lines = f.readlines()
        if len(lines) < 2: return -float('inf')
        
        scores = [float(line.split('\t')[1]) for line in lines[1:] if len(line.split('\t')) > 1]
        return max(scores) if scores else -float('inf')
    except Exception:
        return -float('inf')

def main():
    # Load environment variables from .env file if it exists
    load_env()
    
    print("🚀 Starting True LLM Autoresearch Loop...")
    
    # 1. Read the system prompt / instructions
    system_instructions = get_file_content(PROGRAM_INSTRUCTIONS)
    best_score = get_best_score()
    print(f"Current Best val_auc: {best_score}")
    
    # 2. Ensure git is clean
    stdout, returncode = run_git_command("git status --porcelain")
    if stdout:
        print("⚠️ Git working directory is not clean. Please commit or stash changes before running.")
        sys.exit(1)

    iteration = 1
    while True:
        print(f"\n" + "="*50)
        print(f"🔬 Iteration {iteration} (Best Score: {best_score})")
        
        # 3. Read current code and get LLM suggestion
        full_code = get_file_content(SCRIPT_TO_EDIT)
        
        imports_marker_start = "## START OF AGENT IMPORTS ##"
        imports_marker_end = "## END OF AGENT IMPORTS ##"
        mod_marker_start = "## START OF AGENT MODIFIABLE SECTION ##"
        mod_marker_end = "## END OF AGENT MODIFIABLE SECTION ##"
        
        imports_section = extract_section(full_code, imports_marker_start, imports_marker_end)
        modifiable_section = extract_section(full_code, mod_marker_start, mod_marker_end)
        
        # Read recent results to give context to the LLM
        recent_history = ""
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                recent_history = "".join(f.readlines()[-5:])

        prompt = f"""
        {system_instructions}
        
        Recent experiments history:
        {recent_history}
        
        Please provide the updated code for BOTH the imports section and the modifiable section.
        You must include the exact markers for both sections in your response.
        """
        
        current_context = f"{imports_section}\n\n...\n\n{modifiable_section}" if imports_section else modifiable_section
        
        llm_response = call_llm(prompt, current_context)
        
        new_imports_section = extract_section(llm_response, imports_marker_start, imports_marker_end)
        new_modifiable_section = extract_section(llm_response, mod_marker_start, mod_marker_end)
        
        if not new_modifiable_section:
            # Fallback if the LLM didn't use the markers, assume the whole response is the modifiable section
            new_modifiable_section = f"{mod_marker_start}\n{llm_response}\n{mod_marker_end}"
            
        if new_modifiable_section.strip() == modifiable_section.strip():
            print("⚠️ LLM returned identical code or an error occurred. Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # Replace and save the file
        updated_code = full_code
        if new_imports_section:
            updated_code = inject_section(updated_code, new_imports_section, imports_marker_start, imports_marker_end)
        updated_code = inject_section(updated_code, new_modifiable_section, mod_marker_start, mod_marker_end)
        
        with open(SCRIPT_TO_EDIT, "w") as f:
            f.write(updated_code)
            
        # 4. Commit the experiment (per program.md instructions)
        run_git_command(f"git add {SCRIPT_TO_EDIT}")
        run_git_command('git commit -m "Autoresearch iteration"')
        commit_hash, _ = run_git_command("git rev-parse --short HEAD")
        
        # 5. Run the code
        print(f"[⚙️] Running experiment {commit_hash}...")
        run_log_path = os.path.join(SCRIPT_DIR, "run.log")
        with open(run_log_path, "w") as log_file:
            result = subprocess.run([sys.executable, SCRIPT_TO_EDIT], stdout=log_file, stderr=subprocess.STDOUT, cwd=SCRIPT_DIR)
            
        # 6. Parse Results
        with open(run_log_path, "r") as log_file:
            log_content = log_file.read()
            
        match = re.search(r"val_auc:\s+([0-9.]+)", log_content)
        if result.returncode != 0 or not match:
            print("[❌] Script crashed or no val_auc found. Reverting...")
            run_git_command("git reset --hard HEAD~1") # Revert commit
            continue
            
        val_auc = float(match.group(1))
        print(f"[✅] Run completed. val_auc: {val_auc}")
        
        # 7. Evaluate and Advance/Revert
        if val_auc > best_score:
            print(f"🎉 NEW BEST! {val_auc} > {best_score}. Keeping commit.")
            best_score = val_auc
            # Script remains on this commit, serving as the new baseline
        else:
            print(f"📉 No improvement. Reverting commit.")
            run_git_command("git reset --hard HEAD~1")
            # File reverts back, ready for a new attempt
            
        iteration += 1
        time.sleep(2)

if __name__ == "__main__":
    main()