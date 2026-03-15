import os
import platform
import subprocess
import requests

def setup_smina(script_dir):
    system = platform.system()
    # Smina does not have a native Windows binary, so we use the Linux static binary via WSL
    smina_bin = "smina.osx" if system == "Darwin" else "smina.static"
    smina_path = os.path.join(script_dir, smina_bin)
    
    if not os.path.exists(smina_path):
        print(f"Downloading Smina docking engine...")
        # Direct mirror URL bypasses the SourceForge HTML interstitial page
        url = f"https://downloads.sourceforge.net/project/smina/{smina_bin}"
            
        response = requests.get(url, allow_redirects=True)
        if response.status_code == 200 and len(response.content) > 1000000: # Check if it's the actual binary
            with open(smina_path, 'wb') as f:
                f.write(response.content)
            if system != "Windows":
                os.chmod(smina_path, 0o755)
            print("✅ Smina downloaded successfully.\n")
        else:
            print("❌ Download failed. SourceForge might be blocking the request.")
            return None
        
    return smina_path

def run_docking(receptor_file, ligand_file, output_file, script_dir):
    project_dir = os.path.dirname(script_dir)
    smina_path = setup_smina(script_dir)
    if not smina_path: return
    
    print(f"--- Running Smina Molecular Docking ---")
    print(f"Receptor Protein: {os.path.basename(receptor_file)}")
    print(f"Ligand: {os.path.basename(ligand_file)}")
    
    # Convert absolute paths to relative paths with forward slashes for Linux/WSL compatibility
    def rel_path(p):
        return os.path.relpath(p, project_dir).replace("\\", "/")
        
    smina_rel = rel_path(smina_path)
    receptor_rel = rel_path(receptor_file)
    ligand_rel = rel_path(ligand_file)
    output_rel = rel_path(output_file)

    if platform.system() == "Windows":
        try:
            # Check if WSL is available
            subprocess.run(["wsl", "echo", "test"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("\n❌ Windows Subsystem for Linux (WSL) is required but not found.")
            print("Smina does not have a native Windows executable.")
            print("To fix this, open PowerShell as Administrator and run: wsl --install")
            print("Once installed, restart your computer and try again.")
            return
            
        wsl_command = f"chmod +x ./{smina_rel} && ./{smina_rel} --receptor {receptor_rel} --ligand {ligand_rel} --autobox_ligand {receptor_rel} --exhaustiveness 8 --out {output_rel}"
        cmd = ["wsl", "sh", "-c", wsl_command]
    else:
        cmd = [
            f"./{smina_rel}",
            "--receptor", receptor_rel,
            "--ligand", ligand_rel,
            "--autobox_ligand", receptor_rel, 
            "--exhaustiveness", "8",
            "--out", output_rel
        ]
    
    print("Executing physics simulation... (This may take a minute or two)")
    process = subprocess.run(cmd, capture_output=True, text=True, cwd=project_dir)
    
    best_score = None
    if process.returncode == 0:
        print(f"\n✅ Docking complete! Results saved to {os.path.basename(output_file)}")
        print("\n--- Top Binding Affinities (kcal/mol) ---")
        # Filter the output to just show the scoring table
        for line in process.stdout.split('\n'):
            if line.strip().startswith(("mode", "---", "1 ", "2 ", "3 ")):
                print(line)
            if line.strip().startswith("1 "):
                parts = line.split()
                if len(parts) >= 2:
                    best_score = float(parts[1])
    else:
        print("❌ Docking failed.")
        print(process.stderr)
        
    return best_score

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    
    run_docking(
        receptor_file=os.path.join(script_dir, "3oxz.pdb"),
        ligand_file=os.path.join(project_dir, "super_azt.sdf"),
        output_file=os.path.join(project_dir, "docked_super_azt.sdf"),
        script_dir=script_dir
    )