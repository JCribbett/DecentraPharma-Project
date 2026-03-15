import argparse
import urllib.parse

def generate_3dmol_url(pdb=None, url=None, ipfs_cid=None, file_type=None, style=None):
    base_url = "https://3dmol.csb.pitt.edu/viewer.html"
    params = []

    if pdb:
        params.append(('pdb', pdb))
        if ipfs_cid or url:
            # If we are docking, style the protein as a ribbon so we can see inside
            params.append(('style', 'cartoon'))
            params.append(('color', 'spectrum'))
        elif style:
            params.append(('style', style))
        else:
            params.append(('style', 'stick'))
            
    if ipfs_cid or url:
        if ipfs_cid:
            # Route through a public IPFS gateway
            target_url = f"https://ipfs.io/ipfs/{ipfs_cid}"
            if not file_type:
                file_type = 'sdf'
        else:
            target_url = url
            
        params.append(('url', target_url))
        
        if file_type:
            params.append(('type', file_type))
            
        if style and not pdb:
            params.append(('style', style))
        else:
            params.append(('style', 'stick'))

    query_string = urllib.parse.urlencode(params)
    return f"{base_url}?{query_string}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 3Dmol.js viewer URL.")
    parser.add_argument("--pdb", type=str, help="PDB ID (e.g., 3OXZ)")
    parser.add_argument("--ipfs", type=str, help="IPFS CID of your uploaded .sdf or .pdb file")
    parser.add_argument("--url", type=str, help="Direct HTTP URL to a molecule file")
    parser.add_argument("--type", type=str, help="File format (e.g., sdf, pdb)")
    parser.add_argument("--style", type=str, help="Visualization style (e.g., stick, cartoon, sphere)")
    
    args = parser.parse_args()
    
    if not any([args.pdb, args.ipfs, args.url]):
        print("Please provide at least one source: --pdb, --ipfs, or --url")
    else:
        viewer_url = generate_3dmol_url(args.pdb, args.url, args.ipfs, args.type, args.style)
        print(f"\n🔗 3Dmol.js Viewer URL:\n{viewer_url}\n")