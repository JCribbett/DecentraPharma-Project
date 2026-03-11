# Main entry point for CLI commands
import argparse

def main():
    parser = argparse.ArgumentParser(description="DecentraPharma CLI")
    # Add commands here, e.g.:
    # parser.add_argument("--run-node", action="store_true", help="Run a compute node")
    # parser.add_argument("--train-model", type=str, help="Train a specific model")

    args = parser.parse_args()

    print("Welcome to DecentraPharma CLI!")
    # Add logic to handle commands based on args
    # if args.run_node:
    #     from src.core.node import run_node
    #     run_node()

if __name__ == "__main__":
    main()
