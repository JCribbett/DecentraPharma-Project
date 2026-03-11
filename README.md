# DecentraPharma-Project 🧬

**FOSS decentralized network for AI-driven drug discovery.**

DecentraPharma aims to accelerate the discovery of novel therapeutic molecules by creating a collaborative, open-source platform. Volunteers contribute their compute power to train sophisticated AI models, while researchers worldwide can access and contribute to the discovery process.

## 🚀 Project Vision
*   **Democratize Drug Discovery:** Lower the barrier to entry for AI-driven pharmaceutical research.
*   **Accelerate Research:** Leverage distributed computing to speed up computationally intensive tasks like molecular modeling and simulation.
*   **Foster Collaboration:** Build an open ecosystem where researchers, developers, and citizen scientists can contribute.
*   **Open Science:** Promote transparency and reproducibility in drug discovery.

## 🏗️ Architecture Overview
DecentraPharma is designed with a modular architecture:

*   **Compute Nodes:** (e.g., `src/node.py`) These are the workhorses of the network. They fetch tasks (like molecular calculations or model training jobs), process them using libraries like RDKit, PyTorch, and DeepChem, and submit results.
*   **Data Layer:** (e.g., `src/core/data_handler.py`) Utilizes decentralized storage solutions like IPFS to host and share datasets (e.g., ChEMBL, PubChem) and trained models, ensuring data availability and integrity.
*   **AI Engine:** (e.g., `src/models/drug_discovery_model.py`) Consists of various AI models (GNNs, Transformers, etc.) implemented using frameworks like PyTorch and DeepChem, capable of tasks such as molecular property prediction, virtual screening, and *de novo* molecule generation.
*   **Cheminformatics Utilities:** (e.g., `src/utils/cheminformatics.py`) Provide essential molecular manipulation capabilities, including loading molecules, calculating descriptors, and generating fingerprints using RDKit.
*   **Orchestration Layer:** (Under development) Manages task distribution, node coordination, and result aggregation.
*   **CI/CD:** Automated testing and linting via GitHub Actions (`.github/workflows/main.yml`) ensures code quality and reliability.

## ✨ Key Features
*   **RDKit Integration:** Robust handling of molecular data for property calculation and fingerprint generation.
*   **IPFS Data Handling:** Enables decentralized storage and retrieval of large datasets and models.
*   **AI Model Framework:** Foundation for training and deploying various machine learning models for drug discovery tasks.
*   **Distributed Task Execution:** Designed to leverage distributed computing frameworks like Ray.
*   **Automated Testing:** CI pipeline with `flake8` and `pytest` for maintainability.

## 🛠️ Getting Started
### Prerequisites
*   Python 3.10+
*   `pip` package manager
*   (Optional) Docker for running components in containers
*   (Optional) IPFS daemon running locally if testing IPFS features (`http://127.0.0.1:5001`)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JCribbett/DecentraPharma-Project.git
    cd DecentraPharma-Project
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note: `rdkit-pypi` can sometimes be challenging to install. Ensure you have the necessary build tools if you encounter issues.*

### Running a Compute Node
A basic compute node example is available in `src/node.py`.

```bash
python src/node.py
```
This node will fetch simulated tasks (SMILES strings), calculate molecular weight and LogP using RDKit, and log the results.

### Running Tests
Ensure your virtual environment is activated and run pytest:
```bash
pytest
```

### Docker
Build and run the Docker container:
```bash
docker build -t decentrapharma .
docker run decentrapharma
```

## 📚 Contribution
We welcome contributions! Whether it's implementing new AI models, improving the orchestration layer, adding datasets, or refining existing features.

*   **Found Issues?** Please report them in the [Issues](https://github.com/JCribbett/DecentraPharma-Project/issues) section.
*   **Want to Help?** Look for issues tagged `good first issue` or `help wanted`.
*   **Submitting Code:** Please follow the TDD (Test-Driven Development) approach where applicable. Ensure your code passes linting (`flake8`) and all tests (`pytest`). Submit your changes via Pull Requests.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact
For questions or inquiries, please open an issue on GitHub.
