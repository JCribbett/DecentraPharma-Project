# DecentraPharma Autoresearch — GNN Toxicity Prediction

## Objective
Maximize ROC-AUC on the Tox21 dataset. The goal is to create the most accurate "Oracle" possible for predicting whether a compound is toxic to human cells.

## Background
- **Dataset:** MoleculeNet Tox21 — simplified to a binary classification (toxic/non-toxic). A molecule is labeled 'toxic' if it is active in any of the 12 assay targets.
- **Class imbalance:** The dataset is imbalanced. Use pos_weight in BCEWithLogitsLoss.
- **Why GNNs:** We want to learn the specific chemical substructures (toxicophores) that cause cellular toxicity.

## Architecture
- **Input:** Molecular graph (atoms as nodes, bonds as edges).
- **Model:** Graph Convolutional Network (GCN) with global pooling.
- **Classifier:** MLP head on the pooled graph representation.
- **Output:** Single logit → sigmoid → probability of toxicity.

## Agent Instructions
You may modify the code between `## START OF AGENT MODIFIABLE SECTION ##` and `## END OF AGENT MODIFIABLE SECTION ##` in the target training script.
If you need to import new layers or modules, you can also modify the code between `## START OF AGENT IMPORTS ##` and `## END OF AGENT IMPORTS ##`.

### Things to try:
1.  **Graph Attention (GAT/GATv2):** Focus on specific atoms or bonds that are strong indicators of toxicity.
2.  **Deeper/Wider Networks:** Explore more complex architectures to capture subtle toxicophore patterns.
3.  **Advanced Pooling:** Try Set2Set or GlobalAttention pooling.
4.  **Focal Loss:** Since this is an imbalanced dataset, Focal Loss might outperform weighted BCE.

### Evaluation
- Primary metric: **ROC-AUC** on the validation set.
- Each experiment runs for a fixed time budget.
- Log results to the specified results file.
- Keep changes that improve AUC, revert changes that don't.