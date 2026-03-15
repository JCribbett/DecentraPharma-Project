# DecentraPharma Autoresearch — GNN Antiviral Research

## Objective
Maximize ROC-AUC on the HIV dataset (41,127 compounds) using Graph Neural Networks.
The goal is to surpass the MLP baseline of **0.8740 AUC** and reach **0.92+ AUC**.

## Background
- **Dataset:** MoleculeNet HIV — binary classification (active/inactive HIV inhibitors)
- **Class imbalance:** ~3.5% active compounds — use pos_weight in BCEWithLogitsLoss
- **Previous MLP results:** Best AUC = 0.8740 using 2048-bit Morgan fingerprints + MLP with dropout
- **Why GNNs:** Fingerprints lose spatial/structural information. GNNs learn directly from molecular graphs, capturing substructure patterns that fingerprints miss.

## Architecture
- **Input:** Molecular graph (atoms as nodes with 16 features, bonds as edges)
- **Model:** Graph Convolutional Network (GCN) with BatchNorm, skip connections
- **Pooling:** Concatenated global mean + max pooling
- **Classifier:** MLP head on pooled graph representation
- **Output:** Single logit → sigmoid → probability of HIV inhibition

## Agent Instructions
You may modify the code between `## START OF AGENT MODIFIABLE SECTION ##` and `## END OF AGENT MODIFIABLE SECTION ##` in `train_gnn.py`.
If you need to import new layers or modules, you can also modify the code between `## START OF AGENT IMPORTS ##` and `## END OF AGENT IMPORTS ##`.

### Things to try (in order of expected impact):
1. **Graph Attention Networks (GAT):** Replace GCNConv with GATConv for attention-weighted message passing
2. **Edge features:** Add bond type (single/double/aromatic) as edge attributes
3. **Deeper networks:** Try 4-5 GCN layers with residual connections
4. **Different pooling:** Try Set2Set or attention-based pooling instead of mean+max
5. **Learning rate schedules:** Cosine annealing, warm restarts
6. **Wider hidden dimensions:** 256 or 512 per layer
7. **Virtual node:** Add a virtual node connected to all atoms for global context
8. **Pre-training:** Consider contrastive pre-training on unlabeled molecules

### Evaluation
- Primary metric: **ROC-AUC** on 20% held-out validation set
- Each experiment runs for 15 minutes (TIME_BUDGET = 900s)
- Log results to `results_gnn.tsv`
- Keep changes that improve AUC, revert changes that don't

### Targets
| AUC Range | Assessment |
|-----------|-----------|
| < 0.87 | Below MLP baseline — revert |
| 0.87-0.90 | Marginal improvement |
| 0.90-0.92 | Good — GNN advantage confirmed |
| 0.92+ | Excellent — publish-worthy |
