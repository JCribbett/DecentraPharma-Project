# DecentraPharma Autoresearch: Antiviral Discovery

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for autonomous AI drug discovery research.

## Overview

This is an experiment to have an AI agent autonomously research and optimize molecular property prediction models for **Antiviral Drug Discovery**. 

The task is **HIV Inhibition** classification — predicting whether a molecule can inhibit HIV replication based on the MoleculeNet HIV dataset (~41,000 compounds).

**Note on Class Imbalance**: The dataset is highly imbalanced (~3.5% active). The baseline `train.py` uses a weighted loss (`pos_weight`) to account for this.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `antiviral_mar12`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**:
   - `program.md` — this file. Research context and instructions.
   - `prepare.py` — fixed constants, data prep for HIV dataset, molecular featurization, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify dependencies**: Ensure `rdkit`, `pandas`, `scikit-learn`, `torch`, and `requests` are installed.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU (or CPU if no GPU available). The training script runs for a **fixed time budget of 5 minutes** (wall clock training time). You launch it simply as: `python train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything within the AGENT MODIFIABLE SECTION is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, activation functions, regularization, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading for the HIV dataset, molecular featurization (2048-bit Morgan Fingerprints), and training constants.
- Install new packages or add dependencies beyond what's already available.
- Modify the evaluation harness. The `evaluate_metric` function in `prepare.py` is the ground truth metric (ROC-AUC).

**The goal is simple: get the highest val_auc.** Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything in the modifiable section is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Research directions for Antiviral Discovery:**
- Model architecture: MLP depth/width, skip connections, batch normalization, layer normalization.
- Activation functions: ReLU, GELU, SiLU/Swish, LeakyReLU, Mish.
- Regularization: Dropout rates, weight decay, label smoothing (crucial for imbalanced data).
- Optimizers: Adam, AdamW, SGD with momentum, learning rate schedules (cosine annealing, warmup).
- Handling imbalance: Experiment with different `pos_weight` values, focal loss, or oversampling techniques within the training loop.
- Advanced architectures: Attention mechanisms over fingerprint bits, residual networks.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_auc:          0.785000
training_seconds: 300.1
num_steps:        45000
num_params:       395265
```

You can extract the key metric from the log file:

```
grep "^val_auc:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated).

The TSV has a header row and 5 columns:

```
commit	val_auc	num_params	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.785000) — use 0.000000 for crashes
3. num_params (total model parameters) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_auc	num_params	status	description
a1b2c3d	0.785000	395265	keep	baseline MLP (512-256) with pos_weight
b2c3d4e	0.792000	920577	keep	deeper MLP (512-256-128) with batch norm
c3d4e5f	0.781000	395265	discard	switched to GELU activation
d4e5f6g	0.000000	0	crash	attention layer OOM
```

## The experiment loop

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune `train.py` with an experimental idea by modifying the AGENT MODIFIABLE SECTION
3. git commit
4. Run the experiment: `python train.py > run.log 2>&1`
5. Read out the results: `grep "^val_auc:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
7. Record the results in the tsv
8. If val_auc improved (higher), you "advance" the branch, keeping the git commit
9. If val_auc is equal or worse, you git reset back to where you started

**NEVER STOP**: Once the experiment loop has begun, do NOT pause. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical architectural changes. The loop runs until manually interrupted.
