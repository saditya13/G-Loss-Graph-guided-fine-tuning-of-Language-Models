# G-Loss: Graph-inspired fine-tuning of Language Models

This repository contains code to train and evaluate BERT-based text classifiers using graph-inspired loss functions (G-Loss), supervised contrastive loss (SCL) and standard cross-entropy (CE). The codebase includes a "Combined with CE" supervised training implementation and a "Standalone" unsupervised/contrastive implementation.

## Highlights
- Implementations: G-Loss (graph-based), SCL (supervised contrastive), CE (cross-entropy).
- Training scripts with logging, checkpointing and Optuna hyperparameter search support.
- Example datasets and toy splits included under `data/` for quick runs.
## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Project layout (important files)

- `Combined with CE/` - Supervised training pipeline (CE + optional G-Loss or SCL components)
  - `main.py` - entry point for supervised experiments
  - `config.py` - CLI arguments and config defaults
  - `models.py`, `losses.py`, `training.py`, `utils.py` - core code for model, loss, and training
  - `supervised_checkpoint/` - training outputs and saved checkpoints
- `Standalone/` - Standalone/unsupervised or contrastive experiments
  - `main_unsupervised.py`, `training_unsupervised.py`, `losses_unsupervised.py`, etc.
- `data/` - example datasets and toy datasets. The expected dataset structure is described below.
  
## Usage

### Basic Training

Train a model with Cross-Entropy loss:
```bash
python main.py --dataset ohsumed --loss ce --bert_lr 1e-5
```

### Training with G-Loss

Train with Graph-based Loss:
```bash
python main.py --dataset ohsumed --loss gloss --lam 0.8 --gamma 0.7 --sigma 0.667 
```

### Training with Supervised Contrastive Loss

```bash
python main.py --dataset ohsumed --loss scl --temperature 0.3 --lam 0.9
```

### Hyperparameter Tuning with Optuna

```bash
python main.py --dataset ohsumed --loss gloss --tune --optuna_trials 20
```

### Using Computed Sigma

For G-Loss, you can automatically compute sigma using different methods:

```bash
# Using root method
python main.py --dataset ohsumed --loss gloss --sigmafn root
```

## Arguments

### General Arguments
- `--dataset`: Dataset name (choices: 20ng, R8, R52, ohsumed, MR)
- `--bert_init`: BERT model initialization (default: bert-base-uncased)
- `--max_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size (default: 128)
- `--nb_epochs`: Number of training epochs (default: 200)
- `--bert_lr`: Learning rate for BERT (default: None)

### Loss Function Arguments
- `--loss`: Loss function type (choices: ce, gloss, scl)
- `--lam`: Lambda parameter for G-Loss (default: None)
- `--gamma`: Gamma parameter for G-Loss (default: None)
- `--sigma`: Sigma parameter for G-Loss (default: None)
- `--temperature`: Temperature parameter for SCL (default: 0.3)
- `--sigmafn`: Sigma computation method (choices: mst, root)

### Optuna Arguments
- `--tune`: Enable hyperparameter tuning
- `--optuna_trials`: Number of Optuna trials (default: 15)
- `--optuna_sigma`: Fallback sigma value (default: None)
- `--optuna_storage`: Optuna storage URI (default: None)
- `--optuna_results`: JSON file for best parameters (default: optuna_best_params.json)

### Other Arguments
- `--checkpoint_dir`: Checkpoint directory (default: None)
- `--use_latest_checkpoint`: Use most recent checkpoint

## Data Format

The code expects data in the following format:
- CSV files with columns: `text`, `label`
- Directory structure:
  ```
  data/
  └── <dataset_name>/
      ├── train.csv
      ├── val.csv
      └── test.csv
  ```

## Output

Training outputs are saved in the checkpoint directory:
- `training.log`: Training logs
- `checkpoint.pth`: Model checkpoints
- `best_model.pth`: Best model based on validation F1
- `train_loss.png`: Training loss plot
- `val_f1.png`: Validation F1 plot
- `epoch_stats.csv`: Per-epoch statistics
- `loss_stats.csv`: Loss component statistics

## Module Descriptions

### models.py
Contains the `BertClassifier` class for text classification.

### losses.py
Implements:
- G-Loss (Graph-based Loss) with Label Propagation
- Supervised Contrastive Loss
- Helper functions for adjacency matrix normalization

### training.py
Contains training loop and evaluation functions.

### utils.py
Utility functions for:
- Data loading and preprocessing
- Tokenization and encoding
- Sigma computation
- Embedding extraction

### optuna_tuning.py
Hyperparameter optimization using Optuna framework.

## Citation

If you use this code, please cite the relevant papers for:
- Your paper (if applicable)

## License

[Add your license information here]
